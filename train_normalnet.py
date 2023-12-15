import datetime
import re
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import os
from os.path import join
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.hub import download_url_to_file

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out

class Mixed_6a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Mixed_7a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


def load_weights(mdl, name):
    """Download pretrained state_dict and load into model.

    Arguments:
        mdl {torch.nn.Module} -- Pytorch model.
        name {str} -- Name of dataset that was used to generate pretrained state_dict.

    Raises:
        ValueError: If 'pretrained' not equal to 'vggface2' or 'casia-webface'.
    """
    if name == 'vggface2':
        path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt'
    elif name == 'casia-webface':
        path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt'
    else:
        raise ValueError('Pretrained models only exist for "vggface2" and "casia-webface"')

    model_dir = os.path.join(get_torch_home(), 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)

    cached_file = os.path.join(model_dir, os.path.basename(path))
    if not os.path.exists(cached_file):
        download_url_to_file(path, cached_file)

    state_dict = torch.load(cached_file)
    mdl.load_state_dict(state_dict)

def get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(
            'TORCH_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')
        )
    )
    return torch_home

class InceptionResnetV1(nn.Module):
    """Inception Resnet V1 model with optional loading of pretrained weights.

    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.

    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            (default: {None})
        classify {bool} -- Whether the model should output classification probabilities or feature
            embeddings. (default: {False})
        num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """
    def __init__(self, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6):
        super().__init__()

        # Set simple attributes
        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes

        if pretrained == 'vggface2':
            tmp_classes = 8631
        elif pretrained == 'casia-webface':
            tmp_classes = 10575
        elif pretrained is None and self.classify and self.num_classes is None:
            raise Exception('If "pretrained" is not specified and "classify" is True, "num_classes" must be specified')


        # Define layers
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

        if pretrained is not None:
            self.logits = nn.Linear(512, tmp_classes)
            load_weights(self, pretrained)

        if self.classify and self.num_classes is not None:
            self.logits = nn.Linear(512, self.num_classes)

        self.mean, self.d = 127.5, 128.

    def forward(self, x):
        """Calculate embeddings or logits given a batch of input image tensors.

        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.

        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """

        x = ((x*255 - self.mean) / self.d).contiguous()

        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        if self.classify:
            x = self.logits(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x





class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def save(self, model_path):
        state_dict = self.state_dict().copy()
        torch.save(state_dict, model_path)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.InstanceNorm2d, 
                 padding_type='reflect', last_op=nn.Tanh()):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if last_op is not None:
            model += [last_op]        
        self.model = nn.Sequential(*model)


        self.model.apply(weights_init)

    def save(self, model_path):
        state_dict = self.state_dict().copy()
        torch.save(state_dict, model_path)
            
    def forward(self, x):
        return self.model(x)



class DeepHuman(Dataset):
    def __init__(self, data_path, ratio, index_file, if_train, res=512):
        self.dires = np.array(sorted(glob(join(data_path, "*/*"))))
        self.res = res
        if os.path.exists(index_file):
            index = np.loadtxt(index_file, dtype=np.int32)
            self.dires = self.dires[index]
        else:
            index = np.arange(len(self.dires)).astype(np.int32)
            np.random.shuffle(index)
            self.dires = self.dires[index]
            np.savetxt(index_file, index, fmt='%d')

        num = int(len(self.dires) * ratio)
        if if_train:
            self.dires = self.dires[:num]
        else:
            self.dires = self.dires[num:]

        # self.coco = np.array(glob('/home/public/coco2017/images/*.jpg'))


    def __len__(self):
        return len(self.dires)

    def __getitem__(self, idx):
        img = cv2.imread(join(self.dires[idx], 'color.jpg'))
        normal = cv2.imread(join(self.dires[idx], 'ground_rendered_normal.png'))

        extra_data = np.load(join(self.dires[idx], 'extra.npy'), allow_pickle=True).item()

        bbox = extra_data['bbox'].astype(np.int32)
        min_x, min_y, max_x, max_y = bbox

        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        max_x = min(max_x, img.shape[1])
        max_y = min(max_y, img.shape[0])

        center_x = int((min_x+max_x)/2)
        h,w,_ = img.shape
        min_x = center_x - h//2
        max_x = center_x + h//2

        img = img[:, min_x:max_x]
        img = cv2.resize(img, (self.res, self.res))

        # chance = np.random.rand()
        # if chance > 0.2:
        #     bg = cv2.imread(np.random.choice(self.coco))
        #     bg = cv2.resize(bg, (self.res, self.res))
        #     idx = np.where((normal[:,:,0]>0)&(normal[:,:,1]>0)&(normal[:,:,2]>0))
        #     bg[idx] = img[idx]
        #     img = bg
        img = (img / 255.).transpose(2,0,1).astype(np.float32)

        normal = (normal / 255.).transpose(2,0,1).astype(np.float32)

        return img, normal


def mod(matched):
    value = int(matched.group('value'))
    return '100k'+str((value+10)%20)

class RenderPeople(Dataset):
    def __init__(self, data_path, res=512, ratio=0.9, back=False, if_train=True):
        self.imgs = sorted(glob(join(data_path, 'image/*.png')))
        if if_train:
            self.imgs = self.imgs[:int(ratio*len(self.imgs))]
        else:
            self.imgs = self.imgs[int(ratio*len(self.imgs)):]
        self.normals = []
        self.res = res
        self.back = back

        self.coco = np.array(glob('/home/public/coco2017/images/*.jpg'))

        for img_name in self.imgs:
            name = img_name.split('/')[-1][:-4]
            if back:
                name = re.sub(r'100k(?P<value>\d+)', mod, name)
            self.normals.append(join(data_path, 'normal/%s0001.png'%name))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx], -1)
        alpha = img[:,:,3]
        img = img[:,:,:3]
        normal = cv2.imread(self.normals[idx])
        if self.back:
            normal = cv2.flip(normal, 1)

        # assert normal is not None and img is not None

        index = np.where(alpha==0)
        normal[index] = 0

        img = cv2.resize(img, (self.res, self.res))
        normal = cv2.resize(normal, (self.res, self.res))

        bg = cv2.imread(np.random.choice(self.coco))
        bg = cv2.resize(bg, (self.res, self.res))
        index = np.where((normal[:,:,0]>0)&(normal[:,:,1]>0)&(normal[:,:,2]>0))
        bg[index] = img[index]
        img = bg

        img = (img / 255.).transpose(2,0,1).astype(np.float32)
        normal = (normal / 255.).transpose(2,0,1).astype(np.float32)

        return img, normal




class Trainer(object):
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        self.model = GlobalGenerator(3,3).cuda()
        # self.dataset = DeepHuman('/home/llx/deephuman/dataset', 1, 'data/shuffle_index.txt', True)
        self.dataset = RenderPeople('/home/llx/renderpeople/ortho', 512)
        self.loader = DataLoader(self.dataset, batch_size=16, shuffle=True, num_workers=4)
        self.testloader = DataLoader(self.dataset, batch_size=1, shuffle=True)

        self.inception = InceptionResnetV1('vggface2').cuda()
        self.inception.requires_grad_(False)
        self.inception = nn.DataParallel(self.inception)
        self.inception.eval()

        self.optimizer = Adam(self.model.parameters(), 0.0002)

        self.model = nn.DataParallel(self.model)

    def train(self):
        import torch
        now = datetime.datetime.now()
        writer = SummaryWriter(os.path.join('logs', now.strftime('%Y-%m-%d-%H-%M')))
        output_dir = os.path.join('trained_model', now.strftime('%Y-%m-%d-%H-%M'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.model.train()

        global_idx = 0
        for i in tqdm(range(300)):
            for idx, data in enumerate(self.loader):
                data = [item.cuda() for item in data]
                img, normal = data

                out = self.model(img)
                mask = (normal[:,0] > 0) | (normal[:,1] > 0) | (normal[:,2] > 0)
                mask = mask.unsqueeze(1)

                normal_loss = 5 * F.l1_loss(normal*mask, out*mask, reduction='sum') / mask.sum()
                inception_loss = F.mse_loss(self.inception(normal), self.inception(out))
                loss = normal_loss + inception_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                writer.add_scalar('normal_loss', normal_loss.item(), global_idx)
                writer.add_scalar('inception_loss', inception_loss.item(), global_idx)
                global_idx += 1


            self.model.eval()

            for idx, data in enumerate(self.testloader):
                data = [item.cuda() for item in data]
                img, normal = data
                out = self.model(img)
                normal = normal.squeeze().detach().cpu().numpy()
                out = out.squeeze().detach().cpu().numpy()
                img = img.squeeze().detach().cpu().numpy()
                writer.add_image('img', img[::-1], dataformats='CHW', global_step=i)
                writer.add_image('ori', normal, dataformats='CHW', global_step=i)
                writer.add_image('pred', out, dataformats='CHW', global_step=i)

                break

            self.model.train()

            if (i % 50 == 0 and i != 0) or i == 299:
                model_path = os.path.join(output_dir, 'model_%d.pth'%i)
                if isinstance(self.model, nn.DataParallel):
                    self.model.module.save(model_path)
                else:
                    self.model.save(model_path)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()