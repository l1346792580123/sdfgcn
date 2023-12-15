import yaml
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from glob import glob
import trimesh
import os
from os.path import join
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.autograd as autograd
from models.totalmodel import TotalModel
from models.smpl import SMPL
from models.util import batch_rodrigues
from models.myparallel import MyParallel
from datasets.deephuman import DeepHuman
from datasets.renderpeople import RenderPeople
from datasets.total_datasets import TotalDataset
class Trainer(object):
    def __init__(self, config):
        os.environ['CUDA_VISIBLE_DEVICES'] = config['gpus']
        self.model = TotalModel(config['model'])
        self.dataset = TotalDataset(config['datasets'], config['num_iters'])
        # self.test_set = DeepHuman(**config['datasets']['DeepHuman'])
        config['datasets']['RenderPeople']['if_train']= True
        config['datasets']['RenderPeople']['num_view'] = 1
        self.test_set = RenderPeople(**config['datasets']['RenderPeople'])
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=True)

        self.device = config['device']
        self.use_cuda = True if self.device == 'cuda' else False
        self.config = config
        self.train_delta = config['train_delta']

        if self.use_cuda:
            self.model = self.model.cuda()

        if 'pretrained_model_path' in config.keys() and config['pretrained_model_path'] != '':
            import torch
            state_dict = torch.load(config['pretrained_model_path'])
            for key in list(state_dict.keys()):
                if 'smpl_coma' in key or 'verts_weight' in key or 'smpl.' in key or 'lloss' in key:
                    state_dict.pop(key)

            self.model.load_state_dict(state_dict, strict=False)

        if 'pretrained_normalnet' in config.keys() and config['pretrained_normalnet'] != '':
            import torch
            state_dict = torch.load(config['pretrained_normalnet'])
            self.model.normalnet.load_state_dict(state_dict)
            self.model.normalnet.eval()
            self.model.normalnet.requires_grad_(False)

        optim_config = config['optimizer']
        optim_name = optim_config['name']
        optim_config.pop('name')

        self.optimizer = eval(optim_name)(self.model.parameters(), **optim_config)

        self.delta_optimizer = eval(optim_name)(self.model.gcn_decoder.parameters(), **optim_config)

        scheduler_config = config['scheduler']
        scheduler_name = scheduler_config['name']
        scheduler_config.pop('name')
        self.scheduler = eval(scheduler_name)(self.optimizer, **scheduler_config)

        self.delta_scheduler = eval(scheduler_name)(self.delta_optimizer, **scheduler_config)

        if 'start_combine' in config.keys() and config['start_combine'] != '':
            self.start_combine = config['start_combine']
        else:
            self.start_combine = 0

        if len(config['gpus']) > 1:
            self.model = MyParallel(self.model)

        loss_weight = config['loss_weight']
        self.verts_weight = loss_weight['verts']
        self.edge_weight = loss_weight['edge']
        self.normal_weight = loss_weight['normal']
        self.sdf_weight = loss_weight['sdf']
        self.gcn_sdf_weight = loss_weight['gcn_sdf']
        self.delta_norm_weight = loss_weight['delta_norm']
        self.laplacian_weight = loss_weight['laplacian']
        self.p2s_weight = loss_weight['p2s']
        self.offset_weight = loss_weight['offset']
        self.imgnormal_weight = loss_weight['imgnormal']

    def train(self):
        import torch
        now = datetime.datetime.now()
        writer = SummaryWriter(os.path.join(self.config['logdir'], now.strftime('%Y-%m-%d-%H-%M')))
        output_dir = os.path.join(self.config['output_dir'], now.strftime('%Y-%m-%d-%H-%M'))
        saveobj_dir = join(output_dir, 'saved_obj')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(saveobj_dir)

        yaml.dump(self.config, open(os.path.join(output_dir, 'config.yaml'), 'w'))

        self.model.train()
        if 'normalnet' in self.config['model'].keys():
            if isinstance(self.model, nn.DataParallel):
                self.model.module.normalnet.eval()
            else:
                self.model.normalnet.eval()
        
        # with autograd.detect_anomaly():
        if True:
            for idx, datas in tqdm(enumerate(self.dataset)):
                total_verts_loss = []
                total_edge_loss = []
                total_normal_loss = []
                total_sdf_loss = []
                total_gcn_sdf_loss = []
                total_delta_norm_loss = []
                total_laplacian_loss = []
                total_p2s_loss = []
                total_offset_loss = []
                total_imgnormal_loss = []
                total_transedge_loss = []
                for i in range(len(datas)):
                    loss = 0
                    delta_loss = 0
                    if self.use_cuda:
                        for ii in range(len(datas[i])):
                            if isinstance(datas[i][ii], torch.Tensor):
                                datas[i][ii] = datas[i][ii].cuda()
                            elif isinstance(datas[i][ii], list):
                                datas[i][ii] = [item.cuda() for item in datas[i][ii]]
                            else:
                                raise NotImplementedError
                    if len(datas[i]) == 10:
                        img, sampled_points, sdf, offsets, calib, transform, beta, theta, vertices, faces = datas[i]
                        # b = theta.shape[0]
                        # theta = batch_rodrigues(theta.reshape(-1,3)).reshape(b,-1,3,3)
                        gts = [theta, beta, sdf, offsets, vertices, faces]
                    elif len(datas[i]) == 8:
                        img, sampled_points, sdf, calib, transform, smpl_verts, vertices, faces = datas[i]
                        gts = [smpl_verts, sdf, vertices, faces]

                    losses = self.model(img, sampled_points, calib, transform, if_detach=idx<self.start_combine, gts=gts)
                    losses = [item.mean() for item in losses]
                    verts_loss, normal_loss, edge_loss, sdf_loss, gcn_sdf_loss, delta_norm_loss, laplacian_loss, p2s_loss, offset_loss, img_normal_loss, trans_edge_loss = losses

                    loss += self.verts_weight * verts_loss + self.normal_weight * normal_loss + self.edge_weight * edge_loss +\
                            self.sdf_weight * sdf_loss + self.imgnormal_weight * img_normal_loss

                    delta_loss += self.gcn_sdf_weight * gcn_sdf_loss + self.delta_norm_weight * delta_norm_loss +\
                            self.laplacian_weight * laplacian_loss + self.p2s_weight * p2s_loss + self.offset_weight * offset_loss + self.edge_weight * trans_edge_loss

                    if self.train_delta:
                        self.delta_optimizer.zero_grad()
                        delta_loss.backward()
                        self.delta_optimizer.step()

                        self.delta_scheduler.step()
                    else:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        self.scheduler.step()

                    total_verts_loss.append(verts_loss.item())
                    total_normal_loss.append(normal_loss.item())
                    total_edge_loss.append(edge_loss.item())
                    total_sdf_loss.append(sdf_loss.item())
                    total_gcn_sdf_loss.append(gcn_sdf_loss.item())
                    total_delta_norm_loss.append(delta_norm_loss.item())
                    total_laplacian_loss.append(laplacian_loss.item())
                    total_p2s_loss.append(p2s_loss.item())
                    total_offset_loss.append(offset_loss.item())
                    total_imgnormal_loss.append(img_normal_loss.item())
                    total_transedge_loss.append(trans_edge_loss.item())

                total_verts_loss = self.verts_weight * np.array(total_verts_loss).mean()
                total_normal_loss = self.normal_weight * np.array(total_normal_loss).mean()
                total_edge_loss = self.edge_weight * np.array(total_edge_loss).mean()
                total_sdf_loss = self.sdf_weight * np.array(total_sdf_loss).mean()
                total_gcn_sdf_loss = self.gcn_sdf_weight * np.array(total_gcn_sdf_loss).mean()
                total_delta_norm_loss = self.delta_norm_weight * np.array(total_delta_norm_loss).mean()
                total_laplacian_loss = self.laplacian_weight * np.array(total_laplacian_loss).mean()
                total_p2s_loss = self.p2s_weight * np.array(total_p2s_loss).mean()
                total_offset_loss = self.offset_weight * np.array(total_offset_loss).mean()
                total_imgnormal_loss = self.imgnormal_weight * np.array(total_imgnormal_loss).mean()
                total_transedge_loss = self.edge_weight * np.array(total_transedge_loss).mean()

                writer.add_scalar('verts_loss', total_verts_loss, idx)
                writer.add_scalar('normal_loss', total_normal_loss, idx)
                writer.add_scalar('edge_loss', total_edge_loss, idx)
                writer.add_scalar('sdf_loss', total_sdf_loss, idx)
                writer.add_scalar('gcn_sdf_loss', total_gcn_sdf_loss, idx)
                writer.add_scalar('delta_norm_loss', total_delta_norm_loss, idx)
                writer.add_scalar('laplacian_loss', total_laplacian_loss, idx)
                writer.add_scalar('p2s_loss', total_p2s_loss, idx)
                writer.add_scalar('offset_loss', total_offset_loss, idx)
                writer.add_scalar('imgnormal_loss', total_imgnormal_loss, idx)
                writer.add_scalar('transedge_loss', total_transedge_loss, idx)

                if idx % 10000 == 0 and idx != 0:
                    with torch.no_grad():
                        self.model.eval()
                        for idxx, data in enumerate(self.test_loader):
                            data = [item.cuda() for item in data]

                            if len(data) == 10:
                                img, sampled_points, sdf, offsets, calib, transform, beta, theta, vertices, faces = data
                                gts = [theta, beta, sdf, offsets, vertices, faces]
                            elif len(data) == 8:
                                img, sampled_points, sdf, calib, transform, smpl_verts, vertices, faces = data
                                gts = [smpl_verts, sdf, vertices, faces]

                            pred_verts, trans_verts, pred_sdf = self.model(img, sampled_points, calib, transform)
                            break

                        trans_verts = trans_verts.squeeze(0).detach().cpu().numpy()
                        pred_verts = pred_verts.squeeze(0).detach().cpu().numpy()
                        if isinstance(self.model, nn.DataParallel):
                            faces = self.model.module.faces
                            # gt_verts, _ = self.model.module.smpl(theta, beta)
                            # gt_verts = gt_verts + offsets
                            # gt_verts = gt_verts.squeeze(0).detach().cpu().numpy()
                            self.model.module.reconstruct(img, join(saveobj_dir, 'sdf_%d.ply'%idx), calib, transform, N=512, level=0.5)

                            mesh = trimesh.load(join(saveobj_dir, 'sdf_%d.ply'%idx))
                            cc = mesh.split(only_watertight=False)
                            out_mesh = cc[0]
                            bbox = out_mesh.bounds
                            height = bbox[1,0] - bbox[0,0]
                            for c in cc:
                                bbox = c.bounds
                                if height < bbox[1,0] - bbox[0,0]:
                                    height = bbox[1,0] - bbox[0,0]
                                    out_mesh = c
                            out_mesh.export(join(saveobj_dir, 'sdf_%d.ply'%idx))

                        else:
                            faces = self.model.faces
                            # gt_verts, _ = self.model.smpl(theta, beta)
                            # gt_verts = gt_verts + offsets
                            # gt_verts = gt_verts.squeeze(0).detach().cpu().numpy()

                        with open(join(saveobj_dir, 'smpl_%d.obj'%idx), 'w') as f:
                            for v in pred_verts:
                                f.write('v %f %f %f\n'%(v[0], v[1], v[2]))
                            for face in faces:
                                f.write('f %d %d %d\n'%(face[0]+1, face[1]+1, face[2]+1))
                        with open(join(saveobj_dir, 'trans_smpl_%d.obj'%idx), 'w') as f:
                            for v in trans_verts:
                                f.write('v %f %f %f\n'%(v[0], v[1], v[2]))
                            for face in faces:
                                f.write('f %d %d %d\n'%(face[0]+1, face[1]+1, face[2]+1))


                        self.model.train()
                        if 'normalnet' in self.config['model'].keys():
                            if isinstance(self.model, nn.DataParallel):
                                self.model.module.normalnet.eval()
                            else:
                                self.model.normalnet.eval()

                    model_path = os.path.join(output_dir, 'model_%d.pth'%idx)
                    if isinstance(self.model, nn.DataParallel):
                        self.model.module.save(model_path)
                    else:
                        self.model.save(model_path)

        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, help='training config file')
    args = parser.parse_args()

    # args.cfg_path = 'configs/pifu_gcn.yaml'
    config = yaml.load(open(args.cfg_path,'r'), Loader=yaml.FullLoader)

    trainer = Trainer(config)

    trainer.train()
