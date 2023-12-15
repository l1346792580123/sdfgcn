import numpy as np
from psbody.mesh import Mesh
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
if __name__ == "__main__":
    from mesh_sampling import generate_transform_matrices, laplacian, rescale_L
else:
    from .mesh_sampling import generate_transform_matrices, laplacian, rescale_L

def poolwT(x, L):
    mp = L.shape[0]
    n, m, fin = x.shape
    x = x.permute(1,2,0).reshape(m, fin*n)
    x = torch.matmul(L, x).reshape(mp, fin, n).permute(2,0,1).contiguous()
    return x

class comaGN(nn.Module):
    def __init__(self, channels, G=32, eps=1e-5):
        super(comaGN, self).__init__()
        self.G = G
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1,channels,1))
        self.beta = nn.Parameter(torch.zeros(1,channels,1))

    def forward(self, x):
        x = x.permute(0,2,1)
        N,C,V = x.shape
        G = min(self.G, C)

        x = x.reshape(-1, G, C // G, V)
        mean = torch.mean(x, dim=(2,3), keepdim=True)
        var = torch.var(x, dim=(2,3), unbiased=False, keepdim=True)

        x = (x-mean) / torch.sqrt(var + self.eps)

        output = x.reshape(-1, C, V) * self.gamma + self.beta
        output = output.permute(0,2,1).contiguous()
        return output

class chebConv(nn.Module):
    def __init__(self, fin, fout, k=1, bias=False):
        super(chebConv, self).__init__()
        self.fin = fin
        self.fout = fout
        self.k = k
        self.bias = bias

        self.w = nn.Parameter(torch.zeros(fin*k, fout))
        nn.init.kaiming_uniform_(self.w.data)
        if bias:
            self.b = nn.Parameter(torch.zeros(1,1,fout))

    def concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x,x_], dim=0)

    def forward(self, x, L):
        n, m, fin = x.shape # batch_size num_vertices channels_in
        x0 = x.permute(1,2,0).reshape(m, fin*n)
        x = x0.unsqueeze(0) # 1 m fin*n

        if self.k > 1:
            x1 = torch.matmul(L, x0) # m*m m*(fin*n) -> m*(fin*n)
            x = self.concat(x, x1)

        for i in range(2, self.k):
            x2 = 2 * torch.matmul(L, x1) - x0
            x = self.concat(x,x2)
            x0, x1 = x1, x2

        x = x.reshape(self.k, m, fin, n).permute(3, 1, 2, 0).reshape(n*m, fin*self.k)
        x = torch.matmul(x, self.w).reshape(n, m, self.fout)
        if self.bias:
            x = x + self.b
        return x

class comaResBlock(nn.Module):
    def __init__(self, fin, fout, k):
        super(comaResBlock, self).__init__()

        self.conv1 = chebConv(fin, fout, k)
        self.conv2 = chebConv(fout, fout, k)

        if fin != fout:
            self.filter = chebConv(fin, fout, 1)
        else:
            self.filter = None
        
    def forward(self, x, L, D):
        x_1 = self.conv1(x, L)
        x_1 = F.leaky_relu(x_1, 0.2)

        x_2 = self.conv2(x_1, L)

        if self.filter is not None:
            x = self.filter(x, L)

        x_2 += x
        x_2 = F.leaky_relu(x_2, 0.2)

        out = poolwT(x_2, D)

        return out


class comaResBlockDecoder(nn.Module):
    def __init__(self, fin, fout, k):
        super(comaResBlockDecoder, self).__init__()
        self.gn1 = comaGN(fin)
        self.gn2 = comaGN(fout//2)
        self.gn3 = comaGN(fout//2)
        self.lin1 = chebConv(fin, fout//2, 1)
        self.conv = chebConv(fout//2, fout//2, k)
        self.lin2 = chebConv(fout//2, fout, 1)

        if fin != fout:
            self.filter = chebConv(fin, fout, 1)
        else:
            self.filter = None

    def forward(self, x, L, U):
        x_unpooled = poolwT(x, U)
        x = F.relu(self.gn1(x_unpooled))

        x = self.lin1(x, L)
        x = F.relu(self.gn2(x))

        x = self.conv(x, L)
        x = F.relu(self.gn3(x))

        x = self.lin2(x, L)
        
        if self.filter is not None:
            x_unpooled = self.filter(x_unpooled, L)

        x = x + x_unpooled
        return x

class ResBlockDecoder(nn.Module):
    def __init__(self, fin, fout, k):
        super(ResBlockDecoder, self).__init__()
        self.lin1 = chebConv(fin, fout//2, 1)
        self.conv = chebConv(fout//2, fout//2, k)
        self.lin2 = chebConv(fout//2, fout, 1)
        if fin != fout:
            self.filter = chebConv(fin, fout, 1)
        else:
            self.filter = None

    def forward(self, x, L, U):
        x_unpooled = poolwT(x, U)
        x = F.leaky_relu(x_unpooled, 0.2)

        x = self.lin1(x, L)
        x = F.leaky_relu(x, 0.2)

        x = self.conv(x, L)
        x = F.leaky_relu(x, 0.2)
        
        x = self.lin2(x, L)

        if self.filter is not None:
            x_unpooled = self.filter(x_unpooled, L)

        x = x + x_unpooled
        return x

def convert2sparse(matrix):
    matrix = matrix.tocoo()
    index = torch.from_numpy(np.array([matrix.row, matrix.col]))
    val = torch.from_numpy(matrix.data)
    return torch.sparse_coo_tensor(index, val, matrix.shape)

class GCNDecoder(nn.Module):
    def __init__(self, template_mesh, num_input, num_latent=64, add_gn=True):
        super(GCNDecoder, self).__init__()
        ref_mesh = Mesh(filename=template_mesh)
        self.faces = ref_mesh.f.astype(np.int32)
        ds_factors = [1, 2, 1, 2, 1, 2, 1, 1]
        pkl_name = template_mesh.replace('obj', 'pkl')
        if not os.path.exists(pkl_name):
            M,A,D,U, _ = generate_transform_matrices(ref_mesh, ds_factors)
            with open(pkl_name, 'wb') as fp:
                pickle.dump([M,A,D,U], fp)
        else:
            with open(pkl_name, 'rb') as fp:
                M,A,D,U, = pickle.load(fp)
        self.p = list(map(lambda x: x.shape[0], A))
        A = list(map(lambda x: x.astype('float32'), A))
        U = list(map(lambda x: x.astype('float32'), U))

        L = [laplacian(a, normalized=True) for a in A]

        U = [convert2sparse(u) for u in U]
        L = [convert2sparse(l) for l in L]

        self.register_list(L, 'L')
        self.register_list(U, 'U')

        nf = 64
        self.out_channels = [nf, nf, 2 * nf, 2 * nf, 4 * nf, 4 * nf, 8 * nf, 8 * nf]
        self.order = [2] * 8
        self.num_latent = num_latent
        self.num_1x1encode = 64
        self.num_1x1decode = 512
        self.num = 8

        self.decoder = nn.ModuleList()
        self.decode_conv1x1 = chebConv(self.num_latent, self.num_1x1decode, k=1)
        self.conv3 = chebConv(self.out_channels[0], 3, k=2)

        self.decode_fc = nn.Linear(num_input, self.p[-1]*self.num_latent)

        self.mean = nn.Parameter(torch.zeros(1, self.p[0], 3))
        self.std = nn.Parameter(torch.ones(1, self.p[0], 3)*0.02)

        for i in range(len(self.out_channels)):
            de_fout = self.out_channels[-i-1]
            if i == 0:
                de_fin = self.num_1x1decode
            else:
                de_fin = self.out_channels[-i]

            if add_gn:
                self.decoder.append(comaResBlockDecoder(de_fin, de_fout, self.order[i]))
            else:
                self.decoder.append(ResBlockDecoder(de_fin, de_fout, self.order[i]))


    def forward(self, latent):
        x_tmp = self.decode_fc(latent)
        x_tmp = F.leaky_relu(x_tmp, negative_slope=0.2)
        x = x_tmp.reshape(x_tmp.shape[0], self.p[-1], -1)
        x = self.decode_conv1x1(x, getattr(self, 'L_%d'%(self.num-1)))

        for i in range(len(self.decoder)):
            x = self.decoder[i](x, getattr(self, 'L_%d'%((-i-1)%self.num)), getattr(self, 'U_%d'%((-i-1)%self.num)))

        x_hat = self.conv3(x, getattr(self, 'L_%d'%0))

        return x_hat * self.std + self.mean

    def register_list(self, data, name):
        for i in range(len(data)):
            self.register_buffer('%s_%d'%(name,i), data[i])


class SMPLCOMA(nn.Module):
    def __init__(self, template_mesh, trained_model=None, require_grad=True, ds_factors=[1, 2, 1, 2, 1, 2, 1, 1], num_latent=64, add_gn=True):
        super(SMPLCOMA, self).__init__()
        ref_mesh = Mesh(filename=template_mesh)
        self.faces = ref_mesh.f.astype(np.int32)
        pkl_name = template_mesh.replace('obj', 'pkl')
        if not os.path.exists(pkl_name):
            M,A,D,U, _ = generate_transform_matrices(ref_mesh, ds_factors)
            with open(pkl_name, 'wb') as fp:
                pickle.dump([M,A,D,U], fp)
        else:
            with open(pkl_name, 'rb') as fp:
                M,A,D,U, = pickle.load(fp)
        self.p = list(map(lambda x: x.shape[0], A))
        A = list(map(lambda x: x.astype('float32'), A))
        D = list(map(lambda x: x.astype('float32'), D))
        U = list(map(lambda x: x.astype('float32'), U))

        L = [laplacian(a, normalized=True) for a in A]

        D = [convert2sparse(d) for d in D]
        U = [convert2sparse(u) for u in U]
        L = [convert2sparse(l) for l in L]

        self.register_list(L, 'L')
        self.register_list(D, 'D')
        self.register_list(U, 'U')

        nf = 64
        self.out_channels = [nf, nf, 2 * nf, 2 * nf, 4 * nf, 4 * nf, 8 * nf, 8 * nf]
        self.order = [2] * 8
        self.num_latent = num_latent
        self.num_1x1encode = 64
        self.num_1x1decode = 512
        self.num = 8

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.encode_conv1x1 = chebConv(self.out_channels[-1], self.num_1x1encode, k=1)
        self.decode_conv1x1 = chebConv(self.num_latent, self.num_1x1decode, k=1)

        self.conv3 = chebConv(self.out_channels[0], 3, k=2)

        self.encode_fc = nn.Linear(self.p[-1]*self.num_1x1encode, self.num_latent)
        self.decode_fc = nn.Linear(self.num_latent, self.p[-1]*self.num_latent)

        self.mean = nn.Parameter(torch.zeros(1, self.p[0], 3))
        self.std = nn.Parameter(torch.ones(1, self.p[0], 3))

        for i in range(len(self.out_channels)):
            en_fout = self.out_channels[i]
            de_fout = self.out_channels[-i-1]
            if i == 0:
                en_fin = 3
                de_fin = self.num_1x1decode
            else:
                en_fin = self.out_channels[i-1]
                de_fin = self.out_channels[-i]

            self.encoder.append(comaResBlock(en_fin, en_fout, self.order[i]))
            if add_gn:
                self.decoder.append(comaResBlockDecoder(de_fin, de_fout, self.order[i]))
            else:
                self.decoder.append(ResBlockDecoder(de_fin, de_fout, self.order[i]))

        if trained_model is not None:
            state_dict = torch.load(trained_model)
            state_dict = {k.replace('module.', ''): v for k,v in state_dict.items()}
            for key in list(state_dict.keys()):
                if 'L' in key or 'D' in key or 'U' in key:
                    state_dict.pop(key)
            self.load_state_dict(state_dict, strict=False)
            self.eval()

        self.requires_grad_(require_grad)


    def save_model(self, model_path):
        state_dict = self.state_dict()
        for key in list(state_dict.keys()):
            if 'L_' in key or 'D_' in key or 'U_' in key:
                state_dict.pop(key)

        torch.save(state_dict, model_path)


    def decode(self, latent):
        x_tmp = self.decode_fc(latent)
        x_tmp = F.leaky_relu(x_tmp, negative_slope=0.2)
        x = x_tmp.reshape(x_tmp.shape[0], self.p[-1], -1)
        x = self.decode_conv1x1(x, getattr(self, 'L_%d'%(self.num-1)))

        for i in range(len(self.decoder)):
            x = self.decoder[i](x, getattr(self, 'L_%d'%((-i-1)%self.num)), getattr(self, 'U_%d'%((-i-1)%self.num)))

        x_hat = self.conv3(x, getattr(self, 'L_%d'%0))

        return x_hat * self.std + self.mean

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent)

    def encode(self, x):
        for i in range(len(self.encoder)):
            x = self.encoder[i](x, getattr(self, 'L_%d'%i), getattr(self, 'D_%d'%i))

        x = self.encode_conv1x1(x, getattr(self, 'L_%d'%(self.num-1)))
        x_tmp = x.reshape(x.shape[0], -1)
        latent = self.encode_fc(x_tmp)

        return latent

    def register_list(self, data, name):
        for i in range(len(data)):
            self.register_buffer('%s_%d'%(name,i), data[i])


if __name__ == "__main__":
    smpl_coma = SMPLCOMA('/home/llx/sdfgcn/data/template_mesh.obj', '/home/llx/sdfgcn/data/smpl_decoder.pth', False).cuda()

    verts = torch.randn(2, 6890, 3).cuda()

    ret = smpl_coma(verts)