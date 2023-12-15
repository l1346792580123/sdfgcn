import torch.nn as nn
import torch.nn.functional as F
import torch
import h5py
import numpy as np

class COMALinear(nn.Module):
    def __init__(self, layer):
        super(COMALinear, self).__init__()
        self.fc = nn.Sequential()

        for i in range(len(layer)-1):
            self.fc.add_module(name='fc_%d'%i, module=nn.Linear(layer[i], layer[i+1]))
            if i < len(layer) - 2:
                self.fc.add_module(name='relu_%d'%i, module=nn.ReLU())

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)

    def forward(self, x):
        return self.fc(x)

class SDF(nn.Module):
    def __init__(self, layer, activation='tanh'):
        super(SDF, self).__init__()
        self.fc = nn.Sequential()
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError
        for i in range(len(layer)-1):
            self.fc.add_module(name='fc_%d'%i, module=nn.Linear(layer[i], layer[i+1]))
            if i < len(layer) - 2:
                self.fc.add_module(name='relu_%d'%i, module=nn.ReLU())

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)

    def forward(self, x):
        out = self.fc(x)
        out = self.activation(out)
        return out

class SDFLinear(nn.Module):
    def __init__(self, layer):
        super(SDFLinear, self).__init__()
        self.fc = nn.Sequential()

        for i in range(len(layer)-1):
            self.fc.add_module(name='fc_%d'%i, module=nn.Linear(layer[i], layer[i+1]))
            self.fc.add_module(name='relu_%d'%i, module=nn.ReLU())

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)

    def forward(self, x):
        return self.fc(x)



class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, num_views=1, no_residual=True, last_op=nn.Sigmoid()):
        super(SurfaceClassifier, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        self.last_op = last_op

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        '''

        :param feature: list of [BxC_inxHxW] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        '''

        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)

            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)

        if self.last_op:
            y = self.last_op(y)

        return y

class SMPLLinear(nn.Module):
    def __init__(self, layer, num_iter, mean_param_path):
        super(SMPLLinear, self).__init__()

        self.fc = nn.Sequential()
        for i in range(len(layer)-1):
            self.fc.add_module(name='fc_{}'.format(i), module=nn.Linear(layer[i], layer[i+1]))

            if i < len(layer) - 2:
                self.fc.add_module(name='relu_{}'.format(i), module=nn.ReLU())

        self.num_iter = num_iter

        mean_param = h5py.File(mean_param_path, 'r')
        shape = mean_param['shape'][:]
        pose = mean_param['pose'][:]
        pose[:3] = 0.
        mean = np.hstack([pose, shape])

        self.register_buffer('mean_param', torch.from_numpy(mean).unsqueeze(0).float())
        self._init_params()

    def _init_params(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)


    def forward(self, x):
        b = x.shape[0]
        param = self.mean_param.expand(b, -1)
        params = []
        for i in range(self.num_iter):
            feat = torch.cat([x, param], dim=1)
            param = self.fc(feat) + param
            params.append(param)

        return params

class ImplicitNet(nn.Module):
    def __init__(self, dims, skip_in=[4], geometric_init=True, radius_init=1, beta=100):
        super().__init__()
        self.num_layers = len(dims)
        d_in = dims[0]
        self.skip_in = skip_in
        self.d_in = d_in
        self.d_out = dims[-1]

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)
            # if true preform preform geometric initialization
            if geometric_init:
                if layer == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        if_permute = False
        if_reshape = False
        if len(x.shape) == 3:
            if_reshape = True
            if x.shape[1] == self.d_in:
                b, _, n = x.shape
                x = x.permute(0, 2, 1).contiguous()
                if_permute = True
            elif x.shape[2] == self.d_in:
                b, n, _ = x.shape
            else:
                raise NotImplementedError

            x = x.reshape(b*n, self.d_in)
        out = x
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.skip_in:
                out = torch.cat([out, x], -1) / np.sqrt(2)
            out = lin(out)
            if layer < self.num_layers - 2:
                out = self.activation(out)

        if if_reshape:
            out = out.reshape(b, n, self.d_out)
        if if_permute:
            out = out.permute(0, 2, 1).contiguous()

        return out

if __name__ == "__main__":
    import torch
    model = SurfaceClassifier([257, 1024, 512, 256, 128, 1])
    x = torch.randn(2, 257, 10)
    y = model(x)
    print(y.shape)