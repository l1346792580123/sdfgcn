import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class CustomBCELoss(nn.Module):
    def __init__(self, brock=False, gamma=None):
        super(CustomBCELoss, self).__init__()
        self.brock = brock
        self.gamma = gamma

    def forward(self, pred, gt, gamma, w=None):
        x_hat = torch.clamp(pred, 1e-5, 1.0-1e-5) # prevent log(0) from happening
        # gamma = gamma[:,None,None] if self.gamma is None else self.gamma
        if self.brock:
            x = 3.0*gt - 1.0 # rescaled to [-1,2]

            loss = -(gamma*x*torch.log(x_hat) + (1.0-gamma)*(1.0-x)*torch.log(1.0-x_hat))
        else:
            loss = -(gamma*gt*torch.log(x_hat) + (1.0-gamma)*(1.0-gt)*torch.log(1.0-x_hat))

        if w is not None:
            if len(w.size()) == 1:
                w = w[:,None,None]
            return (loss * w).mean()
        else:
            return loss.mean()

'''
adapted from pose2mesh https://github.com/hongsukchoi/Pose2Mesh_RELEASE/blob/master/lib/core/loss.py
'''

class LaplacianLoss(nn.Module):
    def __init__(self, faces, vnum=6890):
        super(LaplacianLoss, self).__init__()

        laplacian = np.zeros([vnum, vnum], dtype=np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(laplacian.shape[0]):
            laplacian[i, :] /= (laplacian[i, i] + 1e-8)

        self.register_buffer('laplacian', torch.from_numpy(laplacian).float())

    def forward(self, x):
        batch_size = x.size(0)

        x = torch.cat([torch.matmul(self.laplacian, x[i])[None, :, :] for i in range(batch_size)], 0)
        x = x.pow(2).sum(2)
        return x.mean()

class NormalVectorLoss(nn.Module):
    def __init__(self, face):
        super(NormalVectorLoss, self).__init__()
        self.register_buffer('face', torch.from_numpy(face).long())

    def forward(self, coord_out, coord_gt):
        face = self.face

        v1_out = coord_out[:, face[:, 1], :] - coord_out[:, face[:, 0], :]
        v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
        v2_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 0], :]
        v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
        v3_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 1], :]
        v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

        v1_gt = coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 0], :]
        v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
        v2_gt = coord_gt[:, face[:, 2], :] - coord_gt[:, face[:, 0], :]
        v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector

        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))
        loss = torch.cat((cos1, cos2, cos3), 1)
        return loss.mean()


class EdgeLengthLoss(nn.Module):
    def __init__(self, face):
        super(EdgeLengthLoss, self).__init__()
        self.register_buffer('face', torch.from_numpy(face).long())

    def forward(self, coord_out, coord_gt, weight=None):
        face = self.face
        if weight is None:
            w1 = w2 = w3 = torch.ones(1, self.face.shape[0], 1, device=face.device)
        else:
            weight = weight.reshape(1, -1, 1)
            w1 = weight[:, face[:, 0], :] * weight[:, face[:, 1], :]
            w2 = weight[:, face[:, 0], :] * weight[:, face[:, 2], :]
            w3 = weight[:, face[:, 1], :] * weight[:, face[:, 2], :]
            tmp = torch.zeros_like(w1)
            w1 = torch.where(w1!=1., w1, tmp)
            w2 = torch.where(w2!=1., w2, tmp)
            w3 = torch.where(w3!=1., w3, tmp)

        d1_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))

        d1_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
 
        diff1 = torch.abs(d1_out - d1_gt) * w1
        diff2 = torch.abs(d2_out - d2_gt) * w2
        diff3 = torch.abs(d3_out - d3_gt) * w3
        loss = torch.cat((diff1, diff2, diff3), 1)
        return loss.mean()