from glob import glob
import pyembree
import trimesh
import pickle
import numpy as np
import argparse
from tqdm import tqdm
import os
from os.path import join
import subprocess
import shutil
import torch
import torch.nn.functional as F
from torch.optim import Adam
from models.smpl import SMPL
from kaolin.rep import TriangleMesh as tm
from kaolin.metrics.mesh import laplacian_loss as lap_loss
from kaolin.metrics.mesh import point_to_surface

# color intrinsic
c_fx = 1063.8987
c_fy = 1063.6822
c_cx = 954.1103
c_cy = 553.2578

# depth intrinsic
d_fx = 365.4020
d_fy = 365.6674
d_cx = 252.4944
d_cy = 207.7411


# depth-to-color transformation
d2c = np.array([
 [ 9.99968867e-01, -6.80652917e-03, -9.22235761e-03, -5.44798585e-02], 
 [ 6.69376504e-03,  9.99922175e-01, -1.15625133e-02, -3.41685168e-04], 
 [ 9.27761963e-03,  1.14376287e-02,  9.99882174e-01, -2.50539462e-03], 
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00], 
    ])


def get_bbox(verts, w2d):
    pad_ones = np.ones((verts.shape[0],1))
    pad_verts = np.hstack((verts, pad_ones))

    color_verts = np.matmul(d2c, np.matmul(w2d, pad_verts.transpose())).transpose()

    x = color_verts[:,0] / color_verts[:,2] * c_fx + c_cx
    y = color_verts[:,1] / color_verts[:,2] * c_fy + c_cy

    x = np.floor(np.round(np.clip(x, 0, 1920-1))).astype(np.int32)
    y = np.floor(np.round(np.clip(y, 0, 1080-1))).astype(np.int32)

    min_x = np.min(x)
    max_x = np.max(x)
    min_y = np.min(y)
    max_y = np.max(y)

    center = (max_x+min_x) / 2, (max_y+min_y) / 2
    w = (center[0] - min_x) * 1.2
    h = (center[1] - min_y) * 1.2

    return np.array([max(center[0]-w,0), max(center[1]-h,0), min(center[0]+w,1919), min(center[1]+h,1079)])


def optimize(model, pose, shape, gt_mesh, faces):
    tmp = torch.zeros([1, 3]).cuda()

    with torch.no_grad():
        verts, _ = model(torch.cat([tmp, pose], dim=1), shape)
        smpl_mesh = tm.from_tensors(vertices=verts.squeeze(0), faces=faces)

    optimizer = Adam([pose, shape], lr=0.02)

    for i in tqdm(range(30)):
        verts, _ = model(torch.cat([tmp, pose], dim=1), shape)
        optim_mesh = tm.from_tensors(vertices=verts.squeeze(0), faces=faces)

        s2m = 1000 * point_to_surface(gt_mesh.vertices, optim_mesh)
        m2s = 1000 * point_to_surface(optim_mesh.vertices, gt_mesh)
        norm = 0 * F.mse_loss(shape, torch.zeros_like(shape))

        loss = s2m + m2s + norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return torch.cat([tmp, pose.detach()], dim=1), shape.detach()

def main(data_path):
    dires = glob(join(data_path, "*/*"))
    smpl = SMPL('data/neutral_smpl_with_cocoplus_reg.pkl', 'data/J_regressor_h36m.npy').cuda()
    faces = torch.from_numpy(smpl.faces).long().cuda()
    for dire in tqdm(dires):
        mesh_name = join(dire, 'mesh.obj')
        w2d = np.loadtxt(join(dire, 'cam.txt'))
        mesh = trimesh.load(mesh_name)
        tmp = dict()
        tmp['bbox'] = get_bbox(mesh.vertices, w2d)
        file_name = join(dire, 'extra.npy')
        np.save(file_name, tmp)

        with open(join(dire, 'smpl_params.txt'), 'r') as fp:
            lines = fp.readlines()
            lines = [l.strip() for l in lines]

            betas_data = filter(lambda s: len(s)!=0, lines[1].split(' '))
            beta = np.array([float(b) for b in betas_data])

            root_mat_data = lines[3].split(' ') + lines[4].split(' ') +\
                lines[5].split(' ') + lines[6].split(' ')

            root_mat_data = filter(lambda s: len(s)!=0, root_mat_data)
            root_mat = np.reshape(np.array([float(m) for m in root_mat_data]), (4, 4))

            root_rot = root_mat[:3, :3]
            root_trans = root_mat[:3, 3]

            theta_data = lines[8:80]
            theta = np.array([float(t) for t in theta_data])

        pose = torch.from_numpy(theta.astype(np.float32)).unsqueeze(0).cuda()
        pose = pose[:, 3:]
        shape = torch.from_numpy(beta.astype(np.float32)).unsqueeze(0).cuda()

        pose.requires_grad_(True)
        shape.requires_grad_(True)

        mesh = trimesh.load(join(dire, 'mesh.obj'), process=False, maintain_order=True)

        gt_verts = torch.from_numpy(np.array(mesh.vertices.copy()).astype(np.float32)).cuda()
        gt_faces = torch.from_numpy(np.array(mesh.faces.copy()).astype(np.int64)).cuda()
        gt_mesh = tm.from_tensors(vertices=gt_verts, faces=gt_faces)

        pose, shape = optimize(smpl, pose, shape, gt_mesh, faces)

        pose = pose.squeeze().detach().cpu().numpy()
        shape = shape.squeeze().detach().cpu().numpy()

        smpld = pickle.load(open(join(dire, 'smpld.pkl'), 'rb'), encoding='latin1')
        smpld['pose'] = pose
        smpld['betas'] = shape

        pickle.dump(smpld, open(join(dire, 'smpld.pkl'), 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='deep human data path')
    args = parser.parse_args()

    args.data_path = '/home/llx/deephuman/dataset'
    main(args.data_path)