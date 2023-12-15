import numpy as np
import cv2
import os
import torch
from torch.autograd import grad
from kaolin.rep import TriangleMesh as tm
from kaolin.metrics.mesh import laplacian_loss as lap_loss
from kaolin.metrics.mesh import point_to_surface
import torch.nn.functional as F
from models.totalmodel import TotalModel
from models.smpl import SMPL
from models.smplx import SMPLX
import pickle
from os.path import join
import trimesh
from tqdm import tqdm
import yaml
from glob import glob
from models.glrenderer import glRenderer

c_fx = 1063.8987
c_fy = 1063.6822
c_cx = 954.1103
c_cy = 553.2578

d2c = np.array([
 [ 9.99968867e-01, -6.80652917e-03, -9.22235761e-03, -5.44798585e-02], 
 [ 6.69376504e-03,  9.99922175e-01, -1.15625133e-02, -3.41685168e-04], 
 [ 9.27761963e-03,  1.14376287e-02,  9.99882174e-01, -2.50539462e-03], 
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00], ], dtype=np.float32)

def get_data(dire, res=224, with_normal=False):
    img = cv2.imread(join(dire, 'color.jpg'))
    if with_normal:
        normal = cv2.imread(join(dire, 'ground_rendered_normal.png'))
    extra_data = np.load(join(dire, 'extra.npy'), allow_pickle=True).item()
    cam = np.loadtxt(join(dire, 'cam.txt'), dtype=np.float32)

    with open(join(dire, 'smpl_params.txt'), 'r') as fp:
        lines = fp.readlines()
        lines = [l.strip() for l in lines]
        root_mat_data = lines[3].split(' ') + lines[4].split(' ') +\
            lines[5].split(' ') + lines[6].split(' ')
        root_mat_data = filter(lambda s: len(s)!=0, root_mat_data)
        root_mat = np.reshape(np.array([float(m) for m in root_mat_data]), (4, 4))
        root_rot = root_mat[:3, :3]
        root_trans = root_mat[:3, 3]

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
    img = cv2.resize(img, (res, res))

    calib = np.eye(4, dtype=np.float32)

    calib[:3,:3] = root_rot
    calib[:3, 3] = root_trans
    calib = d2c @ cam @ calib

    transform = np.zeros((2,3), np.float32)
    factor = (1080/res)
    tmp = res // 2
    transform[0,0] = c_fx / factor / tmp
    transform[1,1] = c_fy / factor / tmp
    transform[0,2] = (c_cx - min_x) / factor / tmp - 1.
    transform[1,2] = c_cy / factor / tmp - 1.

    img = (img / 255.).transpose(2,0,1).astype(np.float32)
    if with_normal:
        normal = (normal / 255.).transpose(2,0,1).astype(np.float32)
        img = np.concatenate([img, normal], axis=0)

    img = torch.from_numpy(img[None])
    calib = torch.from_numpy(calib[None])
    transform = torch.from_numpy(transform[None])

    smpld = pickle.load(open(join(dire, 'smpld.pkl'), 'rb'))
    beta = smpld['betas'][:10]
    theta = smpld['pose']
    beta = torch.from_numpy(beta[None])
    theta = torch.from_numpy(theta[None])

    return img, min_x, calib, transform, beta, theta

def subdivide(verts, weight):
    device = verts.device

    edge_unique = np.load('data/edge_unique.npy') # k 2
    newfaces = np.loadtxt('data/sub_smpl_faces.txt').astype(np.int64)

    newverts = verts[:, edge_unique].mean(dim=2) # b n 3
    newweight = torch.ones(1,edge_unique.shape[0],1).to(device)
    for i in range(len(edge_unique)):
        if weight[:,edge_unique[i,0],:] == 0 or weight[:,edge_unique[i,1],:] == 0:
            newweight[:,i,:] = 0

    newverts = torch.cat([verts, newverts], dim=1)
    newfaces = torch.from_numpy(newfaces).to(device)
    newweight = torch.cat([weight, newweight], dim=1)

    return newverts, newfaces, newweight



def implicit_optimize(model, img, num_iter, calib, transform):
    model.requires_grad_(False)
    
    device = img.device
    b = img.shape[0]
    assert b == 1

    weight = torch.ones(6890).to(device)
    files = ['data/lh.txt', 'data/rh.txt', 'data/rf.txt', 'data/lf.txt', 'data/face.txt']
    for file in files:
        index = np.loadtxt(file)
        weight[index] = 0.

    weight = weight.reshape(1,-1,1)
        
    model.reconstruct(img, 'sdf.ply', calib, transform, level=0.5, N=256, max_batch=64**3)
    mesh = trimesh.load('sdf.ply')
    cc = mesh.split(only_watertight=False)
    out_mesh = cc[0]
    bbox = out_mesh.bounds
    height = bbox[1,0] - bbox[0,0]
    for c in cc:
        bbox = c.bounds
        if height < bbox[1,0] - bbox[0,0]:
            height = bbox[1,0] - bbox[0,0]
            out_mesh = c
    out_mesh.export('sdf.ply')

    smpl_faces = model.smpl.faces.copy().astype(np.int64)
    
    if model.normalnet is not None:
        image = img[:, :3]
        gt_imgnormal = img[:, 3:]
        pred_normal = model.normalnet(image)
        img = torch.cat([image, pred_normal], dim=1)

    feat, im_feat_list, tmpx, normx = model.backbone(img)
    if model.smpl_type == 'coma':
        latent = model.coma_linear(feat)
        pred_verts = model.smpl_coma.decode(latent)
    elif model.smpl_type == 'linear':
        params = model.smpl_linear(feat)
        param = params[-1]
        pred_theta = param[:, :72]
        pred_beta = param[:, 72:82]
        pred_verts, _ = model.smpl(pred_theta, pred_beta)
        
    verts_delta = model.gcn_decoder(feat)
    trans_verts = pred_verts + verts_delta * model.verts_weight.unsqueeze(2)
    pred_verts = pred_verts.detach()
    trans_verts = trans_verts.detach()

    pred_verts, smpl_face, newweight = subdivide(pred_verts, weight)
    trans_verts, smpl_face, newweight = subdivide(trans_verts, weight)
    weight = newweight

    delta = torch.zeros(trans_verts.shape).to(device)
    delta.requires_grad = True

    optimizer = torch.optim.Adam([delta], lr=0.002)

    smpl_mesh = tm.from_tensors(vertices=pred_verts.squeeze(0), faces=smpl_face)

    only_z = model.only_z

    for i in tqdm(range(num_iter)):
        loss = 0
        it = i // 10
        tmp_delta = torch.cat([torch.zeros_like(delta)[:,:,:2], delta[:,:,2:3]], dim=2)
        optim_verts = (pred_verts + tmp_delta*weight).squeeze(0)

        optim_mesh = tm.from_tensors(vertices=optim_verts, faces=smpl_face)

        offsets = 0 * torch.mean(delta**2)
        lap = 5000 * lap_loss(optim_mesh, smpl_mesh)
        loss += offsets + lap

        optim_verts = optim_verts.unsqueeze(0).permute(0,2,1).contiguous()
        xyz, homo = model.projection(optim_verts, calib, transform)
        xy = xyz[:, :2, :].detach()
        z = xyz[:, 2:3, :]
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0) # B N
        if only_z:
            if model.use_origin:
                tmp = model.embedder(optim_verts[:,2:3].permute(0,2,1).contiguous()*10).permute(0,2,1).contiguous()
            else:
                tmp = model.embedder(((z-model.z_offset)*10).permute(0,2,1).contiguous()).permute(0,2,1).contiguous()
        else:
            tmp = model.embedder(optim_verts.permute(0,2,1).contiguous()).permute(0,2,1).contiguous()

        point_local_feat = torch.cat([model.index(im_feat_list[-1], xy), tmp], dim=1)
        pred_gcn_sdf = model.sdf(point_local_feat).squeeze(1)
        gcnsdf = 10 * F.l1_loss(pred_gcn_sdf*in_img, torch.ones_like(pred_gcn_sdf)*0.5*in_img, reduction='mean')

        loss += gcnsdf

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.requires_grad_(True)
    return pred_verts, delta.detach() * weight, smpl_face

def main():
    pt = 'trained_model/yyyy-MM-DD-HH-mm/'
    config = yaml.load(open(join(pt, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
    res = 512

    model = TotalModel(config['model'])
    model_path = join(pt, 'model_10000.pth')
    state_dict = torch.load(model_path)

    for key in list(state_dict.keys()):
        if 'smpl_coma' in key or 'gcn_decoder.L' in key or 'gcn_decoder.U' in key or 'v2f' in key or 'smpl.' in key:
            state_dict.pop(key)

    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()
    model.eval()

    smplx = SMPLX('data/neutral_smpl_with_cocoplus_reg.pkl', is_smplx=False).cuda()

    index = np.loadtxt('data/shuffle_index.txt', dtype=np.int32)
    dires = np.array(sorted(glob(join('/home/llx/deephuman/dataset', '*/*'))))

    num = int(len(dires) * 0.9)
    dires = dires[index][:num]

    dire = np.random.choice(dires)
    print(dire)

    img, min_x, calib, transform, beta, theta = get_data(dire, res, with_normal=True)

    img = img.cuda()
    calib = calib.cuda()
    transform = transform.cuda()

    pred_verts, delta, smpl_faces = implicit_optimize(model, img, 500, calib, transform)
    optimize_verts = pred_verts + delta
    with open('optiimized.obj', 'w') as f:
        for v in optimize_verts:
            f.write('v %f %f %f\n'%(v[0], v[1], v[2]))
        for face in smpl_faces:
            f.write('f %d %d %d\n'%(face[0]+1, face[1]+1, face[2]+1))


if __name__ == "__main__":
    main()