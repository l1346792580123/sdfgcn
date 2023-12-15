import numpy as np
import cv2
import os
import torch
from torch.autograd import grad
from torch.utils.data.dataloader import DataLoader
from kaolin.rep import TriangleMesh as tm
from kaolin.metrics.mesh import laplacian_loss as lap_loss
from kaolin.metrics.mesh import point_to_surface
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from models.totalmodel import TotalModel
from models.smpl import SMPL
from models.losses import LaplacianLoss
from datasets.renderpeople import RenderPeople
import pickle
from os.path import join
import trimesh
from tqdm import tqdm
import yaml
from glob import glob
from models.glrenderer import glRenderer


def get_data(dire, name, res, bg_name, mesh_name, is_smplx):
    img = cv2.imread(join(dire, 'image/%s.png'%name), -1)
    alpha = img[:,:,3]
    img = img[:,:,:3]
    normal = cv2.imread(join(dire, 'normal/%s0001.png'%name))

    index = np.where(alpha==0)
    normal[index] = 0

    img = cv2.resize(img, (res, res))
    normal = cv2.resize(normal, (res, res))

    bg = cv2.imread(bg_name)
    bg = cv2.resize(bg, (res, res))
    index = (normal[:,:,0]>0)&(normal[:,:,1]>0)&(normal[:,:,2]>0)
    bg[index] = img[index]
    img = bg

    img = (img / 255.).transpose(2,0,1).astype(np.float32)
    normal = (normal / 255.).transpose(2,0,1).astype(np.float32)
    img = torch.from_numpy(np.concatenate([img, normal], axis=0)).unsqueeze(0)

    camera = np.load(join(dire, 'camera/%s.npy'%name), allow_pickle=True).item()
    calib = camera['calib']
    transform = camera['transform']

    trans = np.eye(4, dtype=np.float32)
    if is_smplx:
        trans[:3, 3] = camera['smplx_trans']
        smpl_mesh = trimesh.load(mesh_name.replace('tmp.obj', 'smplx.obj'), process=False, maintain_order=True)
    else:
        trans[:3, 3] = camera['trans']
        smpl_mesh = trimesh.load(mesh_name.replace('tmp.obj', 'smpl.obj'), process=False, maintain_order=True)

    smpl_verts = torch.from_numpy(np.array(smpl_mesh.vertices).astype(np.float32)).unsqueeze(0)

    calib = calib@trans

    factor = (1024/res)
    tmp = res // 2
    transform[0,0] = transform[0,0] / factor / tmp
    transform[1,1] = transform[1,1] / factor / tmp
    transform[0,2] = transform[0,2] / factor / tmp - 1.
    transform[1,2] = transform[1,2] / factor / tmp - 1.

    calib = torch.from_numpy(calib).unsqueeze(0)
    transform = torch.from_numpy(transform).unsqueeze(0)

    mesh = trimesh.load(mesh_name, process=False, maintain_order=True)
    mesh.vertices -= trans[:3,3]
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32)).unsqueeze(0)
    faces = torch.from_numpy(np.array(mesh.faces).astype(np.int64)).unsqueeze(0)

    return img, calib, transform, vertices, faces, smpl_verts, index

def subdivide(verts, weight, is_smplx):
    device = verts.device
    if is_smplx:
        edge_unique = np.load('data/smplx_edge_unique.npy') # k 2
        newfaces = np.loadtxt('data/sub_smplx_faces.txt').astype(np.int64)
    else:
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

def optimize(model, img, num_iter, calib, transform):
    model.requires_grad_(False)

    device = img.device

    b = img.shape[0]
    assert b == 1

    if model.is_smplx:
        weight = torch.ones(10475).to(device)
        files = ['data/smplx_lh.txt', 'data/smplx_rh.txt', 'data/smplx_rf.txt', 'data/smplx_lf.txt', 'data/smplx_face.txt']
        # files = ['data/smplx_lh.txt', 'data/smplx_rh.txt', 'data/smplx_face.txt']
    else:
        weight = torch.ones(6890).to(device)
        files = ['data/lh.txt', 'data/rh.txt', 'data/rf.txt', 'data/lf.txt', 'data/face.txt']
        # files = ['data/lh.txt', 'data/rh.txt', 'data/face.txt']
    for file in files:
        index = np.loadtxt(file)
        weight[index] = 0.

    weight = weight.reshape(1,-1,1)

    model.reconstruct(img, 'sdf.ply', calib, transform, level=0.5, N=512, max_batch=64**3)
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
    smpl_face = torch.from_numpy(smpl_faces).long().to(device)

    if model.normalnet is not None:
        image = img[:, :3]
        gt_imgnormal = img[:, 3:]
        pred_normal = model.normalnet(image)
        write_normal = (pred_normal.squeeze(0).detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
        cv2.imwrite('pred_normal.png', write_normal)
        img = torch.cat([image, pred_normal], dim=1)

    feat, im_feat_list, tmpx, normx = model.backbone(img)
    latent = model.coma_linear(feat)
    verts_delta = model.gcn_decoder(feat)
    pred_verts = model.smpl_coma.decode(latent)
    trans_verts = pred_verts + verts_delta * model.verts_weight.unsqueeze(2)
    pred_verts = pred_verts.detach()

    pred_verts, smpl_face, newweight = subdivide(pred_verts, weight, model.is_smplx)
    trans_verts, smpl_face, newweight = subdivide(trans_verts, weight, model.is_smplx)
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
        lap = 10000 * lap_loss(optim_mesh, smpl_mesh)
        loss += offsets + lap

        optim_verts = optim_verts.unsqueeze(0).permute(0,2,1).contiguous() # b 3 n
        xyz, homo = model.projection(optim_verts, calib, transform)
        xy = xyz[:, :2, :].detach()
        z = xyz[:, 2:3, :]
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0) # B N

        if only_z:
            tmp = model.embedder(((z-model.z_offset)*10).permute(0,2,1).contiguous()).permute(0,2,1).contiguous()
            point_local_feat = torch.cat([model.index(im_feat_list[-1], xy), tmp], dim=1)
        else:
            tmp = model.embedder(optim_verts.permute(0,2,1).contiguous()).permute(0,2,1).contiguous()
            point_local_feat = torch.cat([model.index(im_feat_list[-1], xy), tmp], dim=1)

        pred_gcn_sdf = model.sdf(point_local_feat).squeeze(1)
        gcnsdf = 10 * F.l1_loss(pred_gcn_sdf*in_img, torch.ones_like(pred_gcn_sdf)*0.5*in_img, reduction='mean')
        loss += gcnsdf

        if i == 0 or i == num_iter - 1:
            print('gcnsdf:%f'%gcnsdf.item())
            print('offsets:%f'%offsets.item())
            print('lap:%f'%lap.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.requires_grad_(True)

    return pred_verts, delta.detach() * weight, smpl_face

def main():
    pt = 'trained_model/yyyy-MM-dd-hh-mm/'
    config = yaml.load(open(join(pt, 'config.yaml'), 'r'), Loader=yaml.FullLoader)

    model = TotalModel(config['model']).cuda()
    state_dict = torch.load(join(pt, 'model_100000.pth'))

    for key in list(state_dict.keys()):
        if 'smpl_coma' in key or 'gcn_decoder.L' in key or 'gcn_decoder.U' in key or 'v2f' in key or 'smpl.' in key:
            state_dict.pop(key)

    ret = model.load_state_dict(state_dict, strict=False)

    model.eval()

    dire = 'demo_data/'
    res = 512
    name = 'rp_1_3_100k0'
    mesh_name = 'demo_data/mesh/rp_1_3_OBJ/tmp.obj'
    coco = np.array(glob('/home/public/coco2017/images/*.jpg'))
    bg_name = np.random.choice(coco)

    img, calib, transform, vertices, faces, smpl_verts, index = get_data(dire, name, res, bg_name, mesh_name, model.is_smplx)

    img = img.cuda()
    calib = calib.cuda()
    transform = transform.cuda()
    smpl_verts = smpl_verts.cuda()
    
    pred_verts, delta, smpl_face = optimize(model, img, 500, calib, transform)


    pred_verts = pred_verts.squeeze(0).detach().cpu().numpy()
    delta = delta.squeeze(0).detach().cpu().numpy()

    optimize_verts = pred_verts + delta

    with open('optimized.obj', 'w') as f:
        for v in optimize_verts:
            f.write('v %f %f %f\n'%(v[0], v[1], v[2]))
        for face in smpl_face:
            f.write('f %d %d %d\n'%(face[0]+1, face[1]+1, face[2]+1))



if __name__ == "__main__":
    main()