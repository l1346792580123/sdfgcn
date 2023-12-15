import numpy as np
import cv2
import re
from torch.utils.data import Dataset
from glob import glob
from os.path import join
import math
from torch.utils.data.dataloader import DataLoader
import trimesh
if __name__ == "__main__":
    from data_util import sample_points, perspective
else:
    from .data_util import sample_points, perspective


# import logging
# logging.getLogger('trimesh').setLevel(logging.ERROR)

class RenderPeople(Dataset):
    def __init__(self, data_path, res, num_sample, sigma, num_view, if_train, ratio, **kwargs):
        self.num_sample = num_sample
        self.sigma = sigma
        self.res = res
        self.data_path = data_path

        if 'projection_mode' in kwargs.keys() and kwargs['projection_mode'] is not None:
            self.projection_mode = kwargs['projection_mode']
        else:
            self.projection_mode = 'persp'

        if 'is_smplx' in kwargs.keys() and kwargs['is_smplx'] is not None:
            self.is_smplx = kwargs['is_smplx']
        else:
            self.is_smplx = False

        self.num_view = num_view
        if num_view == 1:
            self.imgs = sorted(glob(join(data_path, self.projection_mode+'/image/*100k10.png')))
        else:
            self.imgs = sorted(glob(join(data_path, self.projection_mode+'/image/*.png')))

        self.normals = []
        self.cameras = []
        self.dires = []
        for img_name in self.imgs:
            name = img_name.split('/')[-1][:-4]
            obj_name = re.sub(r'100k\d+', 'OBJ', name)
            self.normals.append(join(data_path, self.projection_mode+'/normal/%s0001.png'%name))
            self.cameras.append(join(data_path, self.projection_mode+'/camera/%s.npy'%name))
            self.dires.append(join(data_path, 'mesh/%s'%obj_name))

        if if_train:
            self.imgs = self.imgs[:int(len(self.imgs)*ratio)]
            self.normals = self.normals[:int(len(self.normals)*ratio)]
            self.cameras = self.cameras[:int(len(self.cameras)*ratio)]
            self.dires = self.dires[:int(len(self.dires)*ratio)]
        else:
            self.imgs = self.imgs[int(len(self.imgs)*ratio):]
            self.normals = self.normals[int(len(self.normals)*ratio):]
            self.cameras = self.cameras[int(len(self.cameras)*ratio):]
            self.dires = self.dires[int(len(self.dires)*ratio):]


        self.coco = np.array(glob('/home/public/coco2017/images/*.jpg'))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx], -1)
        alpha = img[:,:,3]
        img = img[:,:,:3]
        normal = cv2.imread(self.normals[idx])
        camera = np.load(self.cameras[idx], allow_pickle=True).item()

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
        img = np.concatenate([img, normal], axis=0)

        calib = camera['calib']
        transform = camera['transform']

        trans = np.eye(4, dtype=np.float32)
        if self.is_smplx:
            trans[:3, 3] = camera['smplx_trans']
        else:
            trans[:3, 3] = camera['trans']
        calib = calib@trans

        mesh = trimesh.load(join(self.dires[idx], 'tmp.obj'), process=False, maintain_order=True)
        if self.is_smplx:
            mesh.vertices -= camera['smplx_trans']
            smpl_mesh = trimesh.load(join(self.dires[idx], 'smplx.obj'), process=False, maintain_order=True)
            smpl_verts = np.array(smpl_mesh.vertices).astype(np.float32)
        else:
            mesh.vertices -= camera['trans']
            smpl_mesh = trimesh.load(join(self.dires[idx], 'smpl.obj'), process=False, maintain_order=True)
            smpl_verts = np.array(smpl_mesh.vertices).astype(np.float32)

        sampled_points, sdf = sample_points(mesh, self.num_sample, self.sigma, None, True, False)
        sampled_points = sampled_points.astype(np.float32)
        sdf = sdf.astype(np.float32)

        factor = (1024/self.res)
        tmp = self.res // 2
        transform[0,0] = transform[0,0] / factor / tmp
        transform[1,1] = transform[1,1] / factor / tmp
        transform[0,2] = transform[0,2] / factor / tmp - 1.
        transform[1,2] = transform[1,2] / factor / tmp - 1.

        vertices = np.array(mesh.vertices.copy()).astype(np.float32)
        faces = np.array(mesh.faces.copy()).astype(np.int64)

        vertices = np.array(mesh.vertices.copy()).astype(np.float32)
        faces = np.array(mesh.faces.copy()).astype(np.int64)

        return img, sampled_points.transpose(), sdf, calib, transform, smpl_verts, vertices, faces


if __name__ == '__main__':
    from tqdm import tqdm
    dataset = RenderPeople(**{'data_path':'/home/llx/renderpeople', 'res':512, 'num_sample':10000, 
                'sigma':0.03, 'num_view':20, 'if_train':True, 'ratio': 1, 'projection_mode':'ortho'})

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for idx, data in tqdm(enumerate(loader)):
        img, sampled_points, sdf, calib, transform, smpl_verts, vertices, faces = data

        # img = (img.squeeze(0).detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
        # normal = img[:,:,3:]
        # img = img[:,:,:3]

        # smpl_verts = smpl_verts.squeeze(0).detach().cpu().numpy()
        # vertices = vertices.squeeze(0).detach().cpu().numpy()
        # faces = faces.squeeze(0).detach().cpu().numpy()
        # calib = calib.squeeze(0).detach().cpu().numpy()
        # transform = transform.squeeze(0).detach().cpu().numpy()
        # sdf = sdf.squeeze(0).detach().cpu().numpy().astype(np.bool)
        # sampled_points = sampled_points.squeeze(0).detach().cpu().numpy().transpose(1,0)

        # # sampled_points = sampled_points[sdf]

        # with open('tmp.obj', 'w') as f:
        #     for v in vertices:
        #         f.write('v %f %f %f\n'%(v[0], v[1], v[2]))
        #     for face in faces:
        #         f.write('f %d %d %d\n'%(face[0]+1, face[1]+1, face[2]+1))

        # np.savetxt('inside.xyz', sampled_points[sdf])
        # np.savetxt('outside.xyz', sampled_points[~sdf])

        # verts = np.matmul(calib[:3,:3], sampled_points[sdf].transpose(1,0)) + calib[:3, 3:4]
        # # u = (transform[0,0] * verts[0] / verts[2] + transform[0,2] + 1) * 256
        # # v = (transform[1,1] * verts[1] / verts[2] + transform[1,2] + 1) * 256

        # u = (transform[0,0] * verts[0] + transform[0,2] + 1) * 256
        # v = (transform[1,1] * verts[1] + transform[1,2] + 1) * 256

        # print(verts[2].min()-1.5)
        # print(verts[2].max()-1.5)

        # img[v.astype(np.int64), u.astype(np.int64)] = 255

        # cv2.imwrite('test.png', img)

        # cv2.imwrite('normal.png', normal)

        break
