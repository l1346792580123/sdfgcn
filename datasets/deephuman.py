import numpy as np
import cv2
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm
import os
from os.path import join
import math
import trimesh
import pickle
if __name__ == "__main__":
    from data_util import sample_points,  perspective
else:
    from .data_util import  sample_points, perspective

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
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00], ], dtype=np.float32)


class DeepHuman(Dataset):
    def __init__(self, data_path, ratio, index_file, if_train, num_sample, sigma, res, uniform_sample, even_sample, **kwargs):
        self.num_sample = num_sample
        self.sigma = sigma
        self.res = res
        self.uniform_sample = uniform_sample
        self.even_sample = even_sample

        self.dires = np.array(sorted(glob(join(data_path, "*/*"))))
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
        self.obj_names = []
        self.extra_names = []
        cams = []
        root_rots = []
        root_trans = []
        self.offsets = []
        self.betas = []
        self.thetas = []
        for dire in tqdm(self.dires):
            cam_file = join(dire, 'cam.txt')
            self.obj_names.append(join(dire, 'mesh.obj'))
            cam = np.loadtxt(cam_file)
            cams.append(cam)

            extra_name = join(dire, 'extra.npy')

            self.extra_names.append(extra_name)

            # load smpl params
            with open(join(dire, 'smpl_params.txt'), 'r') as fp:
                lines = fp.readlines()
                lines = [l.strip() for l in lines]

                root_mat_data = lines[3].split(' ') + lines[4].split(' ') +\
                    lines[5].split(' ') + lines[6].split(' ')

                root_mat_data = filter(lambda s: len(s)!=0, root_mat_data)
                root_mat = np.reshape(np.array([float(m) for m in root_mat_data]), (4, 4))

                root_rot = root_mat[:3, :3]
                root_tran = root_mat[:3, 3]

                root_rots.append(root_rot)
                root_trans.append(root_tran)


            smpld = pickle.load(open(join(dire, 'smpld.pkl'), 'rb'))
            self.offsets.append(smpld['offsets'])
            self.betas.append(smpld['betas'][:10])
            self.thetas.append(smpld['pose'])

        self.cams = np.array(cams, dtype=np.float32)
        self.root_rots = np.array(root_rots, dtype=np.float32)
        self.root_trans = np.array(root_trans, dtype=np.float32)

        if 'with_normal' in kwargs.keys():
            self.with_normal = kwargs['with_normal']
        else:
            self.with_normal = False

        for dire in self.dires:
            assert os.path.exists(join(dire, 'mesh.obj'))

        # self.coco = np.array(glob('/home/public/coco2017/images/*.jpg'))

    def __len__(self):
        return len(self.dires)

    def __getitem__(self, idx):
        img = cv2.imread(join(self.dires[idx], 'color.jpg'))
        if self.with_normal:
            normal = cv2.imread(join(self.dires[idx], 'ground_rendered_normal.png'))
            
        # print(self.dires[idx])

        extra_data = np.load(self.extra_names[idx], allow_pickle=True).item()
        bbox = extra_data['bbox'].astype(np.int32)
        min_x, min_y, max_x, max_y = bbox

        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        max_x = min(max_x, img.shape[1])
        max_y = min(max_y, img.shape[0])

        cam = self.cams[idx]
        root_rot = self.root_rots[idx]
        root_trans = self.root_trans[idx]

        mesh = trimesh.load(join(self.dires[idx], 'mesh.obj'), process=False, maintain_order=True)
        inv_rot = np.linalg.pinv(root_rot.transpose())
        mesh.vertices = np.matmul(mesh.vertices - root_trans.reshape(1,-1), inv_rot)
        sampled_points, sdf = sample_points(mesh, self.num_sample, self.sigma, None, self.uniform_sample, self.even_sample)
        sampled_points = sampled_points.astype(np.float32)
        sdf = sdf.astype(np.float32)
        
        center_x = int((min_x+max_x)/2)
        h,w,_ = img.shape
        min_x = center_x - h//2
        max_x = center_x + h//2
        assert min_x >= 0 and max_x < w
        img = img[:, min_x:max_x]
        img = cv2.resize(img, (self.res, self.res))

        calib = np.eye(4, dtype=np.float32)

        calib[:3,:3] = root_rot
        calib[:3, 3] = root_trans

        calib = d2c @ cam @ calib

        transform = np.zeros((2,3), np.float32)
        factor = (1080/self.res)
        tmp = self.res // 2
        transform[0,0] = c_fx / factor / tmp
        transform[1,1] = c_fy / factor / tmp
        transform[0,2] = (c_cx - min_x) / factor / tmp - 1.
        transform[1,2] = c_cy / factor / tmp - 1.


        img = (img / 255.).transpose(2,0,1).astype(np.float32)
        if self.with_normal:
            normal = (normal / 255.).transpose(2,0,1).astype(np.float32)
            img = np.concatenate([img, normal], axis=0)

        vertices = np.array(mesh.vertices.copy()).astype(np.float32)
        faces = np.array(mesh.faces.copy()).astype(np.int64)

        offsets = self.offsets[idx]
        beta = self.betas[idx]
        theta = self.thetas[idx]

        return img, sampled_points.transpose(), sdf, offsets, calib, transform, beta, theta, vertices, faces


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    dataset = DeepHuman('/home/llx/deephuman/dataset', 1.0, '/home/llx/cleansdfgcn/data/shuffle_index.txt', True, 10000, 0.04, 512, True, True)
    print(np.abs(dataset.thetas)[:,:3].sum())
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # import sys
    # sys.path.append('..')
    # from models.smpl import SMPL

    # smpl = SMPL('../data/neutral_smpl_with_cocoplus_reg.pkl', '../data/J_regressor_h36m.npy')

    # mean_depth = []
    # max_depth = []
    # min_depth = []
    # min_z = 1e10
    # max_z = -1e10
    # for idx, data in tqdm(enumerate(loader)):
    #     img, sampled_points, sdf, offsets, calib, transform, beta, theta, vertices, faces = data

    #     # img = img.squeeze(0).numpy().transpose(1,2,0)
    #     # img = (img*255).astype(np.uint8)

    #     # tmp = img.copy()

    #     # sampled_points = sampled_points.squeeze(0).numpy()
    #     # calib = calib.squeeze(0).numpy()
    #     # transform = transform.squeeze(0).numpy()
    #     # xyz = perspective(sampled_points, calib, transform)
    #     # xx = ((xyz[0] + 1) * 256).astype(np.int)
    #     # yy = ((xyz[1] + 1) * 256).astype(np.int)

    #     # yy = np.clip(yy, 0,511)
    #     # xx = np.clip(xx, 0,511)

    #     # inside = sdf.squeeze(0).cpu().numpy().astype(np.bool)

    #     # img[yy[inside],xx[inside]] = 255

    #     # cv2.imwrite('test.png', img)

    #     # break

    #     # verts,_ = smpl(theta, beta)
    #     # verts = verts.squeeze(0).numpy().transpose()

    #     # xyz = perspective(verts, calib, transform)
    #     # xx = ((xyz[0] + 1) * 256).astype(np.int)
    #     # yy = ((xyz[1] + 1) * 256).astype(np.int)

    #     # yy = np.clip(yy, 0,511)
    #     # xx = np.clip(xx, 0,511)

    #     # tmp[yy, xx] = 255
    #     # cv2.imwrite('test2.png', tmp)
    #     # break

    #     vertices = vertices.squeeze(0).cpu().numpy()
    #     calib = calib.squeeze(0).numpy()
    #     transform = transform.squeeze(0).numpy()

    #     trans_verts = np.matmul(calib[:3,:3], vertices.transpose()) + calib[:3, 3:4]
    #     min_z = min(min_z, trans_verts[2].min())
    #     max_z = max(max_z, trans_verts[2].max())


    # print(min_z)
    # print(max_z)
