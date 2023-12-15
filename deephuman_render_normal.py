import os
from os.path import join
import subprocess
import shutil
from glob import glob
import numpy as np
from tqdm import tqdm
import trimesh
import cv2
from models.glrenderer import glRenderer

d2c = np.array([
[ 9.99968867e-01, -6.80652917e-03, -9.22235761e-03, -5.44798585e-02], 
[ 6.69376504e-03,  9.99922175e-01, -1.15625133e-02, -3.41685168e-04], 
[ 9.27761963e-03,  1.14376287e-02,  9.99882174e-01, -2.50539462e-03], 
[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00], ], dtype=np.float32)

# color intrinsic
c_fx = 1063.8987
c_fy = 1063.6822
c_cx = 954.1103
c_cy = 553.2578

def main():
    dires = sorted(glob(join('/home/llx/deephuman/dataset', "*/*")))

    renderer = glRenderer('/home/llx/sdfgcn/data/shaders', 512, 512, render_mode="normal")
    res = 512
    factor = 1080 / res
    persp = np.eye(4, dtype=np.float32)
    persp[0, 0] = c_fx / factor / (res//2)
    persp[1, 1] = c_fy / factor / (res//2)
    persp[2, 2] = 0.
    persp[2, 3] = -1.
    persp[3, 3] = 0.
    persp[3, 2] = 1.

    for dire in tqdm(dires):
        # img = cv2.imread(join(dire, 'color.jpg'))

        extra_data = np.load(join(dire, 'extra.npy'), allow_pickle=True).item()
        bbox = extra_data['bbox'].astype(np.int32)
        min_x, min_y, max_x, max_y = bbox
        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        max_x = min(max_x, 1920)
        max_y = min(max_y, 1080)

        center_x = int((min_x+max_x)/2)
        h, w = 1080, 1920
        min_x = center_x - h//2
        max_x = center_x + h//2

        with open(join(dire, 'smpl_params.txt'), 'r') as fp:
            lines = fp.readlines()
            lines = [l.strip() for l in lines]

            root_mat_data = lines[3].split(' ') + lines[4].split(' ') +\
                lines[5].split(' ') + lines[6].split(' ')

            root_mat_data = filter(lambda s: len(s)!=0, root_mat_data)
            root_mat = np.reshape(np.array([float(m) for m in root_mat_data]), (4, 4))

            root_rot = root_mat[:3, :3]
            root_trans = root_mat[:3, 3]

        try:
            inv_rot = np.linalg.inv(root_rot.transpose())
        except:
            print('pinv')
            inv_rot = np.linalg.pinv(root_rot.transpose())

        mesh = trimesh.load(join(dire, 'mesh.obj'))
        trans_points = np.matmul(mesh.vertices - root_trans.reshape(1,-1), inv_rot)

        trans_mesh = trimesh.Trimesh(vertices=trans_points, faces=mesh.faces)

        cam = np.loadtxt(join(dire, 'cam.txt')).astype(np.float32)
        calib = np.eye(4, dtype=np.float32)
        calib[:3,:3] = root_rot
        calib[:3, 3] = root_trans
        calib = d2c @ cam @ calib


        verts = np.array(trans_mesh.vertices).astype(np.float32) # N 3
        faces = np.array(trans_mesh.faces).astype(np.int32)

        persp[0, 2] = (c_cx-min_x) / factor / (res//2) - 1
        persp[1, 2] = c_cy / factor / (res//2) - 1


        ret = renderer.render(verts, faces, calib, persp)

        ret = cv2.cvtColor(ret, cv2.COLOR_RGB2BGR)
        ret = (ret*255).astype(np.uint8)
        cv2.imwrite(join(dire, "ground_rendered_normal.png"), ret)

        # break



if __name__ == "__main__":
    main()