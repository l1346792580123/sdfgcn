import numpy as np
import trimesh
import time
import sys
sys.path.append('/home/llx/packages/occupancy_networks/') # add check_mesh_contains
from im2mesh.utils.libmesh import check_mesh_contains

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

def get_xyz(verts, w2d):
    pad_ones = np.ones((verts.shape[0],1))
    pad_verts = np.hstack((verts, pad_ones))

    color_verts = np.matmul(d2c, np.matmul(w2d, pad_verts.transpose())).transpose()

    x = color_verts[:,0] / color_verts[:,2] * c_fx + c_cx
    y = color_verts[:,1] / color_verts[:,2] * c_fy + c_cy

    return x, y, color_verts[:,2]

def perspective(points, calibrations, transforms=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [3xN] Tensor of 3D points
    :param calibrations: [4x4] Tensor of projection matrix
    :param transforms: [2x3] Tensor of image transform matrix
    :return: xyz: [3xN] Tensor of xy coordinates in the image plane
    '''
    rot = calibrations[:3, :3]
    trans = calibrations[:3, 3:4]
    homo = trans + np.matmul(rot, points)
    xy = homo[:2, :] / homo[2:3, :]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        xy = shift + np.matmul(scale, xy)

    xyz = np.concatenate([xy, homo[2:3]], axis=0)
    return xyz


def get_root_transform(file_name):
    with open(file_name, 'r') as fp:
        lines = fp.readlines()
        lines = [l.strip() for l in lines]

        root_mat_data = lines[3].split(' ') + lines[4].split(' ') +\
                        lines[5].split(' ') + lines[6].split(' ')
        root_mat_data = filter(lambda s: len(s)!=0, root_mat_data)
        root_mat = np.reshape(np.array([float(m) for m in root_mat_data]), (4, 4))
        root_rot = root_mat[:3, :3]
        root_trans = root_mat[:3, 3]

    return root_rot, root_trans

def sample_points(mesh, num_sample, sigma, sigma2=None, sample_uniform=True, sample_even=True):
    if sample_even:
        surface_points, faces = trimesh.sample.sample_surface_even(mesh, num_sample)
    else:
        surface_points, faces = trimesh.sample.sample_surface(mesh, num_sample)

    if sigma2 is None:
        non_surface_points = surface_points + np.random.normal(scale=sigma, size=surface_points.shape)
    else:
        sigma_points = surface_points[:num_sample//2] + np.random.normal(scale=sigma, size=(num_sample//2, 3))
        sigma2_points = surface_points[num_sample//2:] + np.random.normal(scale=sigma2, size=(num_sample//2,3))
        non_surface_points = np.concatenate([sigma_points, sigma2_points], 0)
    
    if sample_uniform:
        bound = np.abs(mesh.vertices).max()
        random_points = np.random.uniform(-bound, bound, size=(num_sample//8,3))
        sampled_points = np.concatenate([non_surface_points, random_points], 0)
    else:
        sampled_points = non_surface_points

    # inside = mesh.contains(sampled_points)
    start = time.time()
    inside = check_mesh_contains(mesh, sampled_points)
    end = time.time()
    # print(end-start)

    return sampled_points, inside