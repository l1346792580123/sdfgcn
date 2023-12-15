import pickle
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import batch_rodrigues, batch_global_rigid_transformation

class SMPL(nn.Module):
    def __init__(self, model_path, joint_path):
        super(SMPL, self).__init__()
        with open(model_path, 'rb') as f:
            smpl_model = pickle.load(f, encoding='latin1')

        self.register_buffer('J_regressor', torch.FloatTensor(smpl_model['J_regressor'].toarray())) # 24 6890
        self.register_buffer('weights', torch.FloatTensor(smpl_model['weights'])) # 6890 24

        posedirs = np.array(smpl_model['posedirs'])
        num_pose = posedirs.shape[-1]
        posedirs = np.reshape(posedirs, [-1, num_pose]).T
        self.register_buffer('posedirs', torch.FloatTensor(posedirs)) # 207 6890*3

        self.register_buffer('v_template', torch.FloatTensor(smpl_model['v_template'])) # 6890 3

        shapedirs = np.array(smpl_model['shapedirs'])
        num_beta = shapedirs.shape[-1]
        shapedirs = np.reshape(shapedirs, [-1, num_beta]).T
        self.register_buffer('shapedirs', torch.FloatTensor(shapedirs)) # 10  6890*3

        self.faces = smpl_model['f'].astype(np.int32) # 13776 3
        
        self.parents = np.array(smpl_model['kintree_table'])[0].astype(np.int32)
        self.register_buffer('kintree_table', torch.from_numpy(smpl_model['kintree_table'].astype(np.int32)))
        self.id_to_col = {self.kintree_table[1, i].item(): i for i in range(self.kintree_table.shape[1])}
        self.parent = {i: self.id_to_col[self.kintree_table[0,i].item()] for i in range(1,self.kintree_table.shape[1])}
        self.register_buffer('joint_regressor', torch.from_numpy(smpl_model['cocoplus_regressor'].toarray()).float())
        self.register_buffer('J_regressor_extra', torch.from_numpy(np.load(joint_path)).float())

        self.joints_idx = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18, 20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]
        self.j24_to_j14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
        self.h36m_to_j14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
        self.size = smpl_model['v_template'].shape
        self.register_buffer('e3', torch.eye(3).float())


    def save_obj(self, pose, beta, save_path):
        if len(pose.shape) == 1:
            pose = pose.unsqueeze(0)
        if len(beta.shape) == 1:
            beta = beta.unsqueeze(0)
        
        verts, joints = self.forward(pose, beta)
        verts = verts.cpu().squeeze().numpy()

        with open(save_path, 'w') as f:
            for v in verts:
                f.write('v %f %f %f\n'%(v[0], v[1], v[2]))
            for face in self.faces:
                f.write('f %d %d %d\n'%(face[0]+1, face[1]+1, face[2]+1))

        return

    def get_joints(self, verts):
        kps_extra = torch.einsum('bik,ji->bjk', [verts, self.J_regressor_extra])[:,self.h36m_to_j14] # may can improve performance
        kps = torch.einsum('bik,ji->bjk', [verts, self.joint_regressor])
        kps = torch.cat([kps_extra, kps[:,14:]], dim=1)
        return kps

    def forward(self, pose, beta, get_R=False, delta=None):
        '''
        pose: B 24*3 or B 24 3 3
        beta: B 10
        '''
        b = pose.shape[0]
        dev = pose.device
        v_template = self.v_template
        v_shaped = torch.matmul(beta, self.shapedirs).reshape(-1, self.size[0], self.size[1]) + v_template
        J = torch.einsum('bik,ji->bjk', [v_shaped, self.J_regressor])

        if pose.ndimension() == 4:
            R = pose
        elif pose.ndimension() == 2:
            R = batch_rodrigues(pose.reshape(-1,3)).reshape(b, -1, 3, 3)
        else:
            print("error pose shape")
            raise NotImplementedError


        lrotmin = (R[:, 1:, :] - self.e3).reshape(b, -1)

        v_posed = torch.matmul(lrotmin, self.posedirs).reshape(b, self.size[0], self.size[1]) + v_shaped
        if delta is not None:
            v_posed = v_posed + delta
        J_transformed, A = batch_global_rigid_transformation(R, J, self.parents)

        weights = self.weights.expand(b, -1, -1)
        T = torch.matmul(weights, A.reshape(b, 24, 16)).reshape(b, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(b, v_posed.shape[1], 1,device=dev)], dim=2)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))

        verts = v_homo[:, :, :3, 0]

        kps_extra = torch.einsum('bik,ji->bjk', [verts, self.J_regressor_extra])[:,self.h36m_to_j14] # may can improve performance
        kps = torch.einsum('bik,ji->bjk', [verts, self.joint_regressor])
        kps = torch.cat([kps_extra, kps[:,14:]], dim=1)
        if get_R:
            return verts, kps, R
        else:
            return verts,kps