import numpy as np
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from .hourglass import HGFilter
from .smplx import SMPLX
from .losses import NormalVectorLoss, EdgeLengthLoss, LaplacianLoss, CustomBCELoss
from .gcn import SMPLCOMA, GCNDecoder
from .graphcmr import GraphCMR
from .linear import COMALinear, SurfaceClassifier, SMPLLinear
from .normalnet import GlobalGenerator
from .util import convert_sdf_to_ply, BinarizedF, perspective, orthogonal, index, batch_rodrigues
from .embedder import Embedder, FourierEmbedder
from kaolin.rep import TriangleMesh as tm
from kaolin.metrics.mesh import laplacian_loss as lap_loss
from kaolin.metrics.mesh import point_to_surface


class TotalModel(nn.Module):
    def __init__(self, config):
        super(TotalModel, self).__init__()
        
        if 'hgfilter' in config.keys():
            self.backbone = HGFilter(**config['hgfilter'])
            self.model_type = 'hourglass'

        if 'coma_linear' in config.keys():
            self.coma_linear = COMALinear(**config['coma_linear'])

        if 'surfaceclassifier' in config.keys():
            self.sdf = SurfaceClassifier(**config['surfaceclassifier'])

        if 'smpl' in config.keys():
            self.smpl = SMPLX(**config['smpl'])
            self.faces = self.smpl.faces

        if 'smpl_coma' in config.keys():
            self.smpl_coma = SMPLCOMA(**config['smpl_coma'])
            self.smpl_type = 'coma'
        elif 'smpl_linear' in config.keys():
            self.smpl_linear = SMPLLinear(**config['smpl_linear'])
            self.smpl_type = 'linear'
        elif 'graphcmr' in config.keys():
            self.graphcmr = GraphCMR(**config['graphcmr'])
            self.smpl_type = 'graphcmr'
        else:
            raise NotImplementedError

        self.nloss = NormalVectorLoss(self.faces)
        self.eloss = EdgeLengthLoss(self.faces)

        if 'gcn_decoder' in config.keys():
            self.gcn_decoder = GCNDecoder(**config['gcn_decoder'])

        if 'is_smplx' in config.keys():
            self.is_smplx = config['is_smplx']
        else:
            self.is_smplx = False

        if self.is_smplx:
            verts_weight = torch.ones(10475)
        else:
            verts_weight = torch.ones(6890)
        if 'ignore_index_files' in config.keys():
            for file in config['ignore_index_files']:
                ignore_index = np.loadtxt(file)
                verts_weight[ignore_index] = 0.

        self.register_buffer('verts_weight', verts_weight.unsqueeze(0))

        if 'use_brock' in config.keys():
            self.use_brock = config['use_brock']
        else:
            self.use_brock = False

        self.loss = CustomBCELoss(brock=self.use_brock)

        if 'offset_loss' in config.keys():
            self.offset_loss = config['offset_loss']
        else:
            self.offset_loss = False

        if 'normalnet' in config.keys():
            self.normalnet = GlobalGenerator(**config['normalnet'])
        else:
            self.normalnet = None

        if 'only_z' in config.keys() and config['only_z'] is not None:
            self.only_z = config['only_z']
        else:
            self.only_z = False

        if 'use_origin' in config.keys() and config['use_origin'] is not None:
            self.use_origin = config['use_origin']
        else:
            self.use_origin = False

        if 'z_offset' in config.keys():
            self.z_offset = config['z_offset']
        else:
            self.z_offset = 1.5

        if 'train_delta' in config.keys():
            self.train_delta = config['train_delta']
        else:
            self.train_delta = False

        self.index = index
        if 'projection_mode' in config.keys() and config['projection_mode'] is not None:
            self.projection = eval(config['projection_mode'])
        else:
            self.projection = perspective

        if 'embedder' in config.keys():
            self.embedder = Embedder(**config['embedder'])
        else:
            self.embedder = nn.Identity()

    def save(self, model_path):
        state_dict = self.state_dict().copy()
        for key in list(state_dict.keys()):
            if 'smpl_coma' in key or 'gcn_decoder.L' in key or 'gcn_decoder.U' in key or 'gcn_decoder.D' in key or 'smpl.' in key or 'verts_weight' in key or 'lloss' in key:
                state_dict.pop(key)
        torch.save(state_dict, model_path)

    def reconstruct(self, img, file_name, calib=None, transform=None, level=0.5, N=256, max_batch=32**3):
        with torch.no_grad():
            voxel_origin = [-1, -1.5, -1]
            x_size = 2.0 / (N - 1)
            y_size = 3.0 / (N - 1)
            z_size = 2.0 / (N - 1)

            overall_index = torch.arange(0, N ** 3, 1, dtype=torch.long)
            samples = torch.zeros(N ** 3, 4)

            samples[:, 2] = overall_index % N
            samples[:, 1] = (overall_index.long() // N) % N
            samples[:, 0] = ((overall_index.long() // N) // N) % N

            samples[:, 0] = (samples[:, 0] * x_size) + voxel_origin[0]
            samples[:, 1] = (samples[:, 1] * y_size) + voxel_origin[1]
            samples[:, 2] = (samples[:, 2] * z_size) + voxel_origin[2]

            device = next(self.sdf.parameters()).device
            num_samples = N ** 3
            head = 0

            if self.normalnet is not None:
                image = img[:, :3]
                gt_imgnormal = img[:, 3:]
                pred_normal = self.normalnet(image)
                img = torch.cat([image, pred_normal], dim=1)

            feat, im_feat_list, tmpx, normx = self.backbone(img)
            im_feat = im_feat_list[-1]
            while head < num_samples:
                sample_subset = samples[head:min(head+max_batch, num_samples), 0:3].to(device)
                sample_subset = sample_subset.unsqueeze(0).permute(0, 2, 1).contiguous() # B 3 N
                xyz, homo = self.projection(sample_subset, calib, transform)
                xy = xyz[:, :2, :]
                z = xyz[:, 2:3, :]

                in_img = ((xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)).squeeze(0).detach().cpu().numpy()

                point_local_feat = self.index(im_feat, xy)
                if self.only_z:
                    if self.use_origin:
                        tmp = self.embedder(sample_subset[:,2:3].permute(0,2,1).contiguous()*10).permute(0,2,1).contiguous()
                    else:
                        tmp = self.embedder(((z-self.z_offset)*10).permute(0,2,1).contiguous()).permute(0,2,1).contiguous()
                else:
                    tmp = self.embedder(sample_subset.permute(0,2,1).contiguous()).permute(0,2,1).contiguous()
                    
                point_local_feat = torch.cat([point_local_feat, tmp], dim=1)
                samples[head:min(head+max_batch, num_samples), 3] = self.sdf(point_local_feat).squeeze().detach().cpu()

                samples[head:min(head+max_batch, num_samples), 3][~in_img] = 0

                head += max_batch

            sdf_values = samples[:,3].reshape(N, N, N)
            sdf_values = 1 - sdf_values

            try:
                convert_sdf_to_ply(sdf_values, voxel_origin, [x_size, y_size, z_size], file_name, level)
            except:
                traceback.print_exc()
                print('marching cube error')

    def batch_point_to_surface(self, points, meshes):
        distance = [point_to_surface(p, m) for p,m in zip(points, meshes)]
        return torch.stack(distance)

    def forward(self, img, sample_points, calib=None, transform=None, if_detach=False, gts=None):
        '''
        :param img b c h w
        :param sample_points b 3 n
        :param calib b 4 4
        :param transform b 2 3
        '''
        assert calib is not None
        b = img.shape[0]
        if self.normalnet is not None:
            image = img[:, :3]
            gt_imgnormal = img[:, 3:]
            pred_normal = self.normalnet(image)
            img = torch.cat([image, pred_normal], dim=1)

        feat, im_feat_list, tmpx, normx = self.backbone(img)

        if if_detach:
            feat = feat.detach()

        xyz, homo = self.projection(sample_points, calib, transform)
        
        xy = xyz[:, :2, :].detach()
        z = xyz[:, 2:3, :]
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0) # B N

        intermediate_preds_list = []
        point_local_feat_list = []
        
        if self.only_z:
            if self.use_origin:
                tmp = self.embedder(sample_points[:,2:3].permute(0,2,1).contiguous()*10).permute(0,2,1).contiguous()
            else:
                tmp = self.embedder(((z-self.z_offset)*10).permute(0,2,1).contiguous()).permute(0,2,1).contiguous()
        else:
            tmp = self.embedder(sample_points.permute(0,2,1).contiguous()).permute(0,2,1).contiguous()

        for im_feat in im_feat_list:
            point_local_feat = self.index(im_feat, xy)
            point_local_feat = torch.cat([point_local_feat, tmp], dim=1)
            pred = self.sdf(point_local_feat).squeeze(1)

            point_local_feat_list.append(point_local_feat)
            intermediate_preds_list.append(pred)

        pred_sdf = intermediate_preds_list[-1]

        if self.smpl_type == 'coma':
            latent = self.coma_linear(feat)
            pred_verts = self.smpl_coma.decode(latent)
        elif self.smpl_type == 'linear':
            params = self.smpl_linear(feat)
            param = params[-1]
            pred_theta = param[:, :72]
            pred_theta = torch.cat([torch.zeros_like(param[:,:3]), param[:,3:72]], dim=1)
            pred_beta = param[:, 72:82]
            pred_verts, _ = self.smpl(pred_theta, pred_beta)
        elif self.smpl_type == 'graphcmr':
            pred_verts, pred_rotmat, pred_beta = self.graphcmr(feat)
            pred_verts_smpl, _ = self.smpl(pred_rotmat, pred_beta)
        else:
            raise NotImplementedError

        if self.train_delta:
            verts_delta = self.gcn_decoder(feat)
        else:
            verts_delta = torch.zeros_like(pred_verts)

        trans_verts = pred_verts.detach() + verts_delta * self.verts_weight.unsqueeze(2)

        if gts is not None:
            if len(gts) == 6:
                gt_theta, gt_beta, gt_sdf, gt_offsets, vertices, faces = gts
                gt_verts, _ = self.smpl(gt_theta, gt_beta)
            elif len(gts) == 4:
                gt_verts, gt_sdf, vertices, faces = gts

            if self.smpl_type == 'coma':
                verts_loss = F.l1_loss(pred_verts, gt_verts)
                normal_loss = self.nloss(pred_verts, gt_verts)
                edge_loss = self.eloss(pred_verts, gt_verts)
            elif self.smpl_type == 'linear':
                verts_loss = 0
                for param in params:
                    pred_theta = param[:, :72]
                    pred_theta = torch.cat([torch.zeros_like(param[:,:3]), param[:,3:72]], dim=1)
                    pred_beta = param[:, 72:82]
                    gt_rotmat = batch_rodrigues(gt_theta.reshape(-1,3)).reshape(b, 24, 3, 3)
                    pred_rotmat = batch_rodrigues(pred_theta.reshape(-1,3)).reshape(b, 24, 3, 3)
                    verts_loss = verts_loss + F.mse_loss(pred_rotmat, gt_rotmat) + 0.1 * F.mse_loss(gt_beta, pred_beta)
                verts_loss = verts_loss / len(params)
                normal_loss = torch.zeros_like(verts_loss)
                edge_loss = torch.zeros_like(verts_loss)
            elif self.smpl_type == 'graphcmr':
                gt_rotmat = batch_rodrigues(gt_theta.reshape(-1,3)).reshape(b, 24, 3, 3)
                verts_loss = F.l1_loss(pred_verts, gt_verts) + F.l1_loss(pred_verts_smpl, gt_verts)
                normal_loss = F.mse_loss(pred_rotmat, gt_rotmat)
                edge_loss = F.mse_loss(pred_beta, gt_beta)

            delta_norm_loss = F.mse_loss(verts_delta*self.verts_weight.unsqueeze(2), torch.zeros_like(verts_delta), reduction='mean')

            # combine training may make pred normal worse
            img_normal_loss = torch.zeros_like(delta_norm_loss)

            if self.train_delta:
                device = trans_verts.device
                smpl_faces = torch.from_numpy(self.faces).long().to(device)
                smpl_meshes = [tm.from_tensors(vertices=v, faces=smpl_faces) for v in gt_verts]
                trans_meshes = [tm.from_tensors(vertices=v, faces=smpl_faces) for v in trans_verts]
                laplacian_loss = torch.stack([lap_loss(sc, sm) for sc, sm in zip(smpl_meshes, trans_meshes)]).mean()

                gt_meshes = [tm.from_tensors(vertices=v, faces=f) for v,f in zip(vertices, faces)]
                s2m = self.batch_point_to_surface([sm.vertices for sm in gt_meshes], trans_meshes)
                m2s = self.batch_point_to_surface([sm.vertices for sm in trans_meshes], gt_meshes)
                p2s_loss = m2s.mean() + s2m.mean()
            else:
                laplacian_loss = torch.zeros_like(img_normal_loss)

                p2s_loss = torch.zeros_like(laplacian_loss)

            # ignore_verts_weight = torch.where(self.verts_weight==0., torch.ones_like(self.verts_weight)*10, self.verts_weight)
            # trans_edge_loss = self.eloss(trans_verts, gt_verts, ignore_verts_weight)
            trans_edge_loss = torch.zeros_like(p2s_loss)

            if self.offset_loss:
                offset_loss = F.l1_loss(verts_delta, gt_offsets).mean()
            else:
                offset_loss = torch.zeros_like(delta_norm_loss)


            gamma = (gt_sdf.shape[1] - torch.sum(gt_sdf, dim=1, keepdim=True)) / gt_sdf.shape[1] # ratio of outside
            sdf_loss = 0
            for pred, point_local_feat in zip(intermediate_preds_list, point_local_feat_list):
                sdf_loss += self.loss(pred, gt_sdf, gamma, in_img)

            sdf_loss /= len(intermediate_preds_list)

            trans_verts = trans_verts.permute(0, 2, 1).contiguous() # B 3 N
            xyz, homo = self.projection(trans_verts.detach(), calib, transform)
            xy = xyz[:, :2, :].detach()
            z = xyz[:, 2:3, :]

            in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0) # B N
            weight = in_img * self.verts_weight
            if self.only_z:
                if self.use_origin:
                    tmp = self.embedder(trans_verts[:,2:3].permute(0,2,1).contiguous()*10).permute(0,2,1).contiguous()
                else:
                    tmp = self.embedder(((z-self.z_offset)*10).permute(0,2,1).contiguous()).permute(0,2,1).contiguous()
            else:
                tmp = self.embedder(trans_verts.permute(0,2,1).contiguous()).permute(0,2,1).contiguous()

            point_local_feat = torch.cat([self.index(im_feat_list[-1], xy), tmp], dim=1)
            pred_gcn_sdf = self.sdf(point_local_feat).squeeze(1)

            gcn_sdf_loss = F.l1_loss(pred_gcn_sdf*weight, torch.ones_like(pred_gcn_sdf)*0.5*weight, reduction='mean')

            return verts_loss, normal_loss, edge_loss, sdf_loss, gcn_sdf_loss, delta_norm_loss, laplacian_loss, p2s_loss, offset_loss, img_normal_loss, trans_edge_loss

        else:
            return pred_verts, trans_verts, pred_sdf
