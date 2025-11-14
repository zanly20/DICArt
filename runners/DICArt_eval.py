import sys
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d
from tqdm import tqdm
import wandb
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import get_config
from utils.metrics import rot_diff_degree
from networks.gf_algorithms.discrete_angle import *
from datasets.dataloader import get_data_loaders_from_cfg, process_batch
from networks.gf_algorithms.discrete_number import *
from networks.d3pmnet import D3PM,D3PM_Flow,D3PM_Guassion_Flow
from scipy.spatial.transform import Rotation as R
from datasets.dataloader import vis_cloud


class PoseEstimator(torch.nn.Module):
    def __init__(self, num_parts, init_r, init_t, device, joint_axes=None, rt_k=None):
        super(PoseEstimator, self).__init__()
        self.num_parts = num_parts
        self.device = device
        # self.joint_axes = joint_axes
        # if isinstance(rt_k,list):
        #     self.rt_k = torch.from_numpy(np.array(rt_k)).float().to(device)
        # else:
        #     self.rt_k = rt_k
        self.rot_quat_s = []
        self.tra_s = []
        for idx in range(self.num_parts):
            x, y, z, w = R.from_matrix(init_r[idx].cpu().numpy()).as_quat()
            rot_quat = torch.nn.Parameter(torch.tensor(
                [w, x, y, z], device=device, dtype=torch.float), requires_grad=True)  # q=a+bi+ci+di
            tra = torch.nn.Parameter(init_t[idx].reshape(3,1).clone().detach().to(device), requires_grad=True)
            self.rot_quat_s.append(rot_quat)
            self.tra_s.append(tra)

        self.rot_quat_s = nn.ParameterList([torch.nn.Parameter(torch.tensor( \
            [w, x, y, z], device=device, dtype=torch.float), requires_grad=True) for idx in range(self.num_parts)])
        self.tra_s = nn.ParameterList([torch.nn.Parameter(init_t[idx].reshape(3,1).clone().detach().to(device), requires_grad=True) for idx in range(self.num_parts)])
             
   
    def E_geo(self, x, y):
        try:
            x = x.to(torch.float64)
            y = y.to(torch.float64)
            dist_matrix = torch.cdist(x, y)

            min_dist_x_to_y, _ = torch.min(dist_matrix, dim=1)
            Dxy = torch.mean(min_dist_x_to_y, dim=0)

            min_dist_y_to_x, _ = torch.min(dist_matrix, dim=0)
            Dyy = torch.mean(min_dist_y_to_x, dim=0)

            e_geo = torch.mean(Dxy + Dyy)

            return e_geo

        except:
            print(x.shape, y.shape)
            tensor = torch.tensor(1.0, dtype=torch.float64, device='cuda:0', requires_grad=True)
            return tensor
   
    def E_kin(self, transforms, joint_axes,x, y): 
        distances = []
        for i in range(self.num_parts):
            distances.append(torch.norm(transforms[i].matmul(x[i].T).T - y[i], dim=-1).mean())
       
        distances = sum(distances) / self.num_parts
        
        qj_homo = torch.from_numpy(joint_axes).float().to(self.device) # [parts-1, 2, 4]
        transforms = torch.stack(transforms,axis=0)   # [parts, 4, 4]
        diff = []

        for i in range(self.num_parts-1):
            Tj_q = self.rt_k[0] @ transforms[0] @ qj_homo[0][0].T                          
            Tj1_q = self.rt_k[i+1] @ transforms[i+1] @ qj_homo[i][1].T  
            diff.append((Tj_q - Tj1_q)[:3])
                                     
        norm = (torch.norm(torch.stack(diff), p=2)) / self.num_parts
        e_kin = torch.log(0.1*norm + 1) + distances 
        return e_kin
    
    def forward(self, camera_pts, cad_pts, part_weight):
        all_objective = 0.
        transforms = []
        
        scad_pts = cad_pts.clone()
        scamera_pts = camera_pts.clone()

        scad_pts = [torch.cat([pts.to(self.device), torch.ones(pts.shape[0], 1, device=self.device)], dim=-1) for pts in scad_pts]
        scamera_pts = [torch.cat([pts.to(self.device), torch.ones(pts.shape[0], 1, device=self.device)], dim=-1) for pts in scamera_pts]
        eners = []
        errors = []
        for idx in range(self.num_parts):

            base_r_quat = self.rot_quat_s[idx] / torch.norm(self.rot_quat_s[idx])
            a, b, c, d = base_r_quat[0], base_r_quat[1], base_r_quat[2], base_r_quat[3]  # q=a+bi+ci+di
            base_rot_matrix = torch.stack([1 - 2 * c * c - 2 * d * d, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                                        2 * b * c + 2 * a * d, 1 - 2 * b * b - 2 * d * d, 2 * c * d - 2 * a * b,
                                        2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d,
                                        1 - 2 * b * b - 2 * c * c]).reshape(3, 3)
            base_transform = torch.cat([torch.cat([base_rot_matrix, self.tra_s[idx]], dim=1),
                                        torch.tensor([[0., 0., 0., 1.]], device=self.device)], dim=0).float()
            transforms.append(base_transform)
            cad_base = base_transform.matmul(scad_pts[idx].float().T).T
            camera_base = scamera_pts[idx]
           
            base_objective = self.E_geo(cad_base, camera_base)
            all_objective += part_weight[idx] * base_objective
            # eners.append(base_objective)
           
       
        # eners_tensor = torch.tensor(eners, device=self.device)
        # min_val, max_val = torch.min(eners_tensor),torch.max(eners_tensor)
        # eners_tensor = (eners_tensor-min_val)/(max_val-min_val+1e-8)  # Normalize to [0, 1]
        # eners_softmax = F.softmax(eners_tensor, dim=0)
        # for idx in range(self.num_parts): 
            # errors.append(part_weight[idx] * eners_softmax[idx] * eners[idx])
            # all_objective += eners_softmax[idx] * eners[idx]
        # e_kin = max(errors) * self.E_kin(transforms, self.joint_axes,scad_pts, scamera_pts)
        # errors.append(e_kin)
        # all_objective += e_kin
        transforms = torch.stack(transforms, axis=0)
        return all_objective, None, transforms.detach()



class D3PMTest:

    def __init__(self, cfg, trans_stats=None,pretrained_path=None):
        self.cfg = cfg
        self.device = cfg.device
        self.trans_stats = trans_stats
        self.steps = cfg.sampling_steps
        self.model = D3PM(
            cfg, 
            device=self.device
        ).to(self.device)
        if pretrained_path is not None:
            print(f"Loading pretrained model from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print("Model loaded successfully")
    
    def RotateAnyAxis(self, v1, v2, step):
        ROT = np.identity(4)

        axis = v2 - v1
        axis = axis / math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)

        step_cos = math.cos(step)
        step_sin = math.sin(step)

        ROT[0][0] = axis[0] * axis[0] + (axis[1] * axis[1] + axis[2] * axis[2]) * step_cos
        ROT[0][1] = axis[0] * axis[1] * (1 - step_cos) + axis[2] * step_sin
        ROT[0][2] = axis[0] * axis[2] * (1 - step_cos) - axis[1] * step_sin
        ROT[0][3] = 0

        ROT[1][0] = axis[1] * axis[0] * (1 - step_cos) - axis[2] * step_sin
        ROT[1][1] = axis[1] * axis[1] + (axis[0] * axis[0] + axis[2] * axis[2]) * step_cos
        ROT[1][2] = axis[1] * axis[2] * (1 - step_cos) + axis[0] * step_sin
        ROT[1][3] = 0

        ROT[2][0] = axis[2] * axis[0] * (1 - step_cos) + axis[1] * step_sin
        ROT[2][1] = axis[2] * axis[1] * (1 - step_cos) - axis[0] * step_sin
        ROT[2][2] = axis[2] * axis[2] + (axis[0] * axis[0] + axis[1] * axis[1]) * step_cos
        ROT[2][3] = 0

        ROT[3][0] = (v1[0] * (axis[1] * axis[1] + axis[2] * axis[2]) - axis[0] * (v1[1] * axis[1] + v1[2] * axis[2])) * (1 - step_cos) + \
                    (v1[1] * axis[2] - v1[2] * axis[1]) * step_sin

        ROT[3][1] = (v1[1] * (axis[0] * axis[0] + axis[2] * axis[2]) - axis[1] * (v1[0] * axis[0] + v1[2] * axis[2])) * (1 - step_cos) + \
                    (v1[2] * axis[0] - v1[0] * axis[2]) * step_sin

        ROT[3][2] = (v1[2] * (axis[0] * axis[0] + axis[1] * axis[1]) - axis[2] * (v1[0] * axis[0] + v1[1] * axis[1])) * (1 - step_cos) + \
                    (v1[0] * axis[1] - v1[1] * axis[0]) * step_sin
        ROT[3][3] = 1

        return ROT.T
   
    
    def optimization(self, estimator, camera_pts, cad_pts, part_weight):
        # Force enable gradients even if in no_grad context
        with torch.enable_grad():
            return self._optimization_impl(estimator, camera_pts, cad_pts, part_weight)
    
    def _optimization_impl(self, estimator, camera_pts, cad_pts, part_weight):
        cad_pts = cad_pts.clone()
        camera_pts = camera_pts.clone()
        estimator.rot_quat_s.requires_grad_(True)
        estimator.tra_s.requires_grad_(True)

        lr = 5e-2  
        MAX_EPOCH = 300   # 5e-3
        et_lr = 1e-2
       
        optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=0)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=et_lr)
        best_loss = float('inf')
        transforms_best = None
        best_dif = float('inf')
        current_lr = lr
        transforms = None
        # cad_pts
        
        for iter in range(MAX_EPOCH):
            loss, es, transform = estimator(camera_pts, cad_pts, part_weight)
            if loss.item() < best_loss:
                best_loss = loss.item()
                transforms_best = transform
                
            optimizer.zero_grad()
            loss.backward()
        
            optimizer.step()
            scheduler.step()
            print(f'iter: {iter} loss: {loss.item():.4f}')
        return loss, transforms_best

    def calculate_child_transformation(self, batch_sample, d_normals_canonical, pred_euler_matrix, 
                                 pivotunitvec, heat, mean_parts_gt, parts_angle_deg):
        
        batch_size = pred_euler_matrix.shape[0]
        child_rotation_matrices = []
        child_translations = []
        
        for i in range(batch_size):
            joint_position = (pivotunitvec[i] + mean_parts_gt[i]) * heat[i]
            
            axis_direction = pivotunitvec[i].cpu().numpy()
            
            v1 = joint_position.cpu().numpy()
            v2 = v1 + axis_direction
            
            angle_rad = math.radians(parts_angle_deg[i].item())
            
            rotation_matrix_4x4 = self.RotateAnyAxis(v1=v1, v2=v2, step=angle_rad)
            
            child_rotation_3x3 = rotation_matrix_4x4[:3, :3]
            
            child_translation_3x1 = rotation_matrix_4x4[:3, 3]
            
            child_rotation_matrices.append(child_rotation_3x3)
            child_translations.append(child_translation_3x1)
        
        child_rotation_matrix = torch.tensor(np.array(child_rotation_matrices), dtype=torch.float32).to(self.device)
        child_translation = torch.tensor(np.array(child_translations), dtype=torch.float32).to(self.device)
        
        return child_rotation_matrix, child_translation
    
    def optimize_rotation(self, RT):
        if isinstance(RT, list):
            RT = np.array(RT)
        RT = torch.from_numpy(RT)
        single = False
        if RT.ndim == 2:
            RT = RT.unsqueeze(0)
            single = True

        R = RT[:, :3, :3]
        T = RT[:, :3, 3]

        logR = self.rotation_matrix_log(R)     # [n, 3, 3]
        R_new = self.rotation_matrix_exp(logR)  # [n, 3, 3]

        RT_new = torch.eye(4, device=RT.device).unsqueeze(0).repeat(R.shape[0], 1, 1)
        RT_new[:, :3, :3] = R_new
        RT_new[:, :3, 3] = T

        return RT_new.squeeze(0) if single else RT_new
    
    def rotation_matrix_log(self, R):
        """R: [n, 3, 3] or [3, 3]"""
        if R.ndim == 2:
            R = R.unsqueeze(0)  # [1, 3, 3]

        cos_theta = ((torch.diagonal(R, dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1+1e-6, 1-1e-6)
        theta = torch.acos(cos_theta).unsqueeze(-1).unsqueeze(-1)  # [n, 1, 1]
        log_R = (theta / (4 * torch.sin(theta))) * (R - R.transpose(-1, -2))   
        log_R[torch.isnan(log_R)] = 0
        return log_R.squeeze(0) if log_R.shape[0] == 1 else log_R

    def test_step(self, batch_sample):
        self.model.eval()
        gt_normals_canonical = batch_sample['gt_parts_normals_canonical']
        es_normals_canonical = batch_sample['parts_normals_canonical']
        gt_d_normals_canonical = gt_normals_canonical - es_normals_canonical

        rot_part = batch_sample['zero_mean_gt_pose'][:, :6]
        true_trans = batch_sample['zero_mean_gt_pose'][:, -3:]
        rot_matrix = pytorch3d.transforms.rotation_6d_to_matrix(rot_part)

        pts_feat = self.model.extract_pts_feature(batch_sample).to(self.device)

        with torch.no_grad():
            pred, d_normals_canonical, pivotunitvec, heat, pivotvec, mean_parts_gt = self.model.sample(pts_feat, self.steps)
            canonical_error =  torch.acos(F.cosine_similarity(d_normals_canonical, gt_d_normals_canonical, dim=1))
            canonical_error = canonical_error.cpu().numpy()
            angle_error =  torch.acos(F.cosine_similarity(pivotvec, batch_sample['gt_joint_rpy'], dim=1))
            angle_error = angle_error.cpu().numpy()
            distance_error = torch.norm((pivotunitvec+mean_parts_gt)*heat - batch_sample['gt_joint_xyz'],dim=1,p=2)   
            distance_error = distance_error.cpu().numpy()

            pred_rot_bins = pred[:, :3]
            pred_trans_bins = pred[:, 3:6]  
            pred_euler_angles = euler_angles_from_bins(pred_rot_bins, self.cfg.num_bins)
            pred_euler_matrix = pytorch3d.transforms.euler_angles_to_matrix(pred_euler_angles, "ZYX")
            pred_spherical_angles = bins_to_angles(pred_rot_bins, self.cfg.num_bins, self.cfg.num_bins, self.cfg.num_bins)
            pred_matrix = spherical_angles_to_matrix(pred_spherical_angles)
            pred_trans = bins_to_numbers(pred_trans_bins, self.trans_stats, self.cfg.num_bins)
            diff_angle = rot_diff_degree(rot_matrix, pred_euler_matrix)
            diff_trans = torch.norm(pred_trans - true_trans, p=2, dim=-1)

            pre_normals_canonical = batch_sample['parts_normals_canonical'] + d_normals_canonical
            base_normal = pred_euler_matrix[:, :, 2]
            cosine_sim = F.cosine_similarity(base_normal, pre_normals_canonical, dim=1)
            parts_angle_rad = torch.acos(cosine_sim)
            parts_angle_deg = parts_angle_rad * (180 / math.pi)

            child_rotation_matrix, child_translation = self.calculate_child_transformation(
            batch_sample, d_normals_canonical, pred_euler_matrix, 
            pivotunitvec, heat, mean_parts_gt, parts_angle_deg
        )
            print('child_rotation_matrix:', child_rotation_matrix.shape)
            print('child_translation:', child_translation.shape)
            print('pred_euler_matrix:', pred_euler_matrix.shape)
            print('pred_trans:', pred_trans.shape)

        init_r = [pred_euler_matrix.squeeze(0), child_rotation_matrix.squeeze(0)]
        init_r = torch.stack(init_r, dim=0)
        init_t = [pred_trans.squeeze(0), child_translation.squeeze(0)]
        init_t = torch.stack(init_t, dim=0)
       
            # init_base_rt = np.array(optimize_rotation(pred_rt_list))  
            # init_base_rt = torch.from_numpy(init_base_rt)
            # init_base_r = init_base_rt[...,:3,:3]
            # init_base_t = init_base_rt[...,:3,3]
        n_parts = 1
        transformation = batch_sample['transformation'][0]
        pose_estimator = PoseEstimator(num_parts=n_parts, init_r=init_r[0].unsqueeze(0), init_t=init_t[0].unsqueeze(0),
                                            device=torch.device('cuda:0'))
        
        xyz_camera = [pc.cuda().squeeze(0) for pc in copy.deepcopy(batch_sample['xyz_camera'])]
        xyz_camera = torch.cat(xyz_camera, dim=0).unsqueeze(0)
        cad = [pc.cuda().squeeze(0) for pc in copy.deepcopy(batch_sample['cad'])]  

        cad = torch.cat(cad, dim=0).to(self.device)
        transformation = transformation.to(self.device)
        ones = torch.ones(cad.shape[0], 1, device=self.device)
        cad_homo = torch.cat([cad, ones], dim=1)
        transformed_camera = torch.matmul(cad_homo.double(), transformation.T.double())[:, :3].unsqueeze(0)
        cad1 = cad.unsqueeze(0)
       
        gt_rot_matrix = transformation[:3,:3]
        gt_trans = transformation[:3,3]

        loss, optim_transforms = self.optimization(pose_estimator, transformed_camera, cad1, [1, ])
        optim_transforms = optim_transforms[0]
        print('optim_transforms:', optim_transforms)
        optim_transforms_matrix = optim_transforms[:3,:3]
        optim_transforms_trans = optim_transforms[:3,3]
        optim_diff_angle = rot_diff_degree(gt_rot_matrix.to(self.device), optim_transforms_matrix.to(self.device))
        optim_diff_trans = torch.norm(optim_transforms_trans.to(self.device) - gt_trans.to(self.device), p=2, dim=-1)


        # optim_inv_transformation = np.linalg.inv(optim_transforms.cpu().numpy())
        # pc_1 = np.dot(transformed_camera[0].cpu().numpy(), optim_inv_transformation[:3,:3].T) + optim_inv_transformation[:3, 3]
        # vis_cloud(cloud_canonical=cad1[0].cpu().numpy(), cloud_camera=pc_1)
        # exit()

        # print('original_rot_matrix:', rot_matrix)
        # print('optim_transforms_matrix:', optim_transforms_matrix)
        # print('original_trans:', true_trans)
        # print('optim_transforms_trans:', optim_transforms_trans)

        print('optim_diff_angle:', f'{optim_diff_angle.item():.4f}')
        print('optim_diff_trans:', optim_diff_trans.item())
        print('original_diff_angle:', diff_angle.item())
        print('original_diff_trans:', diff_trans.item())
        # exit()
        return optim_diff_angle, optim_diff_trans, canonical_error, angle_error, distance_error

class D3PMTrainer:
    def __init__(self, cfg, trans_stats=None,pretrained_path=None):
        self.cfg = cfg
        self.device = cfg.device
        self.trans_stats = trans_stats  
        self.steps = cfg.sampling_steps
        self.model = D3PM(
            cfg, 
            num_bins=cfg.num_bins,  
            T=cfg.diffusion_steps,  
            device=self.device
        ).to(self.device)
        if pretrained_path is not None:
            print(f"Loading pretrained model from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print("Model loaded successfully")
        # Optimizer & Scheduler
        self.optimizer = torch.optim.RAdam(
            self.model.parameters(),
            lr=3e-4,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[

                torch.optim.lr_scheduler.ConstantLR(
                    self.optimizer,
                    factor=1.0,  
                    total_iters=int(cfg.n_epochs * 0.4) 
                ),

                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=int(cfg.n_epochs * 0.6),  
                    eta_min=1e-6 
                )
            ],
            milestones=[int(cfg.n_epochs * 0.4)]
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer,
        #     T_max=cfg.n_epochs,
        #     eta_min=2e-6
        # )

    def train_step(self, batch_sample):
        self.model.train()
        
        rot_part = batch_sample['zero_mean_gt_pose'][:, :6]
        trans_part = batch_sample['zero_mean_gt_pose'][:, -3:]

        rot_matrix = pytorch3d.transforms.rotation_6d_to_matrix(rot_part)
        euler_angles = pytorch3d.transforms.matrix_to_euler_angles(rot_matrix, "ZYX")
        spherical_angles = matrix_to_spherical_angles(rot_matrix)
        discretized_angles = get_bin_index(spherical_angles, self.cfg.num_bins, self.cfg.num_bins, self.cfg.num_bins)
        discretized_euler = discretize_euler_angles(euler_angles, self.cfg.num_bins)
        trans_bins = translation_to_bins(trans_part, self.trans_stats, self.cfg.num_bins)

        gt_pose = torch.cat([discretized_euler, trans_bins], dim=1)
        y = gt_pose.to(self.device)
        
        pts_feat = self.model.extract_pts_feature(batch_sample).to(self.device)
        loss,loss_description = self.model.loss(y, pts_feat)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item(),loss_description

    def eval_step(self,cfg, batch_sample):
        self.model.eval()
        
        rot_part = batch_sample['zero_mean_gt_pose'][:, :6]
        true_trans = batch_sample['zero_mean_gt_pose'][:, -3:]

        rot_matrix = pytorch3d.transforms.rotation_6d_to_matrix(rot_part)
        pts_feat = self.model.extract_pts_feature(batch_sample).to(self.device)
        
        try:
            with torch.no_grad():
               
                pred = self.model.sample(pts_feat, steps=self.steps)
           
                pred_rot_bins = pred[:, :3]
                pred_trans_bins = pred[:, 3:6]  
                pred_euler_angles = euler_angles_from_bins(pred_rot_bins, self.cfg.num_bins)
                pred_euler_matrix = pytorch3d.transforms.euler_angles_to_matrix(pred_euler_angles, "ZYX")
                pred_spherical_angles = bins_to_angles(pred_rot_bins, self.cfg.num_bins, self.cfg.num_bins, self.cfg.num_bins)
                pred_matrix = spherical_angles_to_matrix(pred_spherical_angles)
                pred_trans = bins_to_numbers(pred_trans_bins, self.trans_stats, self.cfg.num_bins)
                diff_angle = rot_diff_degree(rot_matrix,pred_euler_matrix).mean().item()
                diff_trans = torch.norm(pred_trans - true_trans, p=2, dim=-1).mean().item()
                return diff_angle, diff_trans
            
        except Exception as e:
            print(f"Error in eval_step: {e}")
            return float('inf'), float('inf')  
        

def train_data(cfg, train_loader, val_loader, test_loader,trans_stats):

    trainer = D3PMTrainer(cfg, trans_stats,pretrained_path=None)
    wandb.watch(trainer.model, log="all", log_freq=100)
    save_dir = "D3PM_6D"
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(cfg.n_epochs):

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        train_losses = []
        
        for batch in pbar:
            batch = process_batch(batch, cfg.device, cfg.pose_mode, mini_batch_size=96, PTS_AUG_PARAMS=None)
            loss,loss_description = trainer.train_step(batch)
            train_losses.append(loss)
            
            pbar.set_postfix({
                "Loss": f"{loss:.4f}",  
                "Details": loss_description   
            })
            
        avg_train_loss = np.mean(train_losses)
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss})
        
        trainer.model.eval()

        if (epoch + 1) % 10 == 0:
            val_diff_angles = []
            val_diff_trans = []
            
            print(f"Running full validation on epoch {epoch}...")
            for val_batch in tqdm(val_loader, desc=f"Epoch {epoch} Full Validation"):
                val_batch = process_batch(val_batch, cfg.device, cfg.pose_mode, mini_batch_size=96, PTS_AUG_PARAMS=None)
                try:
                    diff_angle, diff_trans = trainer.eval_step(val_batch)
                    if diff_angle != float('inf') and diff_trans != float('inf'):
                        val_diff_angles.append(diff_angle)
                        val_diff_trans.append(diff_trans)
                except Exception as e:
                    print(f"Error during validation: {e}")
                    continue
            
            val_mean_angle_diff = sum(val_diff_angles) / len(val_diff_angles) if val_diff_angles else float('inf')
            val_mean_trans_diff = sum(val_diff_trans) / len(val_diff_trans) if val_diff_trans else float('inf')
            
            print(f"Epoch {epoch} Validation Mean Angle Difference: {val_mean_angle_diff:.4f} degrees")
            print(f"Epoch {epoch} Validation Mean Translation Difference: {val_mean_trans_diff:.4f} meters")

            #Test dataloader
            test_diff_angles = []
            test_diff_trans = []
            
            for test_batch in tqdm(test_loader, desc=f"Epoch {epoch} Full Validation"):
                test_batch = process_batch(test_batch, cfg.device, cfg.pose_mode, mini_batch_size=96, PTS_AUG_PARAMS=None)
        
                try:
                    diff_angle, diff_trans = trainer.eval_step(test_batch)
                    if diff_angle != float('inf') and diff_trans != float('inf'):
                        test_diff_angles.append(diff_angle)
                        test_diff_trans.append(diff_trans)
                except Exception as e:
                    print(f"Error during validation: {e}")
                    continue
            
            test_mean_angle_diff = sum(test_diff_angles) / len(test_diff_angles) if test_diff_angles else float('inf')
            test_mean_trans_diff = sum(test_diff_trans) / len(test_diff_trans) if test_diff_trans else float('inf')
            
            print(f"Epoch {epoch} testidation Mean Angle Difference: {test_mean_angle_diff:.4f} degrees")
            print(f"Epoch {epoch} testidation Mean Translation Difference: {test_mean_trans_diff:.4f} meters")

            wandb.log({
                "epoch": epoch, 
                "val_mean_angle_diff": val_mean_angle_diff,
                "val_mean_trans_diff": val_mean_trans_diff,
                "test_mean_angle_diff": test_mean_angle_diff,
                "test_mean_trans_diff": test_mean_trans_diff
            })
            
            save_path = os.path.join(
                save_dir,
                f"epoch_model_epoch_{epoch}_angle_diff_{val_mean_angle_diff:.4f}_trans_diff_{val_mean_trans_diff:.4f}.pt"
            )
            torch.save(trainer.model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch}")
        else:
            wandb.log({
                "epoch": epoch
            })
            
        trainer.scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val Angle Diff {val_mean_angle_diff:.4f} | Val Trans Diff {val_mean_trans_diff:.4f}")
            print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | test Angle Diff {test_mean_angle_diff:.4f} | test Trans Diff {test_mean_trans_diff:.4f}")
        else:
            print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Validation skipped")

            
def Test_data(cfg, test_loader, trans_stats):

    pretrained_path = cfg.pretrained_model_path_test
    Test_Model = D3PMTest(cfg, trans_stats,pretrained_path=pretrained_path)
    Test_Model.model.eval()

    pbar = tqdm(test_loader)
    total_diff_angle = 0.0
    total_diff_trans = 0.0
    total_canonical_error = 0.0
    total_angle_error = 0.0
    total_distance_error = 0.0
    batch_count = 0
    for batch in pbar:
        test_batch = process_batch(batch, cfg.device, cfg.pose_mode, mini_batch_size=96, PTS_AUG_PARAMS=None)
        with torch.no_grad():
            diff_angle, diff_trans, canonical_error, angle_error, distance_error = Test_Model.test_step(test_batch)  
            diff_avg_angle = diff_angle.mean().item()
            diff_avg_trans = diff_trans.mean().item()
            total_diff_angle += diff_avg_angle
            total_diff_trans += diff_avg_trans

            diff_avg_canonical = canonical_error.mean().item()
            diff_avg_anglej = angle_error.mean().item()
            diff_avg_distance = distance_error.mean().item()
            total_canonical_error += diff_avg_canonical 
            total_angle_error += diff_avg_anglej
            total_distance_error += diff_avg_distance


    avg_diff_angle = total_diff_angle / batch_count
    avg_diff_trans = total_diff_trans / batch_count

    avg_diff_canon = total_canonical_error / batch_count
    avg_diff_angj = total_angle_error / batch_count
    avg_diff_dis = total_distance_error / batch_count
    print(f"Avg Angle Diff across all batches: {avg_diff_angle:.4f}")
    print(f"Avg Trans Diff across all batches: {avg_diff_trans:.4f}")
    print(f"Avg Canon Diff across all batches: {avg_diff_canon:.4f}")
    print(f"Avg Angj Diff across all batches: {avg_diff_angj:.4f}")
    print(f"Avg Distance Diff across all batches: {avg_diff_dis:.4f}")


def main():

    cfg = get_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    
    wandb.init(project="test", config={
        "batch_size": 96,
        "n_epochs": 200,
        "learning_rate": 3e-4,
        "encoder": "pointnet2",
        "num_bins": cfg.num_bins,

    })
    data_loaders = get_data_loaders_from_cfg(cfg, ['train', 'val', 'test'])
    train_loader = data_loaders['train_loader'] 
    val_loader = data_loaders['val_loader']   
    test_loader = data_loaders['test_loader'] 
    print('train_set: ', len(train_loader))
    print('val_set: ', len(val_loader))
    print('test_set: ', len(test_loader))
    #trans_stats = [-0.3785014748573303, 0.39416784048080444, -0.4042277932167053, 0.39954620599746704, -0.30842161178588867, 0.7598943710327148]
    trans_stats = get_dataset_translation_min_max(train_loader, cfg)

    if not (cfg.eval or cfg.pred):
        train_data(cfg, train_loader, val_loader,test_loader,trans_stats)
    else:
        Test_data(cfg,test_loader=test_loader,trans_stats=trans_stats)
            
if __name__ == "__main__":
    main()