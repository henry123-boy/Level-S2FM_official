import copy
import importlib
import pdb
import numpy as np
import os, sys, time
import torch.nn as nn
import torch
import torch.nn.functional as torch_F
import tqdm
import utils.util as util
from utils.camera import img2cam, to_hom, cam2world, world2cam, cam2img
from easydict import EasyDict as edict

sys.path.append(os.path.join(os.path.dirname(__file__), "../external"))
import utils.camera as camera
from . import Camera
from . import Point3D
from typing import Optional

epsilon = 1e-6


class BA(nn.Module):
    def __init__(self,
                 opt,
                 cameraset: Camera.CameraSet,
                 pointset: Point3D.Point3DSet,
                 sdf_func,
                 color_func,
                 cam_pick_ids: Optional[list] = None,
                 mode="sfm_refine"):
        """
        :param opt:
        :param cameraset: 相机集合
        :param pointset:  三维点的集合
        :param sdf_func:  sdf场
        :param color_func: 颜色场
        :param pts_pick_ids: 相关的三维点索引
        :param cam_pick_ids: 相关相机的索引
        :param mode: 包括  sfm、rad_init、sfm_refine
        """
        super(BA, self).__init__()
        self.opt = opt
        self.mode = mode
        # ----------set optimizer-----------------
        max_iter = opt.optim.ba.max_iter
        if (cam_pick_ids is not None):
            if (len(cam_pick_ids) == 1):
                max_iter = int(max_iter // 2)
        self.max_iter = max_iter
        lr_sdf = opt.optim.ba.lr_sdf
        lr_sdf_end = opt.optim.ba.lr_sdf_end
        lr_pose = opt.optim.ba.lr_pose
        lr_pose_end = opt.optim.ba.lr_pose_end
        lr_color = opt.optim.ba.lr_color
        lr_color_end = opt.optim.ba.lr_color_end
        optim = getattr(torch.optim, opt.optim.algo)
        # init_tem_para: poses
        num_cam = len(cameraset.cameras) if cam_pick_ids is None else len(cam_pick_ids)
        if opt.optim.optim_split == False:  # 旋转和平移用相同的学习率
            self.se3_refine = nn.Parameter(torch.zeros(num_cam, 6).to(opt.device))
            cams_opt = cameraset.cameras if cam_pick_ids is None else [cameraset(i) for i in cam_pick_ids]
            for i in range(num_cam):
                self.se3_refine.data[i:i + 1, :] = cams_opt[i].se3_refine.data
            cameras_para = [{'params': self.parameters(), 'lr': lr_pose}]
        else:
            se3_refine = torch.zeros(num_cam, 6).to(opt.device)
            cams_opt = cameraset.cameras if cam_pick_ids is None else [cameraset(i) for i in cam_pick_ids]
            for i in range(num_cam):
                se3_refine.data[i:i + 1, :] = cams_opt[i].se3_refine.data
            self.r_paras = rot_para(opt, se3_refine[..., :3])
            self.t_paras = trans_para(opt, se3_refine[..., 3:])
            cameras_para = [{'params': self.r_paras.parameters(), 'lr': opt.optim.ba.lr_pose_r},
                            {'params': self.t_paras.parameters(), 'lr': opt.optim.ba.lr_pose_t}]

        # init_tem_para: pts3d
        self.xyzs_all = torch.cat(pointset.get_all_parameters()["xyzs"], dim=0)
        if mode == "rad_init":  # 固定住pose来优化辐射场
            self.optim = optim([{'params': sdf_func.parameters(), 'lr': lr_sdf},
                                {'params': color_func.parameters(), 'lr': lr_color}])
        else:
            self.optim = optim(cameras_para + [{'params': sdf_func.parameters(), 'lr': lr_sdf},
                                               {'params': color_func.parameters(), 'lr': lr_color}])
        # optim_pose = getattr(torch.optim, opt.optim.algo_split)
        # self.optim_pose=optim_pose(cameras_para)

        sched = getattr(torch.optim.lr_scheduler, getattr(opt.optim.sched, "type"))
        self.sched = sched(self.optim, gamma=(lr_sdf_end / lr_sdf) ** (1. / max_iter))
        # -----------pick the pts ---------------
        if cam_pick_ids is not None:
            self.cam_pick_ids = cam_pick_ids
            pts_pick_ids, pose_idx, kypts2D_forward = util.get_idx3d_camset(cameraset=cameraset,
                                                                            cam_id=self.cam_pick_ids)
            self.pts_pick_ids = np.concatenate(pts_pick_ids, axis=0)
            self.pose_idx = np.array(pose_idx)
            self.ba_name = "local_ba"
        else:
            self.cam_pick_ids = [cam_i for cam_i in cameraset.cam_ids]
            pts_pick_ids, pose_idx, kypts2D_forward = util.get_idx3d_camset(cameraset=cameraset,
                                                                            cam_id=self.cam_pick_ids)
            self.pose_idx = np.array(pose_idx)
            self.pts_pick_ids = np.concatenate(pts_pick_ids, axis=0)
            self.ba_name = "global_ba"
        _, _, self.rgbs_gt = cameraset.get_ren_data(sdf_func=sdf_func, color_func=color_func, cam_ids=self.cam_pick_ids,
                                                    dp_req=True)
        self.kypts2D_forward = torch.cat(kypts2D_forward, dim=0)
        self.pointset = pointset
        self.cameraset = cameraset

    def run_ba(self,
               sdf_func,
               color_func,
               Renderer):
        loader = tqdm.trange(self.max_iter, desc=self.ba_name, leave=False)
        MAX_SAMPLES = self.opt.Res
        sdf_threshold = (sdf_func.bound_max.squeeze()[0] - sdf_func.bound_min.squeeze()[0]) / 10 / MAX_SAMPLES
        for it in loader:
            if self.opt.optim.optim_split == True:
                self.se3_refine = torch.cat([self.r_paras(), self.t_paras()], dim=1)
            self.optim.zero_grad()
            ret = edict()
            # ---------------global reproj error------------------------------
            xyzs_new = self.xyzs_all[self.pts_pick_ids]
            xyzs_new, normals_value = sdf_func.get_surface_pts(xyzs_new)
            sdfs = sdf_func.infer_sdf(xyzs_new, mode="ret_sdf").view(-1, 1)
            ret.update(edict(sdfs=sdfs, gradients=normals_value))
            poses_forward = camera.lie.se3_to_SE3(self.se3_refine[self.pose_idx]).to(self.opt.device)
            # xyzs_forward [n,1,3]   poses_forward [n,3,4]    kypts2D_forward [n,2]
            xyzx_forward = world2cam(xyzs_new.unsqueeze(1), poses_forward)  # [n,1,3]
            uvs_forward = cam2img(xyzx_forward, self.cameraset.cameras[0].intrinsic.repeat(xyzx_forward.shape[0], 1, 1))
            uvs_forward = (uvs_forward / (uvs_forward[..., 2:] + epsilon))[..., :2].squeeze(1)  # [n,2]
            # # ---------------------
            mask_surf = abs(sdfs) < 2 * sdf_threshold
            inf_mask = torch.isinf(uvs_forward)
            inf_mask = ((inf_mask[mask_surf.squeeze()][:, 0]) | (inf_mask[mask_surf.squeeze()][:, 1]))
            ret.reproj_loss = 0.5 * ((2 * torch.log(
                1 + torch.norm(uvs_forward - self.kypts2D_forward, dim=-1)[mask_surf.squeeze()] ** 2 / 4))[
                                         ~inf_mask].mean()) + 0.5 * \
                              torch.norm(uvs_forward - self.kypts2D_forward, dim=-1)[mask_surf.squeeze()][
                                  ~inf_mask].mean()
            # print(mask_surf.sum()/len(mask_surf))
            if torch.isinf(ret.reproj_loss).any() == True:
                pdb.set_trace()
            if torch.isnan(ret.reproj_loss).any() == True:
                pdb.set_trace()
            if mask_surf.sum() == 0:
                ret.reproj_loss = 0

            # ---------------------
            reproj_loss_tem = torch.norm(uvs_forward - self.kypts2D_forward, dim=-1)[mask_surf.squeeze()][
                ~inf_mask].mean()
            if self.mode != "sfm":
                pose_input = camera.lie.se3_to_SE3(self.se3_refine).to(self.opt.device) if len(
                    self.cam_pick_ids) == 1 else camera.lie.se3_to_SE3(self.se3_refine).to(self.opt.device).detach()
                # #---------------global rendering---------------------------------
                self.cameraset.render(sdf_func=sdf_func, color_func=color_func,
                                      ret=ret, cam_ids=self.cam_pick_ids,
                                      dp_req=True if self.ba_name == "local_ba" else False,
                                      pose_input=pose_input,
                                      rgbs_gt=self.rgbs_gt.detach_(),
                                      Renderer=Renderer,
                                      pointset=self.pointset)
            loss = self.compute_loss(ret)
            if (loss.reproj_error.item() > 10):
                self.opt.loss_weight.ba.reproj_error = 1
            else:
                self.opt.loss_weight.ba.reproj_error = 0
            # print(torch.norm(uvs_forward - self.kypts2D_forward, dim=-1)[mask_surf.squeeze()].mean())
            loss = self.summarize_loss(self.opt, loss)
            loss.all.backward()

            if self.opt.optim.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)  # set the poses grad
                torch.nn.utils.clip_grad_norm_(sdf_func.parameters(), 5)
                torch.nn.utils.clip_grad_norm_(color_func.parameters(), 10)

            self.optim.step()
            self.sched.step()
            # ----------------print the result--------------------------
            loader.set_postfix(it=it, loss="{:.3f}".format(loss.all))
            # ----------------update the xyzs--------------------------
            self.xyzs_all[self.pts_pick_ids] = xyzs_new.detach()
        print(f"pts3d_ratio:{mask_surf.sum() / len(mask_surf)}")
        self.pointset.update_xyzs(self.pts_pick_ids, xyzs_new.detach())
        for cam_idx, j in zip(self.cam_pick_ids, range(len(self.cam_pick_ids))):
            self.cameraset(cam_idx).se3_refine.data = self.se3_refine[j:j + 1].to(self.opt.device)
        print("reprojection error{}".format(reproj_loss_tem))
        return torch.norm(uvs_forward - self.kypts2D_forward, dim=-1)[mask_surf.squeeze()][~inf_mask].mean()

    def compute_loss(self, ret):
        loss = edict()
        if self.mode != "sfm":
            loss.eikonal_loss = torch_F.l1_loss(torch.norm(ret.normals[ret.mask_bg], dim=-1),
                                                torch.ones_like(torch.norm(ret.normals[ret.mask_bg], dim=-1)))
            loss.rgb = ret.rgb_loss
            loss.DC_Loss = ret.DC_loss
            loss.reproj_error = ret.reproj_loss
            loss.sdf_surf = torch_F.l1_loss(ret.sdfs, torch.zeros_like(ret.sdfs))
            loss.tracing_loss = ret.tracing_loss
        else:
            loss.reproj_error = ret.reproj_loss
            loss.sdf_surf = torch_F.l1_loss(ret.sdfs, torch.zeros_like(ret.sdfs))
            loss.eikonal_loss = torch_F.l1_loss(ret.gradients, torch.ones_like(ret.gradients))
        return loss

    def summarize_loss(self, opt, loss):
        loss_all = 0.
        assert ("all" not in loss)
        # weigh losses
        for key in loss:
            assert (key in opt.loss_weight.ba)
            assert (loss[key].shape == ())
            if opt.loss_weight.ba[key] is not None:
                assert not torch.isinf(loss[key]), "loss {} is Inf".format(key)
                assert not torch.isnan(loss[key]), "loss {} is NaN".format(key)
                loss_all += 10 ** float(opt.loss_weight.ba[key]) * loss[key]
        loss.update(all=loss_all)
        return loss


class rot_para(nn.Module):
    def __init__(self, opt, para_in):
        super(rot_para, self).__init__()
        self.paras = nn.Parameter(copy.deepcopy(para_in).to(opt.device))

    def forward(self):
        return self.paras


class trans_para(nn.Module):
    def __init__(self, opt, para_in):
        super(trans_para, self).__init__()
        self.paras = nn.Parameter(copy.deepcopy(para_in).to(opt.device))

    def forward(self):
        return self.paras
