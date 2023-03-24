import pdb

import cv2
import numpy as np
import os, sys, time
import torch
import torch.nn.functional as torch_F
import tqdm
from utils import util
from utils import util_vis
from easydict import EasyDict as edict
import utils.camera as camera
from . import Camera
from . import Point3D
from typing import Optional
from multiprocessing.dummy import Pool as ThreadPool
import pycolmap


class Initializer():
    def __init__(self,
                 opt,
                 cameraset: Optional[Camera.CameraSet],
                 pointset: Optional[Point3D.Point3DSet],
                 sdf_func,
                 color_func,
                 var,
                 cam_info_reloaded=None
                 ):
        '''
        :param opt: config
        :param cameraset: 
        :param pointset: 
        :param var:
            var.indx_init
            var.imgs_init  [2,3,H,W]
            var.kypts_init: list
            var.intrs_init: [2,3,3]
            var.mchs_init: list
            var.inliers_init: list
        '''
        super(Initializer, self).__init__()
        self.opt = opt
        Depths_omn = getattr(var, "omn_depths", None)
        Normals_omn = getattr(var, "omn_norms", None)
        cam_infos = cam_info_reloaded

        # ------------------ two view pose initialization --------------------------------
        # seen in supp equ (16) of our paper
        if (cam_infos == None):

            # init the two view camera on a sphere
            rad_init = getattr(self.opt.data[f"{self.opt.data.scene}"],
                               "rad_init", self.opt.data.bound_max[0] / 2)
            rad = rad_init

            if self.opt.data.inside == True:
                theta_y = torch.tensor([-np.pi / 4])
                theta_y = theta_y
                theta_x = torch.tensor([0])
                theta_x = theta_x
            else:
                theta_y = torch.tensor([np.pi / 4])
                theta_y = theta_y
                theta_x = torch.tensor([np.pi / 2])
                theta_x = theta_x
            theta_z = torch.tensor([np.pi / 4])
            theta_z = theta_z
            R_z = camera.angle_to_rotation_matrix(theta_z, "Z")
            R_y = camera.angle_to_rotation_matrix(theta_y, "Y")
            R_x = camera.angle_to_rotation_matrix(theta_x, "X")
            w2c_rot = torch.inverse(R_x) @ torch.inverse(R_y) @ torch.inverse(R_z)
            t = w2c_rot @ (torch.tensor(
                [-rad * np.cos(theta_y) * np.cos(theta_z), -rad * np.cos(theta_y) * np.sin(theta_z),
                 -rad * np.sin(theta_y)]).float().view(3, 1))
            w2c_pose = torch.cat([w2c_rot, t], dim=-1)
            se3 = camera.Lie().SE3_to_se3(w2c_pose).float()
            Extrinsic_init_cam0 = se3

            # ---------------2 view initialization-----------------
            id0, id1 = var.indx_init
            rel_id = id1 if id1 < id0 else id1 - 1
            matches = var.mchs_init[0][rel_id]
            kypts0 = var.kypts_init[0][matches[:, 0].astype(int)][var.inliers_init[0][rel_id]].cpu().detach().numpy()
            kypts1 = var.kypts_init[1][matches[:, 1].astype(int)][var.inliers_init[0][rel_id]].cpu().detach().numpy()
            intr = var.intrs_init[0].cpu().detach().numpy()
            f, cx, cy = intr[0, 0], intr[0, 2], intr[1, 2]
            camera_colmap = pycolmap.Camera(model='SIMPLE_PINHOLE', width=self.opt.data.image_size[1],
                                            height=self.opt.data.image_size[0], params=[f, cx, cy], )
            answer = pycolmap.essential_matrix_estimation(kypts0, kypts1, camera_colmap, camera_colmap)
            two_view_rot = torch.from_numpy(pycolmap.qvec_to_rotmat(answer["qvec"]))
            scale_init = getattr(self.opt.data[f"{self.opt.data.scene}"], "scale_init", 1)
            two_view_t = torch.from_numpy(answer["tvec"]).view(3, 1) * scale_init
            rel_pose = torch.cat([two_view_rot, two_view_t], dim=-1).float()
            Extrinsic_init_cam1 = camera.pose.compose_pair(w2c_pose, rel_pose)
            Extrinsic_init_cam1 = camera.Lie().SE3_to_se3(Extrinsic_init_cam1).float()
            Extrinsic_init = [Extrinsic_init_cam0, Extrinsic_init_cam1]

        else:
            Extrinsic_init = [None, None]

        # ------------------------------- create  cameras -----------------------------------------------
        for i in range(2):
            cameraset.add_camera(id=var.indx_init[i],
                                 img_gt=var.imgs_init[i:i + 1],
                                 pose_gt=var.poses_gt[var.indx_init][i:i + 1],
                                 kypts2D=var.kypts_init[i],
                                 Match_mask=var.mchs_init[i],
                                 Inlier_mask=var.inliers_init[i],
                                 Intrinsic=var.intrs_init[i],
                                 Depth_omn=Depths_omn[i] if Depths_omn is not None else None,
                                 Normal_omn=Normals_omn[i] if Normals_omn is not None else None,
                                 Extrinsic=Extrinsic_init[i] if cam_infos == None else cam_infos["pose_para"][i:i + 1],
                                 idx2d_to_3d=None if cam_infos == None else cam_infos["idx2d_to_3ds"][i]
                                 )

        # ------------set optimizer-------------------------
        max_iter = opt.optim.init.max_iter
        self.max_iter = max_iter
        lr_sdf = opt.optim.init.lr_sdf
        lr_sdf_end = opt.optim.init.lr_sdf_end
        lr_color = opt.optim.init.lr_color
        optim = getattr(torch.optim, opt.optim.algo)
        self.optim = optim([{'params': sdf_func.parameters(), 'lr': lr_sdf},
                            {'params': color_func.parameters(), 'lr': lr_color}])

        sched = getattr(torch.optim.lr_scheduler, getattr(opt.optim.sched, "type"))
        self.sched = sched(self.optim, gamma=(lr_sdf_end / lr_sdf) ** (1. / max_iter))

    def run(self,
            cameraset: Camera.CameraSet,
            pointset: Point3D.Point3DSet,
            sdf_func,
            color_func,
            Renderer,
            ref_indx: Optional[int] = 0,
            src_indx: Optional[int] = 1):
        loader = tqdm.trange(self.max_iter, desc="Initialization", leave=False)
        camera0 = cameraset.cameras[ref_indx]
        camera1 = cameraset.cameras[src_indx]
        for it in loader:

            ret = edict()
            self.optim.zero_grad()
            # -------------- explicit match------------------------
            ret0 = camera0.proj_cam_i(camera1, sdf_func, mode="kypts")
            ret1 = camera1.proj_cam_i(camera0, sdf_func, mode="kypts")
            for key in ret0.keys():
                if type(getattr(ret0, key)) is torch.Tensor:
                    setattr(ret, key, torch.cat([getattr(ret0, key).squeeze(0), getattr(ret1, key).squeeze(0)], dim=0))
                elif type(getattr(ret0, key)) is np.ndarray:
                    setattr(ret, key, np.concatenate([getattr(ret0, key), getattr(ret1, key)], axis=0))
            # validate two-view
            poses_gt = torch.cat([camera0.pose_gt, camera1.pose_gt], dim=0)
            if it == 0:
                self.essential_2view(ret1.kypts_src, ret0.kypts_src, camera0.intrinsic, cameraset.get_all_poses()[0],
                                     ret, poses_gt=poses_gt)
            # ------------- implicit match by rendering------------
            cameraset.render(sdf_func=sdf_func,
                             color_func=color_func,
                             ret=ret, Renderer=Renderer)

            loss = self.compute_loss(ret)
            print(loss)
            loss = self.summarize_loss(self.opt, loss)
            loss.all.backward()

            self.optim.step()
            self.sched.step()
        print("----------------final loss-----------------")
        print(loss)
        # ------------update feat tracks&Point3D---------------------
        with torch.no_grad():
            pts_surface = ret.pts_surface.view(2, -1, 3)  # [2,N,3]
            mask_finish = ret.mask_finish.view(2, -1, 1)  # [2,N,1] sdf是否足够小
            diff = (pts_surface[0] - pts_surface[1]).norm(dim=-1)
            pts_avg = (pts_surface[0] + pts_surface[1]) / 2
            rel_diff = diff / (pts_avg).norm(dim=-1)
            if self.opt.Ablate_config.sdf_filter == True:  # 用sdf收敛情况来过滤
                mask_pts3d = (diff < (diff.mean() + 3 * diff.std())) & (mask_finish[0, :, 0] | mask_finish[1, :, 0])
            else:  # 不用sdf收敛情况过滤
                mask_pts3d = (diff < (diff.mean() + 3 * diff.std()))
            print(f"Triangulation ratio {mask_pts3d.sum()}/{len(mask_pts3d)}")
            # pdb.set_trace()
            save_path = self.opt.output_path + "/init_mch"
            os.makedirs(save_path, exist_ok=True)
            if (~mask_pts3d).sum() > 2:
                util_vis.draw_matches(self.opt, camera0.img_gt, camera1.img_gt, ret1.kypts_src[~mask_pts3d],
                                      ret0.kypts_src[~mask_pts3d],
                                      store_path=save_path + f"/{camera0.id}_{camera1.id}_iter{it}_filter.jpg",
                                      vis_num=100)
            # pycolmap.triangulate_points()
            util_vis.draw_matches(self.opt, camera0.img_gt, camera1.img_gt, ret1.kypts_src, ret0.kypts_src,
                                  store_path=save_path + f"/{camera0.id}_{camera1.id}_iter{it}_org.jpg")

            kypts2d_indx = ret.kypts2d_indx.reshape(2, -1)[:, mask_pts3d.cpu().numpy()]
            feat_track = [[(i, kypts2d_indx[i, j]) for i in range(2)] for j in range(kypts2d_indx.shape[-1])]
            pts_avg = pts_avg.detach()[mask_pts3d]
            pts_avg = list(torch.split(pts_avg, 1, dim=0))
            pts_indx = []
            for pts_avg_i, feat_track_i in zip(pts_avg, feat_track):
                pts_indx.append(pointset.add_point3d(pts_avg_i, feat_track_i))
            # -------------update 2d-3d matches---------------------
            camera0.idx2d_to_3d[kypts2d_indx[0]] = pts_indx
            camera1.idx2d_to_3d[kypts2d_indx[1]] = pts_indx
        # --------- eval poses estimation----------------------------
        cameraset.eval_poses()

    def essential_2view(self, kypts0, kypts1, intr, poses, ret, poses_gt=None):
        kypts0 = kypts0.cpu().detach().numpy()
        kypts1 = kypts1.cpu().detach().numpy()
        intr = intr.cpu().detach().numpy()
        f, cx, cy = intr[0, 0], intr[0, 2], intr[1, 2]
        camera_colmap = pycolmap.Camera(model='SIMPLE_PINHOLE', width=self.opt.data.image_size[1],
                                        height=self.opt.data.image_size[0], params=[f, cx, cy], )
        answer = pycolmap.essential_matrix_estimation(kypts0, kypts1, camera_colmap, camera_colmap)
        if answer["success"] == False:
            loss_rel1 = loss_rel2 = 0
        else:
            two_view_rot = pycolmap.qvec_to_rotmat(answer["qvec"])
            rel_pose_est = camera.pose.compose_pair(camera.pose.invert(poses[0]), poses[1])
            loss_rel1 = (1 - torch.sum(torch_F.normalize(rel_pose_est[:, 3:], dim=0) * torch_F.normalize(
                torch.from_numpy(answer["tvec"]).cuda()[:, None].float(), dim=0)))
            loss_rel2 = torch_F.smooth_l1_loss(rel_pose_est[:3, :3], torch.from_numpy(two_view_rot).cuda().float(),
                                               reduction="sum")
            # print the eval
            rel_pose_gt = camera.pose.compose_pair(camera.pose.invert(poses_gt[0]), poses_gt[1])
            t_error = camera.translation_dist(rel_pose_gt[:3, 3], torch.from_numpy(answer["tvec"]).cuda())
            r_error = camera.rotation_distance(rel_pose_gt[:3, :3].cpu().float(),
                                               torch.from_numpy(two_view_rot[:3, :3]).float()) / torch.pi * 180
            print(f"5 points algo rot_error:{r_error}")
            print(f"5 points algo translation_error:{t_error}")
            t_error = camera.translation_dist(rel_pose_gt[:3, 3], rel_pose_est[:3, 3])
            r_error = camera.rotation_distance(rel_pose_gt[:3, :3], rel_pose_est[:3, :3]) / torch.pi * 180
            print(f"our algo rot_error:{r_error}")
            print(f"our algo translation_error:{t_error}")
        ret.update(rel_loss=(5 * loss_rel1 + loss_rel2 * 20))

    def compute_loss(self, ret):
        loss = edict()
        # loss.rel_loss=ret.rel_loss

        loss.reproj_error = torch.norm(ret.uv_proj - ret.kypts_src, dim=-1).mean()
        loss.sdf_surf = torch_F.l1_loss(ret.sdf_surf, torch.zeros_like(ret.sdf_surf))
        loss.eikonal_loss = torch_F.l1_loss(torch.norm(ret.normals, dim=-1),
                                            torch.ones_like(torch.norm(ret.normals, dim=-1)))
        loss.rgb = ret.rgb_loss
        loss.DC_Loss = ret.DC_loss
        return loss

    def summarize_loss(self, opt, loss):
        loss_all = 0.
        assert ("all" not in loss)
        # weigh losses
        for key in loss:
            assert (key in opt.loss_weight.init)
            assert (loss[key].shape == ())
            if opt.loss_weight.init[key] is not None:
                assert not torch.isinf(loss[key]), "loss {} is Inf".format(key)
                assert not torch.isnan(loss[key]), "loss {} is NaN".format(key)
                loss_all += 10 ** float(opt.loss_weight.init[key]) * loss[key]
        loss.update(all=loss_all)
        return loss
