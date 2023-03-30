import copy
import pdb

import cv2
import numpy as np
import os, sys, time
import torch.nn as nn
import torch
import torch.nn.functional as torch_F
import tqdm
from utils import util
from utils import util_vis
from utils.camera import img2cam, to_hom, cam2world, world2cam, cam2img
from easydict import EasyDict as edict

sys.path.append(os.path.join(os.path.dirname(__file__), "../external"))
import random
from utils.util import log, debug
from utils import camera

# from models.cnn_model.encoder import SpatialEncoder
epsilon = 1e-6
from typing import Optional

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01
from utils.custom_functions import \
    RayAABBIntersector, RayMarcher, VolumeRenderer
from einops import rearrange
import vren


class Camera(nn.Module):
    def __init__(
            self,
            opt,
            id: int,
            img_gt: torch.Tensor,
            pose_gt: torch.Tensor,
            kypts2D: torch.Tensor,
            Match_mask: np.array,
            Inlier_mask: np.array,
            f: Optional[float] = None,
            cx: Optional[float] = None,
            cy: Optional[float] = None,
            Intrinsic: Optional[torch.Tensor] = None,
            Extrinsic: Optional[torch.Tensor] = None,
            Depth_omn: Optional[torch.Tensor] = None,
            Normal_omn: Optional[torch.Tensor] = None,
            idx2d_to_3d: Optional[np.array] = None
    ):
        super(Camera, self).__init__()
        '''
        :param opt: config
        :param id: The id number of Camera
        :param img_gt: The image ground truth of the Camera
        :param f: focal length of Camera
        :param cx: x of the principle point
        :param cy: y of the principle point
        :param Intrinsic: Intrinsic Mat of Camera
        :param kypts2D: the extracted kypts
        :param Match_mask: [n-1]长度的list, 每一项[num_mch_i,2]
        :param Inlier_mask: [n-1]长度的list, 每一项[num_mch_i,2]
        :param Extrinsic: 1*6 Lie alg of pose         W2C
        '''
        self.id = id
        self.opt = opt
        if f is not None:
            self.f, self.cx, self.cy = f, cx, cy
            self.intrinsic = torch.from_numpy(torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]]).to(opt.device)).float()
        else:
            self.f, self.cx, self.cy = Intrinsic[0, 0], Intrinsic[0, 2], Intrinsic[1, 2]
            self.intrinsic = Intrinsic
        self.img_gt = img_gt
        # 初始化pose
        self.se3_refine = torch.nn.Parameter(torch.zeros(1, 6).to(self.opt.device), requires_grad=True)
        if Extrinsic is not None:  # init with input
            self.se3_refine.data[0, 0] = Extrinsic[0, 0].to(opt.device)
            self.se3_refine.data[0, 1] = Extrinsic[0, 1].to(opt.device)
            self.se3_refine.data[0, 2] = Extrinsic[0, 2].to(opt.device)
            self.se3_refine.data[0, 3] = Extrinsic[0, 3].to(opt.device)
            self.se3_refine.data[0, 4] = Extrinsic[0, 4].to(opt.device)
            self.se3_refine.data[0, 5] = Extrinsic[0, 5].to(opt.device)

        self.kypts = kypts2D  # [n,2]
        self.mch_msks = Match_mask
        self.inlier_msks = Inlier_mask
        self.idx2d_to_3d = idx2d_to_3d if idx2d_to_3d is not None else -1 * np.ones(self.kypts.shape[0]).astype(int)
        self.depth_omn = util.pad_omn2org(Depth_omn.unsqueeze(0).unsqueeze(0), img_gt, mode="depth").squeeze(1)
        self.normal_omn = util.pad_omn2org(Normal_omn.permute(2, 0, 1).unsqueeze(0), img_gt, mode="normal").permute(0,
                                                                                                                    2,
                                                                                                                    3,
                                                                                                                    1)
        self.mesh_grid = camera.mesh_grid(opt)
        self.pose_gt = pose_gt
        # #----------- init_with gt------------------
        # pose_gt_scale=torch.cat([self.pose_gt[:,:3,:3],self.pose_gt[:,:3,3:]/100],dim=-1)
        # pdb.set_trace()
        # self.se3_refine.data=camera.lie.SE3_to_se3(pose_gt_scale)

    def get_pose(self, var=None):
        if var is not None:
            var.se3_refine = self.se3_refine
            # pose_refine=torch.cat([camera.lie.so3_to_SO3(var.se3_refine[:,:3]),var.se3_refine[:,3:,None]],dim=-1)
            pose_refine = camera.lie.se3_to_SE3(var.se3_refine)
        else:
            w2c_se3 = self.se3_refine
            # pose_refine=torch.cat([camera.lie.so3_to_SO3(w2c_se3[:,:3]),w2c_se3[:,3:,None]],dim=-1)
            pose_refine = camera.lie.se3_to_SE3(w2c_se3)
        pose_refine = pose_refine.to(self.opt.device)
        return pose_refine

    def get_pts3D(self,
                  sdf_func,
                  idxs: Optional[np.array] = None,
                  src_cam_id: Optional[int] = None,
                  mode: Optional[str] = None,
                  ):
        """
        :param sdf_func:
        :param idxs: the idx of kypts2D in this image
        :param src_cam_id: the idx for the source camera
        :return:   返回在世界坐标系下的三维点坐标
        """
        if idxs is not None:
            kypts_select = self.kypts[idxs]
        elif src_cam_id is not None:
            if src_cam_id > self.id:
                src_cam_id_rel = src_cam_id - 1
            else:
                src_cam_id_rel = src_cam_id
            self.mch_msks[src_cam_id_rel].dtype = np.int32
            kypts_select = self.kypts[self.mch_msks[src_cam_id_rel][..., 0]][self.inlier_msks[src_cam_id_rel]]
        ref_kypts = img2cam(to_hom(kypts_select), self.intrinsic.unsqueeze(0))
        center_3D = torch.zeros_like(ref_kypts).to(self.opt.device)
        pose_self = self.get_pose()
        center_3D = cam2world(center_3D, pose_self)
        ray = (cam2world(ref_kypts, pose_self) - center_3D)
        d_surface, sdf_surf, sample_pts, mask_finish = sdf_func.sphere_tracing(center_3D,
                                                                               ray, sdf_func)
        pts_surface = center_3D + ray * d_surface[..., None]
        if mode == "only_center_ray":
            return center_3D, ray, self.mch_msks[src_cam_id_rel][..., 0][self.inlier_msks[src_cam_id_rel]]
        if src_cam_id is not None:
            return pts_surface, mask_finish, sdf_surf, sample_pts, self.mch_msks[src_cam_id_rel][..., 0][
                self.inlier_msks[src_cam_id_rel]]
        elif idxs is not None:
            return pts_surface, mask_finish, sdf_surf, sample_pts

    def proj_cam_i(self,
                   cam_i,
                   sdf_func,
                   mode="kypts",
                   depth: Optional[torch.Tensor] = None,
                   pts_surface: Optional[torch.Tensor] = None,
                   mode_3d: Optional[str] = None):
        '''
        :param cam_i:   source camera
        :param sdf_func:
        :param depth: 为了对于不同的source camera不反复计算 reference的depth，可以传递已算好的depth
        :return:
        '''
        id_cam_i = cam_i.id
        if mode == "kypts":
            if mode_3d == "only_center_ray":
                center_3D, ray, kypts2d_indx = self.get_pts3D(sdf_func, src_cam_id=id_cam_i, mode=mode_3d)
                src_cam_id_rel = id_cam_i if id_cam_i < self.id else id_cam_i - 1
                kypts2d_src = cam_i.kypts[self.mch_msks[src_cam_id_rel][..., 1]][self.inlier_msks[src_cam_id_rel]]
                kypts2d_indx_self = kypts2d_indx
                ret = edict(center=center_3D, ray=ray, kypts_src=kypts2d_src, kypts2d_indx=kypts2d_indx_self)
                return ret
            else:
                pts_surface, mask_finish, sdf_surf, sample_pts, kypts2d_indx = self.get_pts3D(sdf_func,
                                                                                              src_cam_id=id_cam_i)
                xyzx_ = world2cam(pts_surface, cam_i.get_pose())
                uvs_forward = cam2img(xyzx_, cam_i.intrinsic.repeat(xyzx_.shape[0], 1, 1))
                kypts_dp = uvs_forward[..., 2:].squeeze(0)
                uvs_forward = (uvs_forward / (uvs_forward[..., 2:] + epsilon))[..., :2].squeeze(1)  # [n,2]
                src_cam_id_rel = id_cam_i if id_cam_i < self.id else id_cam_i - 1
                ret = edict(uv_proj=uvs_forward, mask_finish=mask_finish, pts_surface=pts_surface, sdf_surf=sdf_surf,
                            sample_pts=sample_pts, kypts_src=cam_i.kypts[self.mch_msks[src_cam_id_rel][..., 1]][
                        self.inlier_msks[src_cam_id_rel]], kypts2d_indx=kypts2d_indx,
                            kypts_depth=kypts_dp)
        elif mode == "depth":
            ret = edict()
            if depth is None:
                ret = self.get_depth(sdf_func=sdf_func)
            else:
                ret.update(edict(depth=depth, pts_surface=pts_surface))
            xyzx_ = world2cam(ret.pts_surface, cam_i.get_pose())
            uvs_forward = cam2img(xyzx_, cam_i.intrinsic.repeat(xyzx_.shape[0], 1, 1))
            uvs_forward = (uvs_forward / (uvs_forward[..., 2:] + epsilon))[..., :2].squeeze(1)  # [n,2]
            ret.update(edict(uv_proj=uvs_forward))

        return ret

    def depth_norm_loss(self,
                        midas_fn,
                        sdf_func=None,
                        depth: Optional[torch.Tensor] = None,
                        normal: Optional[torch.Tensor] = None,
                        vis_depth: Optional[bool] = True):
        ret = edict()
        if depth is None:
            ret = self.get_depth(sdf_func=sdf_func)
        else:
            ret.update(edict(depth=depth, norms=normal))
        if vis_depth == True:
            self.vis_depth_normal(self.opt, depth=ret.depth, normal=ret.norms)
        ret.midas_loss, _, _ == midas_fn()

    def vis_depth_normal(self,
                         opt,
                         sdf_func=None,
                         depth: Optional[torch.Tensor] = None,
                         normal: Optional[torch.Tensor] = None,
                         ):
        if depth is not None:
            depth_np = util_vis.tensor2opencv(opt, depth)

    def get_depth(self,
                  sdf_func,
                  mode="train"):
        if mode == "eval":
            with torch.no_grad():
                pose_self = self.get_pose()
                H, W = self.opt.H, self.opt.W
                H_max, factor = 300, 4
                self.opt.H, self.opt.W = int(self.opt.H // factor), (self.opt.W // factor)
                intr_tem = copy.deepcopy(self.intrinsic)
                intr_tem[0, 0] /= factor
                intr_tem[1, 1] /= factor
                intr_tem[0, 2] /= factor
                intr_tem[1, 2] /= factor
                center, ray = camera.get_center_and_ray(self.opt, pose_self, intr=intr_tem)
                d_points, sdf_surf, sampled_pts, mask_finish = sdf_func.sphere_tracing(center, ray, sdf_func)
                mask_inf = torch.isinf(d_points)
                pts3d = center[~mask_inf] + (d_points[~mask_inf][:, None] * ray[~mask_inf])
                depth = d_points[~mask_inf][:, None]
                normals_value = (sdf_func.gradient(pts3d).norm(dim=-1, keepdim=True))
                normals_vis = torch_F.normalize(sdf_func.gradient(pts3d), dim=-1)
                normals_vis = normals_vis.view(1, self.opt.H, self.opt.W, -1)
                self.opt.H, self.opt.W = self.opt.data.image_size
                ret = edict(depth=depth, mask_finish=mask_finish, norms=normals_vis, pts_surface=pts3d,
                            normals_value=normals_value, sdf_surf=sdf_surf, sample_pts=sampled_pts.detach())
        else:
            pose_self = self.get_pose()
            center, ray = camera.get_center_and_ray(self.opt, pose_self, intr=self.intrinsic)
            d_points, sdf_surf, sampled_pts, mask_finish = sdf_func.sphere_tracing(center, ray, sdf_func)
            mask_inf = torch.isinf(d_points)
            pts3d = center[~mask_inf] + (d_points[~mask_inf][:, None] * ray[~mask_inf])
            depth = d_points[~mask_inf][:, None]
            normals_value = (sdf_func.gradient(pts3d).norm(dim=-1, keepdim=True))
            normals_vis = torch_F.normalize(sdf_func.gradient(pts3d), dim=-1)
            normals_vis = normals_vis.view(1, self.opt.data.image_size[0], self.opt.data.image_size[1], -1)
            ret = edict(depth=depth, mask_finish=mask_finish, norms=normals_vis, pts_surface=pts3d,
                        normals_value=normals_value, sdf_surf=sdf_surf, sample_pts=sampled_pts.detach())
        return ret

    def render(self,
               sdf_func,
               color_func,
               dp_req=True,
               batch_size=None,
               rays_idx=None,
               mode="full_rgb"):
        if rays_idx is None:
            rays_idx = torch.randperm(self.opt.H * self.opt.W, device=self.opt.device)[
                       :self.opt.Renderer.rand_rays // batch_size]
        pose_self = self.get_pose()
        center, ray = camera.get_center_and_ray(self.opt, pose_self, intr=self.intrinsic.unsqueeze(0),
                                                rays_idx=rays_idx, xy_grid=self.mesh_grid)  # [B,HW,3]
        if mode == "full_rgb":
            rgb_gt = self.img_gt.view(3, -1).permute(1, 0).unsqueeze(0)
        else:
            rgb_gt = self.img_gt.view(3, -1)[:, rays_idx].permute(1, 0).unsqueeze(0)
        return center, ray, rgb_gt

    @torch.no_grad()
    def render_img_by_slices(self,
                             sdf_func,
                             color_func,
                             Renderer,
                             pose_input=None
                             ):
        ret = edict()
        if pose_input is None:
            pose_self = self.get_pose()
        else:
            pose_self = pose_input
        center, ray = camera.get_center_and_ray(self.opt, pose_self, intr=self.intrinsic.unsqueeze(0))  # [B,HW,3]
        H, W = self.opt.data.image_size
        ray_bat = self.opt.Renderer.rand_rays
        depths_list = []
        norms_list = []
        rgbs_list = []
        for i in tqdm.tqdm(range(int(center.shape[1] / ray_bat) + 1)):
            start = i * ray_bat
            end = (i + 1) * ray_bat if (i + 1) * ray_bat < center.shape[1] else center.shape[1]

            ret_out = Renderer.forward(opt=self.opt, center=center[:, start:end, :],
                                       ray=ray[:, start:end, :], SDF_Field=sdf_func, Rad_Field=color_func)

            depth_mlp = ret_out["depth_mlp"]
            normal_mlp = ret_out["normal_mlp"]
            rgb = ret_out["rgb"]
            depths_list.append(depth_mlp)
            B, _, _ = depth_mlp.shape
            norms_list.append(normal_mlp.view(B, -1, 3))
            rgbs_list.append(rgb.view(B, -1, 3))
        depth = torch.cat(depths_list, dim=1)
        norm = torch.cat(norms_list, dim=1)
        rgb = torch.cat(rgbs_list, dim=1)
        ret.update(edict(depth=depth, norm=norm, rgb=rgb))
        return ret

    @torch.no_grad()
    def generate_videos_synthesis(self,
                                  sdf_func,
                                  color_func,
                                  Renderer,
                                  N=60):
        opt = self.opt
        # render the novel views
        novel_path = f"{opt.output_path}/novel_view/{self.id}"
        save_path_img = f"{opt.output_path}/novel_view/{self.id}/img"
        save_path_loss = f"{opt.output_path}/novel_view/{self.id}/img_loss"
        os.makedirs(novel_path, exist_ok=True)
        os.makedirs(save_path_img, exist_ok=True)
        os.makedirs(save_path_loss, exist_ok=True)
        pose_self = self.get_pose()
        pose_novel = camera.get_novel_view_poses(self.opt, pose_self[0], N=N, scale=0.1).to(opt.device)
        pose_novel_tqdm = tqdm.tqdm(pose_novel, desc="rendering novel views", leave=False)
        for i, pose in enumerate(pose_novel_tqdm):
            ret_render = self.render_img_by_slices(sdf_func=sdf_func, color_func=color_func,
                                                   pose_input=pose.unsqueeze(0),Renderer=Renderer)
            # img_gt
            rgb_vis = ret_render.rgb.view(self.opt.H, self.opt.W, 3)
            img_gt_vis = self.img_gt[0].detach().permute(1, 2, 0).cpu().numpy() * 255
            img_gt_vis = cv2.cvtColor(img_gt_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path_img + "/{}.png".format(i),
                        cv2.cvtColor(rgb_vis.detach().cpu().numpy() * 255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_path_loss + "/{}.png".format(i),
                        abs(img_gt_vis - (cv2.cvtColor(rgb_vis.detach().cpu().numpy() * 255, cv2.COLOR_RGB2BGR))))
        print("writing videos...")
        rgb_vid_fname = f"{save_path_img}/novel_view_rgb.mp4"
        rgb_loss_vid_fname = f"{save_path_loss}/novel_view_rgb_loss.mp4"
        # -------------------- 
        img_tem = cv2.imread(f"{save_path_img}/0.png")
        imgInfo = img_tem.shape
        size = (imgInfo[1], imgInfo[0])
        f = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
        videoWrite = cv2.VideoWriter(rgb_vid_fname, f, N, size)  # 文件名     编码器   帧率   图片大小
        for i in range(1, N):
            fileName = str(i) + ".png"
            img_path = f"{save_path_img}/{fileName}"
            img = cv2.imread(img_path)
            videoWrite.write(img)
        print("end!")
        # -------------------- 
        img_tem = cv2.imread(f"{save_path_img}/0.png")
        imgInfo = img_tem.shape
        size = (imgInfo[1], imgInfo[0])
        f = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
        videoWrite = cv2.VideoWriter(rgb_loss_vid_fname, f, N, size)  # 文件名     编码器   帧率   图片大小
        for i in range(1, N):
            fileName = str(i) + ".png"
            img_path = f"{save_path_loss}/{fileName}"
            img = cv2.imread(img_path)
            videoWrite.write(img)
        print("end!")


class CameraSet():
    def __init__(self, opt):
        super(CameraSet, self).__init__()
        self.opt = opt
        self.cameras = []
        self.cam_ids = []

    def __len__(self):
        return len(self.cameras)

    def add_camera(self,
                   id: int,  # 与colmap里面camera id对应
                   img_gt: Optional[torch.Tensor] = None,
                   kypts2D: Optional[torch.Tensor] = None,
                   Match_mask: Optional[np.array] = None,
                   Inlier_mask: Optional[np.array] = None,
                   pose_gt: Optional[torch.Tensor] = None,
                   f: Optional[float] = None,
                   cx: Optional[float] = None,
                   cy: Optional[float] = None,
                   Intrinsic: Optional[torch.Tensor] = None,
                   Extrinsic: Optional[torch.Tensor] = None,
                   Depth_omn: Optional[torch.Tensor] = None,
                   Normal_omn: Optional[torch.Tensor] = None,
                   idx2d_to_3d: Optional[np.array] = None,
                   CameraNew: Optional[Camera] = None):
        if CameraNew is None:
            camera_new = Camera(opt=self.opt,
                                id=id,
                                img_gt=img_gt,
                                kypts2D=kypts2D,
                                pose_gt=pose_gt,
                                Match_mask=Match_mask,
                                Inlier_mask=Inlier_mask,
                                Intrinsic=Intrinsic,
                                Depth_omn=Depth_omn,
                                Normal_omn=Normal_omn,
                                Extrinsic=Extrinsic,
                                idx2d_to_3d=idx2d_to_3d)
        else:
            CameraNew.id = id
            camera_new = CameraNew
        self.cam_ids.append(id)
        self.cameras.append(camera_new)

    def __call__(self,  # 返回cam_id为id的相机实体
                 id: int):
        return self.cameras[self.cam_ids.index(id)]

    def get_ren_data(self,
                     sdf_func,
                     color_func,
                     cam_ids=None,
                     dp_req=True,
                     mode="full_rgb",
                     ):
        batch_size = len(self) if cam_ids is None else len(cam_ids)
        centers = []
        rays = []
        rgbs_gt = []
        # 统一sampling
        rays_idx = torch.randperm(self.opt.H * self.opt.W, device=self.opt.device)[
                   :self.opt.Renderer.rand_rays // batch_size]
        if cam_ids is None:
            cam_ids = self.cam_ids
        # --------------get poses+centers+rays+dp_omn+norm_omn -------------------
        for cam_i in [self(id) for id in cam_ids]:
            ct_i, ray_i, rgb_gt_i = cam_i.render(sdf_func=sdf_func, color_func=color_func,
                                                 batch_size=batch_size,
                                                 dp_req=dp_req, rays_idx=rays_idx, mode=mode)
            centers.append(ct_i)
            rays.append(ray_i)
            rgbs_gt.append(rgb_gt_i)
        centers = torch.cat(centers, dim=0)
        rays = torch.cat(rays, dim=0)
        rgbs_gt = torch.cat(rgbs_gt, dim=0)
        return centers, rays, rgbs_gt

    def render(self,
               sdf_func,
               color_func,
               Renderer,
               ret=None,
               mode="train",
               cam_ids=None,
               dp_req=True,
               pose_input=None,
               rgbs_gt=None,
               dps_omn=None,
               norms_omn=None,
               pointset=None
               ):
        if pose_input is None:
            centers, rays, rgbs_gt = self.get_ren_data(cam_ids=cam_ids, dp_req=dp_req,
                                                       sdf_func=sdf_func, color_func=color_func,
                                                       mode="picked_rgb")
        else:
            batch_size = len(self) if cam_ids is None else len(cam_ids)
            rays_idx = torch.randperm(self.opt.H * self.opt.W, device=self.opt.device)[
                       :self.opt.Renderer.rand_rays // batch_size]
            rgbs_gt = rgbs_gt[:, rays_idx, :]
            centers, rays = camera.get_center_and_ray(self.opt, pose_input, intr=self.cameras[0].intrinsic.unsqueeze(0),
                                                      rays_idx=rays_idx,
                                                      xy_grid=self.cameras[0].mesh_grid)

        # -------------multi-view st consistence------------------
        if pointset is not None:
            rand_cam_id = random.randint(0, len(cam_ids) - 1)
            rand_cam_id = cam_ids[rand_cam_id]
            cam_rand = self(rand_cam_id)
            kypts_indx = np.where(cam_rand.idx2d_to_3d != -1)
            pts_surface, mask_finish, sdf_surf, sample_pts = cam_rand.get_pts3D(sdf_func, kypts_indx)
            id_3d = cam_rand.idx2d_to_3d[kypts_indx]
            xyzs = torch.cat(pointset.get_xyzs(idxs=id_3d), dim=0)
            tracing_loss = torch.norm(xyzs - pts_surface[0], dim=-1).mean()
            if getattr(ret, "sdfs", None) is None:
                ret.update(edict(sdfs=sdf_surf))
        else:
            tracing_loss = 0

        # get the volumetric rendering results
        """
        ret={"rgb":rgb,
             "sdfs":sdfs,
             "normals":normals,
             "depth_mlp":depth_mlp,
             "normal_mlp":normal_mlp}
        """
        ret_out = Renderer.forward(opt=self.opt, center=centers,
                               ray=rays, SDF_Field=sdf_func, Rad_Field=color_func)

        ret.update(ret_out)
        depth_mlp = ret["depth_mlp"]
        normal_mlp = ret["normal_mlp"]
        rgb = ret["rgb"]

        d_points, sdf_tracks, _, mask_finish = sdf_func.sphere_tracing(centers.view(1, -1, 3), rays.view(1, -1, 3),
                                                                       sdf_func, iter=0)
        depth_mlp = depth_mlp.view([*centers.shape[:2], 1])
        normal_mlp = normal_mlp.view(*centers.shape)
        rgb = rgb.view(*centers.shape)
        d_points = d_points.view(*depth_mlp.shape)
        sdf_tracks = sdf_tracks.view(*depth_mlp.shape)

        mask_finish = mask_finish.view(*depth_mlp.shape)
        mask_bg = (rgbs_gt.mean(dim=-1) < 0.95) & (rgbs_gt.mean(dim=-1) > 0.05)
        mask_finish = mask_finish & mask_bg.view(*mask_finish.shape)

        if mode == "train":
            with torch.no_grad():
                rgb_loss = torch.abs(rgb - rgbs_gt).mean(dim=-1).detach_()
            if mask_finish.sum() > 0:
                d_consistent = torch_F.smooth_l1_loss(d_points.view(*depth_mlp.shape)[mask_finish],
                                                      depth_mlp[mask_finish], reduction="mean")
                if self.opt.data.dataset not in ["TanksAndTemple", "BlendedMVS", "scannet", "DTU", "llff", "ETH3D",
                                                 "ETH3D_sp"]:
                    weight_dc = torch.exp(-100 * rgb_loss.view(*depth_mlp.shape)[~mask_finish].detach())
                    d_consistent += self.opt.data.unfinish_dc * rgb.shape[0] * (
                            weight_dc.detach_() * torch_F.smooth_l1_loss(
                        d_points.view(*depth_mlp.shape)[~mask_finish], depth_mlp[~mask_finish].detach(),
                        reduce=False)).mean()
            else:
                d_consistent = torch.zeros_like(d_points).mean()
            # print(f"PSNR:{-10 * torch_F.mse_loss(rgb[mask_bg], rgbs_gt[mask_bg]).log10()}")
            PSNR = -10 * torch_F.mse_loss(rgb[mask_bg], rgbs_gt[mask_bg]).log10()
            ret.update(edict(tracing_loss=tracing_loss, mask_bg=mask_bg,
                             rgb_loss=torch_F.l1_loss(rgb, rgbs_gt), DC_loss=d_consistent,
                             PSNR=PSNR))  # [B,HW,K]
        return ret

    def get_all_poses(self, pick_cam_id=None):
        poses_all = []
        poses_gt_all = []
        for cam_i in (self.cameras if pick_cam_id is None else [self(i) for i in pick_cam_id]):
            poses_all.append(cam_i.get_pose())
            poses_gt_all.append(cam_i.pose_gt)
        poses_all = torch.cat(poses_all, dim=0)
        poses_gt_all = torch.cat(poses_gt_all, dim=0)
        return poses_all, poses_gt_all

    def get_all_parameters(self):
        para = {}
        pose_para = camera.lie.SE3_to_se3(self.get_all_poses()[0])
        cam_id = []
        idx2d_to_3ds = []
        for cam_i in self.cameras:
            cam_id.append(cam_i.id)
            idx2d_to_3ds.append(cam_i.idx2d_to_3d)
        para.update({
            "pose_para": pose_para,
            "cam_id": cam_id,
            "idx2d_to_3ds": idx2d_to_3ds
        })
        return para

    @torch.no_grad()
    def prealign_cameras(self, opt, pose, pose_GT):
        # compute 3D similarity transform via Procrustes analysis
        center = torch.zeros(1, 1, 3, device=opt.device)
        center_pred = camera.cam2world(center, pose)[:, 0]  # [N,3]
        center_GT = camera.cam2world(center, pose_GT)[:, 0]  # [N,3]
        try:
            sim3 = camera.procrustes_analysis(center_GT, center_pred)  # 求解一个放射变换，将求解得到的pose与pose_gt进行对齐
        except:
            print("warning: SVD did not converge...")
            sim3 = edict(t0=0, t1=0, s0=1, s1=1, R=torch.eye(3, device=opt.device))
        # align the camera poses
        center_aligned = (center_pred - sim3.t1) / sim3.s1 @ sim3.R.t() * sim3.s0 + sim3.t0
        R_aligned = pose[..., :3] @ sim3.R.t()
        t_aligned = (-R_aligned @ center_aligned[..., None])[..., 0]
        pose_aligned = camera.pose(R=R_aligned, t=t_aligned)
        return pose_aligned, sim3

    @torch.no_grad()
    def eval_poses(self,
                   pick_cam_id=None,
                   mode="normal"):
        poses_all, poses_gt_all = self.get_all_poses(pick_cam_id=pick_cam_id)

        if poses_all.shape[0] > 2:
            pose_aligned, _ = self.prealign_cameras(self.opt, poses_all.float(), poses_gt_all)
            error = self.evaluate_camera_alignment(self.opt, camera.pose.invert(pose_aligned),
                                                   camera.pose.invert(poses_gt_all))
            print("rot_error:{}".format(np.rad2deg(error.R.mean().cpu())))
            print("t_error:{}".format(error.t.mean()))
            r_error = np.rad2deg(error.R.mean().cpu())
            t_error = error.t.mean()
        else:
            rel_pose_gt = camera.pose.compose_pair(camera.pose.invert(poses_gt_all[0]), poses_gt_all[1])
            rel_pose_est = camera.pose.compose_pair(camera.pose.invert(poses_all[0]), poses_all[1])
            t_error = camera.translation_dist(rel_pose_est[:3, 3], rel_pose_gt[:3, 3])
            r_error = camera.rotation_distance(rel_pose_gt[:3, :3], rel_pose_est[:3, :3]) / torch.pi * 180
            print("rot_error:{}".format(r_error))
            print("t_error:{}".format(t_error))
        if mode == "ATE":
            return np.rad2deg(error.R.cpu()), error.t
        else:
            return r_error, t_error

    @torch.no_grad()
    def evaluate_camera_alignment(self, opt, pose_aligned, pose_GT):
        # measure errors in rotation and translation
        R_aligned, t_aligned = pose_aligned.split([3, 1], dim=-1)
        R_GT, t_GT = pose_GT.split([3, 1], dim=-1)
        R_error = camera.rotation_distance(R_aligned, R_GT)
        t_error = (t_aligned - t_GT)[..., 0].norm(dim=-1)
        avg_scale = (torch.norm(t_aligned[..., 0], dim=-1) + torch.norm(t_GT[..., 0], dim=-1)) / 2
        # error = edict(R=R_error, t=t_error / avg_scale)
        error = edict(R=R_error, t=t_error)
        print("ATE:{}".format(torch.sqrt(((t_aligned - t_GT)[..., 0] ** 2).sum(dim=-1).mean())))
        return error
