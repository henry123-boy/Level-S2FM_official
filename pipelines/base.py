import os, sys, time
import torch
import torch.utils.tensorboard
import visdom
import importlib
from utils import util
from utils.util import log, debug
from wis3d import Wis3D
import utils.camera as camera
import cv2
import matplotlib.pyplot as plt
from . import Camera
from . import Point3D
from utils.util import extract_mesh
import numpy as np
import torch.nn.functional as torch_F


# ============================ main engine for training and evaluation ============================

class Model():

    def __init__(self, opt):
        super().__init__()
        os.makedirs(opt.output_path, exist_ok=True)

    def load_dataset(self, opt):
        data = importlib.import_module("data.{}".format(opt.data.dataset))
        log.info("loading training data...")
        self.train_data = data.Dataset(opt, split="train")
        self.train_loader = self.train_data.setup_loader(opt, shuffle=True)

    def build_networks(self, opt):
        graph = importlib.import_module("model.{}".format(opt.model))
        log.info("building networks...")
        self.graph = graph.Graph(opt).to(opt.device)

    def setup_optimizer(self, opt):
        return NotImplementedError

    def restore_checkpoint(self, opt):
        epoch_start, iter_start = None, None
        if opt.resume:
            log.info("resuming from previous checkpoint...")
            epoch_start, iter_start = util.restore_checkpoint(opt, self, resume=opt.resume)
        elif opt.load is not None:
            log.info("loading weights from checkpoint {}...".format(opt.load))
            epoch_start, iter_start = util.restore_checkpoint(opt, self, load_name=opt.load)
        else:
            log.info("initializing weights from scratch...")
        self.epoch_start = epoch_start or 0
        self.iter_start = iter_start or 0

    def setup_visualizer(self, opt):
        log.info("setting up visualizers...")
        if opt.tb:
            self.tb = torch.utils.tensorboard.SummaryWriter(log_dir=opt.output_path, flush_secs=10)
        if opt.visdom:
            # check if visdom server is runninng
            is_open = util.check_socket_open(opt.visdom.server, opt.visdom.port)
            retry = None
            while not is_open:
                retry = input("visdom port ({}) not open, retry? (y/n) ".format(opt.visdom.port))
                if retry not in ["y", "n"]: continue
                if retry == "y":
                    is_open = util.check_socket_open(opt.visdom.server, opt.visdom.port)
                else:
                    break
            self.vis = visdom.Visdom(server=opt.visdom.server, port=opt.visdom.port, env=opt.group)
            self.wis3d = Wis3D(f"{opt.output_path}/wis3d", 'Incremental GeoSDF')

    def train(self, opt):
        return NotImplementedError

    def train_epoch(self, opt):
        return NotImplementedError

    def train_iteration(self, opt, var, loader):
        return NotImplementedError

    @torch.no_grad()
    def validate(self, opt, ep=None):
        return NotImplementedError

    @torch.no_grad()
    def log_scalars(self, opt, var, loss, metric=None, step=0, split="train"):
        for key, value in loss.items():
            if key == "all": continue
            if opt.loss_weight[key] is not None:
                self.tb.add_scalar("{0}/loss_{1}".format(split, key), value, step)
        if metric is not None:
            for key, value in metric.items():
                self.tb.add_scalar("{0}/{1}".format(split, key), value, step)
        # log learning rate
        if split == "train":
            if getattr(self, "optim_base", None) is None:
                lr = self.optim.param_groups[0]["lr"]
            elif getattr(self, "optim_base", None) is not None:
                lr = self.optim_base.param_groups[0]["lr"]
                beta__ = torch.exp(self.graph.Disp_RadFiled.beta_ * self.graph.Disp_RadFiled.beta_speed)
                self.tb.add_scalar("{0}/{1}".format(split, "beta"), beta__, step)
            self.tb.add_scalar("{0}/{1}".format(split, "lr"), lr, step)
            self.tb.add_scalar("{0}/{1}".format(split, "beta_"), self.graph.Disp_RadFiled.beta_, step)
        if getattr(var, "pick_ref_ind", None) is not None:
            relative_poses_ind = [var.pick_ref_ind] + var.pick_src_full_inds
            pose_gt = var.poses_gt.to(opt.device)[relative_poses_ind]
            poses_est = self.graph.get_pose(opt, var)[relative_poses_ind]
        else:
            pose_gt = var.pose.to(opt.device)
            poses_est = self.graph.get_pose(opt, var)
        if len(pose_gt) > 2:
            pose_aligned, _ = self.prealign_cameras(opt, poses_est, pose_gt)
            error = self.evaluate_camera_alignment(opt, pose_aligned, pose_gt)
            self.tb.add_scalar("{0}/r_error".format(split), np.rad2deg(error.R.mean().cpu()), step)
            self.tb.add_scalar("{0}/t_error".format(split), error.t.mean(), step)
        else:
            t_error = poses_est[0, :, 3] - poses_est[1, :, 3]
            t_gt = pose_gt[0, :, 3] - pose_gt[1, :, 3]
            tran1_inverse = camera.pose.invert(pose_gt[0, :, :])
            tran2_inverse = camera.pose.invert(pose_gt[1, :, :])
            tran1 = camera.pose.compose_pair(tran1_inverse, pose_gt[1, :, :])
            tran1_inverse_est = camera.pose.invert(poses_est[0, :, :])
            tran1_est = camera.pose.compose_pair(tran1_inverse_est, poses_est[1, :, :])
            t_error = camera.translation_dist(tran1_est[:, 3], tran1[:, 3])
            self.tb.add_scalar("{0}/{1}".format(split, "t_error"), t_error, step)
            r_error = torch.matmul(poses_est[0, :3, :3], torch.inverse(poses_est[1, :3, :3]))
            r_gt = torch.matmul(pose_gt[0, :3, :3], torch.inverse(pose_gt[1, :3, :3]))
            r_error = camera.rotation_distance(r_gt, r_error) / torch.pi * 180
            self.tb.add_scalar("{0}/{1}".format(split, "r_error"), r_error, step)

    @torch.no_grad()
    def vis_geo_rgb(self,
                    opt,
                    cameraset: Camera.CameraSet,
                    new_camera: Camera.Camera,
                    pointset: Point3D.Point3DSet,
                    vis_only: bool = True,
                    cam_only: bool = False):
        os.makedirs("{0}/mesh".format(opt.output_path), exist_ok=True)
        opt.mesh_dir = "{0}/mesh".format(opt.output_path)
        # ---------------------------- vis pts ------------------------------------------
        view_ord = len(cameraset)
        # vis pointset
        pts3d_vis = torch.cat(pointset.get_all_parameters()["xyzs"], dim=0).detach().cpu().numpy()
        sdf_value = abs(
            self.sdf_func.infer_sdf(torch.cat(pointset.get_all_parameters()["xyzs"], dim=0))).detach().cpu().numpy()
        sdf_grad = torch_F.normalize(self.sdf_func.gradient(torch.cat(pointset.get_all_parameters()["xyzs"], dim=0)),
                                     dim=-1)
        sdf_grad = (sdf_grad + 1) / 2
        sdf_grad = sdf_grad.detach().cpu().numpy()
        util.draw_pcd(np.clip(pts3d_vis, -10, 10), f"{opt.output_path}/mesh/{view_ord}_pointcloud_org.ply",
                      np.zeros_like(pts3d_vis))
        MAX_SAMPLES = opt.Res
        sdf_threshold = (self.sdf_func.bound_max.squeeze()[0] - self.sdf_func.bound_min.squeeze()[0]) / 10 / MAX_SAMPLES
        mask_vis = (sdf_value < sdf_threshold.item())
        pts3d_vis = pts3d_vis[mask_vis.squeeze()]
        util.draw_pcd(pts3d_vis, f"{opt.output_path}/mesh/{view_ord}_pointcloud.ply",
                      (sdf_value - sdf_value.min()) / (sdf_value.max() - sdf_value.min()) * sdf_grad)
        # ---------------------------- vis cam ------------------------------------------
        cameras = {}
        for cam_i in cameraset.cameras + [new_camera]:
            camera_i = {"{}".format(cam_i.id): {"K": util.intr2list(cam_i.intrinsic),
                                                "W2C": util.pose2list(cam_i.get_pose().squeeze()),
                                                "img_size": opt.data.image_size}}
            cameras.update(camera_i)
        util.dict2json(os.path.join(opt.mesh_dir, 'cam{:08d}.json'.format(view_ord)), cameras)
        poses_est, poses_gt = cameraset.get_all_poses()
        pose_aligned_gt, _ = cameraset.prealign_cameras(opt, poses_gt, poses_est)
        pose_wis = torch.cat([poses_est.cpu(),
                              torch.tensor([[0, 0, 0, 1]]).repeat(poses_est.shape[0], 1, 1)], dim=1)
        # self.wis3d.add_camera_trajectory(torch.linalg.inv(pose_wis), name=f"{view_ord}_poses_est")
        for cam_i, i in zip(cameraset.cameras, range(pose_aligned_gt.shape[0])):
            camera_i = {"{}".format(cam_i.id): {"K": util.intr2list(cam_i.intrinsic),
                                                "W2C": util.pose2list(pose_aligned_gt[i]),
                                                "img_size": opt.data.image_size}}
            cameras.update(camera_i)

        pose_wis = torch.cat([pose_aligned_gt.cpu(),
                              torch.tensor([[0, 0, 0, 1]]).repeat(poses_est.shape[0], 1, 1)], dim=1)
        # self.wis3d.add_camera_trajectory(torch.linalg.inv(pose_wis), name=f"{view_ord}_poses_gt")
        util.dict2json(os.path.join(opt.mesh_dir, 'cam{:08d}_gt.json'.format(view_ord)), cameras)
        if cam_only == True:
            return
        if vis_only == False:
            # visualize the mesh
            extract_mesh(
                self.sdf_func,
                filepath=os.path.join(opt.mesh_dir, '{:08d}.ply'.format(view_ord)),
                volume_size=3,
                log=log,
                show_progress=True,
                extra_info=None,
                N=512)
            # util_vis.vis_by_wis3d_mesh(self.wis3d, os.path.join(opt.mesh_dir, '{:08d}.ply'.format(view_ord)),
            #                            f"{view_ord}_mesh")

        # visualize the novel view's rgb
        ret = new_camera.get_depth(sdf_func=self.sdf_func, mode="eval")
        ret_render = new_camera.render_img_by_slices(sdf_func=self.sdf_func,
                                                     color_func=self.color_func,
                                                     Renderer=self.Renderer)
        save_path = opt.output_path + "/image/{}".format(new_camera.id)
        os.makedirs(opt.output_path + "/image/{}".format(new_camera.id), exist_ok=True)
        norm_vis = (ret.norms + 1) / 2
        norm_vis_mlp = (ret_render.norm.view(opt.H, opt.W, 3) + 1) / 2
        rgb_vis = ret_render.rgb.view(opt.H, opt.W, 3)
        # depth
        plt.imsave(save_path + "/dp.jpg",
                   ret.depth.view(int(opt.H // 4), int(opt.W // 4), 1).detach().cpu().squeeze(), cmap='viridis')
        plt.imsave(save_path + "/dp_render.jpg", ret_render.depth.view(opt.H, opt.W, 1).detach().cpu().squeeze(),
                   cmap='viridis')
        # img_gt
        cv2.imwrite(save_path + "/rgb_gt.jpg",
                    cv2.cvtColor(new_camera.img_gt[0].detach().permute(1, 2, 0).cpu().numpy() * 255,
                                 cv2.COLOR_RGB2BGR))
        cv2.imwrite(save_path + "/rgb_render.jpg",
                    cv2.cvtColor(rgb_vis.detach().cpu().numpy() * 255, cv2.COLOR_RGB2BGR))
        # norm_vis
        cv2.imwrite(save_path + "/norm.jpg",
                    cv2.cvtColor(norm_vis[0].detach().cpu().numpy() * 255, cv2.COLOR_RGB2BGR))
        cv2.imwrite(save_path + "/norm_mlp.jpg",
                    cv2.cvtColor(norm_vis_mlp.detach().cpu().numpy() * 255, cv2.COLOR_RGB2BGR))

        torch.cuda.empty_cache()
        return

    @torch.no_grad()
    def vis_all_rgb(self,
                    opt,
                    cameraset: Camera.CameraSet):
        save_path = opt.output_path + "/image_all"
        if os.path.exists(save_path) == True:
            os.system(f'rm -rf {save_path}')
        os.makedirs(save_path, exist_ok=True)
        if len(cameraset.cameras) > 3:
            _, pose_error_t = cameraset.eval_poses(mode="ATE")
            mask = pose_error_t < pose_error_t.mean() + pose_error_t.std()
            filter_ids = np.array(cameraset.cam_ids)[~mask.cpu()]
        # _,pose_error_t=cameraset.eval_poses(mode="ATE",pick_cam_id=filter_ids)
        # pose_error=pose_error_t.cpu().numpy()
        filter_ids = np.array(cameraset.cam_ids)
        rgb_error = []
        pts3d_num = []
        cameras_vis = [cameraset(i) for i in filter_ids.tolist()]
        for new_camera in cameras_vis:
            new_camera.generate_videos_synthesis(sdf_func=self.sdf_func,
                                                 color_func=self.color_func,
                                                 N=30, Renderer=self.Renderer)
            # # visualize the novel view's rgb
            ret = new_camera.get_depth(sdf_func=self.sdf_func)
            ret_render = new_camera.render_img_by_slices(sdf_func=self.sdf_func,
                                                         color_func=self.color_func,
                                                         Renderer=self.Render)
            norm_vis = (ret.norms + 1) / 2
            norm_vis_mlp = (ret_render.norm.view(opt.H, opt.W, 3) + 1) / 2
            rgb_vis = ret_render.rgb.view(opt.H, opt.W, 3)
            # depth
            plt.imsave(save_path + "/dp_{}.jpg".format(new_camera.id),
                       ret.depth.view(opt.H, opt.W, 1).detach().cpu().squeeze(), cmap='viridis')
            plt.imsave(save_path + "/dp_render_{}.jpg".format(new_camera.id),
                       ret_render.depth.view(opt.H, opt.W, 1).detach().cpu().squeeze(), cmap='viridis')
            # img_gt
            if_3d_exist = new_camera.idx2d_to_3d != -1
            kypts_vis = new_camera.kypts[if_3d_exist]
            img_gt_vis = new_camera.img_gt[0].detach().permute(1, 2, 0).cpu().numpy() * 255
            img_gt_vis = cv2.cvtColor(img_gt_vis, cv2.COLOR_RGB2BGR)
            for ky0 in kypts_vis.detach().cpu().numpy().astype('int'):
                cv2.circle(img_gt_vis, ky0, radius=2, color=(0, 0, 255), thickness=2)
            pts3d_num.append(if_3d_exist.sum())
            rgb_error.append(abs(new_camera.img_gt[0].permute(1, 2, 0) - rgb_vis).mean())
            cv2.imwrite(save_path + "/rgb_gt_{}.jpg".format(new_camera.id), img_gt_vis)
            cv2.imwrite(save_path + "/rgb_render_{}.jpg".format(new_camera.id),
                        cv2.cvtColor(rgb_vis.detach().cpu().numpy() * 255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_path + "/rgb_loss_{}.jpg".format(new_camera.id),
                        abs(img_gt_vis - (cv2.cvtColor(rgb_vis.detach().cpu().numpy() * 255, cv2.COLOR_RGB2BGR))))

            # norm_vis
            cv2.imwrite(save_path + "/norm_{}.jpg".format(new_camera.id),
                        cv2.cvtColor(norm_vis[0].detach().cpu().numpy() * 255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_path + "/norm_mlp_{}.jpg".format(new_camera.id),
                        cv2.cvtColor(norm_vis_mlp.detach().cpu().numpy() * 255, cv2.COLOR_RGB2BGR))

            torch.cuda.empty_cache()
        return

    @torch.no_grad()
    def visualize(self, opt, var, step=0, split="train"):
        raise NotImplementedError

    def save_checkpoint(self, opt, ep=0, it=0, latest=False):
        util.save_checkpoint(opt, self, ep=ep, it=it, latest=latest)
        if not latest:
            log.info("checkpoint saved: ({0}) {1}, epoch {2} (iteration {3})".format(opt.group, opt.name, ep, it))
