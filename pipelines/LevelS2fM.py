import importlib
import pdb
import numpy as np
import os, sys, time
import torch
import tqdm
import utils.util as util
from easydict import EasyDict as edict

from utils.util import log
import utils.camera as camera
from . import base
from . import Camera
from . import Point3D
from . import Initialization
from . import Registration
from . import BA
from . import rendering_refine
from . import Initialization_Trad
from . import BA_Trad
from . import Registration_Trad

@torch.no_grad()
def slerp(pose0, pose1, t):
    import pyquaternion
    quat0 = pyquaternion.Quaternion._from_matrix(matrix=pose0[:3,:3].cpu().numpy(),rtol=1e-5, atol=1e-5)
    quat1 = pyquaternion.Quaternion._from_matrix(matrix=pose1[:3,:3].cpu().numpy(),rtol=1e-5, atol=1e-5)
    quatt = pyquaternion.Quaternion.slerp(quat0, quat1, t)
    R = torch.tensor(quatt.rotation_matrix,dtype=pose0.dtype,device=pose0.device)
    T = (1 - t) * pose0[:3,3] + t * pose1[:3,3]
    return torch.cat([R, T[None,:].T], dim=1)
class Model(base.Model):

    def __init__(self, opt):
        super().__init__(opt)

        # init the geometry and radiance field
        sdf_func = importlib.import_module("models.SDF")
        color_func = importlib.import_module("models.RadF")
        Renderer = importlib.import_module("models.Renderer")
        self.sdf_func = sdf_func.SDF(opt).to(opt.device)
        self.color_func = color_func.RadF(opt).to(opt.device)
        self.Renderer = Renderer.Renderer(opt)

        self.camera_set = Camera.CameraSet(opt=opt)  # 初始化 相机
        self.point_set = Point3D.Point3DSet(opt=opt)  # 初始化 三维点
        self.cam_info_reloaded = None
        self.pts_info_reloaded = None

    def load_dataset(self, opt, eval_split="val"):
        super().load_dataset(opt)
        # prefetch all training data
        self.train_data.prefetch_all_data(opt)
        self.train_data.all = edict(util.move_to_device(self.train_data.all, opt.device))
        n_views_geo_path = f"data/{opt.data.dataset}/{opt.data.scene}/n_views.npy"
        self.n_views_geo = np.load(n_views_geo_path, allow_pickle=True)

    def save_checkpoint(self, opt, ep=0, it=0, latest=True):
        util.save_checkpoint_sfm(opt, self, ep=ep, it=it, latest=latest)
        if not latest:
            log.info("checkpoint saved: ({0}) {1}, epoch {2} (iteration {3})".format(opt.group, opt.name, ep, it))

    def restore_checkpoint(self, opt):
        epoch_start, iter_start = None, None
        if opt.resume:
            log.info("resuming from previous checkpoint...")
            epoch_start, iter_start = util.restore_checkpoint_sfm(opt, self, resume=opt.resume)
        elif opt.load is not None:
            log.info("loading weights from checkpoint {}...".format(opt.load))
            epoch_start, iter_start = util.restore_checkpoint_sfm(opt, self, load_name=opt.load)
        else:
            log.info("initializing weights from scratch...")
        self.epoch_start = epoch_start or 0
        self.iter_start = iter_start or 0

    def load_matches(self, opt):
        var = edict({"iter": 0})
        # get the keypoints
        var.kypts = []
        var.matches = []
        var.masks = []
        var.poses_gt = self.train_data.all.pose
        for tem_dict in self.n_views_geo:
            dsamp_xy = np.array([self.train_data.factor_x,
                                 self.train_data.factor_y])
            dsamp_xy = dsamp_xy.reshape(-1, 2)
            var.kypts.append(torch.from_numpy(tem_dict["kypts"] / dsamp_xy).to(opt.device).float())
            var.matches.append(tem_dict["indxes"])
            var.masks.append(tem_dict["mask"])
        return var

    def train(self, opt):
        self.timer = edict(start=time.time(), it_mean=None)
        self.ep = 0  # dummy for timer
        # loading the matches
        var = self.load_matches(opt)

        # pose graph keep the same with colmap
        pose_graph_path = f"data/{opt.data.dataset}/{opt.data.scene}/pose_graph.npy"

        if os.path.exists(pose_graph_path) == False:
            pose_graph = [i for i in range(len(self.train_data.all.image)) if i % 3 == 0]
        else:
            pose_graph = np.load(pose_graph_path, allow_pickle=True)[:]
            full_graph = [i for i in range(len(self.train_data.all.image))]
            if len(pose_graph) <= len(self.train_data.all.image) / 2:
                # 认为colmap基本上 fail了
                print("------supplement the pose graph------------")
                pose_graph = pose_graph + [j for j in full_graph if j not in pose_graph]

        # training
        loader = tqdm.trange(20000000, desc="Incremental GeoSDF", leave=False)
        pose_graph_i = 1
        # 记录pnp数量
        nbv_if = False

        # ------------reloading the point set--------------------
        if opt.resume == True:
            for xyz_i, feat_track_i in zip(self.pts_info_reloaded["xyzs"], self.pts_info_reloaded["feat_tracks"]):
                self.point_set.add_point3d(xyz=xyz_i,
                                           feat_tracks=feat_track_i)
        load_finish = False

        random_view = None

        rendered_rgb_list = []
        for self.it in loader:
            var.iter = self.it
            if random_view is not None:
                rendered_output = self.camera_set.cameras[0].render_img_by_slices(
                    self.sdf_func,
                    self.color_func,
                    self.Renderer,
                    random_view
                )
                rendered_rgb = rendered_output['rgb'][0].reshape(*opt.data.image_size,3).cpu().numpy()

                rendered_rgb_list.append(rendered_rgb)
                # import matplotlib.pyplot as plt 
                # for rgb_ in rendered_rgb_list:
                #     plt.imshow(rgb_)
                #     plt.show()
                # import pdb; pdb.set_trace()
            if len(self.camera_set) < 2:
                # Initialization
                if opt.resume == False:
                    var.indx_init = [pose_graph[0], pose_graph[1]]
                else:
                    var.indx_init = self.cam_info_reloaded["cam_id"][:2]
                var.imgs_init = self.train_data.all.image[var.indx_init]
                var.kypts_init = [var.kypts[i] for i in var.indx_init]
                var.intrs_init = self.train_data.all.intr[var.indx_init]
                var.mchs_init = [var.matches[i] for i in var.indx_init]
                var.inliers_init = [var.masks[i] for i in var.indx_init]
                var.gt_depths = None

                if opt.Ablate_config.tri_trad == True:
                    Initializer = Initialization_Trad.Initializer(opt, self.camera_set, self.point_set,
                                                                  self.sdf_func, self.color_func, var,
                                                                  cam_info_reloaded=self.cam_info_reloaded)
                else:
                    Initializer = Initialization.Initializer(opt, self.camera_set, self.point_set,
                                                             self.sdf_func, self.color_func, var,
                                                             cam_info_reloaded=self.cam_info_reloaded)

                # opt.mesh_dir = "{0}/mesh".format(opt.output_path)
                # extract_mesh(
                #     self.color_func,
                #     filepath=os.path.join(opt.mesh_dir, 'init.ply'),
                #     volume_size=2 if opt.data.dataset in ["DTU", "ETH3D"] else 7,
                #     log=log,
                #     show_progress=True,
                #     extra_info=self.sdf_func,
                #     N=512)
                if opt.resume == False:
                    Initializer.run(self.camera_set, self.point_set,
                                    self.sdf_func, self.color_func, Renderer=self.Renderer)
                    self.save_checkpoint(opt, ep=None, it=self.it + 1, latest=False)
                    self.vis_geo_rgb(opt, cameraset=self.camera_set, new_camera=self.camera_set.cameras[0],
                                     pointset=self.point_set, vis_only=True, cam_only=False)

                random_view = slerp(self.camera_set.cameras[0].get_pose()[0], self.camera_set.cameras[1].get_pose()[0], 0.5)                                
                del Initializer
            else:
                if (opt.resume == True) & (load_finish == False):
                    print("------reloading cameras-------")
                    for i in tqdm.tqdm(range(len(self.cam_info_reloaded["cam_id"][2:])), desc="reloading cameras"):
                        new_id = self.cam_info_reloaded["cam_id"][2:][i]
                        img_new = self.train_data.all.image[new_id:new_id + 1]
                        idx_reload = self.cam_info_reloaded["cam_id"].index(new_id)
                        camera_new = Camera.Camera(opt=opt,
                                                   id=new_id,
                                                   img_gt=img_new,
                                                   pose_gt=var.poses_gt[new_id:new_id + 1],
                                                   kypts2D=var.kypts[new_id],
                                                   Match_mask=var.matches[new_id],
                                                   Inlier_mask=var.masks[new_id],
                                                   Intrinsic=self.train_data.all.intr[new_id],
                                                   Depth_omn=None,
                                                   Normal_omn=None,
                                                   Extrinsic=self.cam_info_reloaded["pose_para"][
                                                             idx_reload:idx_reload + 1],
                                                   idx2d_to_3d=self.cam_info_reloaded["idx2d_to_3ds"][idx_reload])
                        self.camera_set.add_camera(id=camera_new.id, CameraNew=camera_new)
                    print("reloading finished")
                    load_finish = True
                    if opt.get_result == True:
                        if getattr(opt, "refine_again", None) == True:
                            opt.optim.refine.max_iter = 10000
                            Refiner = rendering_refine.Refine(opt=opt,
                                                              cameraset=self.camera_set,
                                                              pointset=self.point_set,
                                                              sdf_func=self.sdf_func,
                                                              color_func=self.color_func)
                            Refiner.run(sdf_func=self.sdf_func, color_func=self.color_func)
                            del Refiner
                            torch.cuda.empty_cache()
                            self.save_checkpoint(opt, ep=None, it=self.it + 1, latest=True)
                        if getattr(opt, "vis_all_rgb", None) == True:
                            self.vis_all_rgb(opt, self.camera_set)
                        from utils import plots as plt
                        mesh_dir = "{0}/mesh".format(opt.output_path)
                        os.makedirs("{0}/mesh".format(opt.output_path), exist_ok=True)
                        mesh_dir = f"{mesh_dir}/high_res.ply"
                        self.vis_geo_rgb(opt, cameraset=self.camera_set, new_camera=self.camera_set.cameras[0],
                                         pointset=self.point_set, vis_only=False, cam_only=False)
                        # pdb.set_trace()
                        plt.get_surface_high_res_mesh(self.sdf_func.infer_sdf, resolution=512,
                                                      grid_boundary=[-0.6, 0.6],
                                                      level=0., take_components=True, path=mesh_dir)

                        return
                    else:
                        continue
                # 开始incremental sfm
                pose_graph_left = [p for p in pose_graph if p not in self.camera_set.cam_ids]
                print(f"---------------- {len(pose_graph_left)} frames left ------------------")
                if len(pose_graph_left) == 0:
                    self.vis_geo_rgb(opt, cameraset=self.camera_set, new_camera=self.camera_set.cameras[0],
                                     pointset=self.point_set, cam_only=True)
                    print(f"finish!")

                print("---------- searching next best view -------------")
                if opt.nbv_mode == "colmap":
                    new_id = pose_graph_left[0]
                else:
                    pnp_num_mches = []
                    pnp_inlier_ratio = []
                    pnp_views = []
                    for new_id_i in pose_graph_left:
                        new_id = new_id_i
                        img_new = self.train_data.all.image[new_id:new_id + 1]
                        camera_new = Camera.Camera(opt=opt,
                                                   id=new_id,
                                                   img_gt=img_new,
                                                   pose_gt=var.poses_gt[new_id:new_id + 1],
                                                   kypts2D=var.kypts[new_id],
                                                   Match_mask=var.matches[new_id],
                                                   Inlier_mask=var.masks[new_id],
                                                   Intrinsic=self.train_data.all.intr[new_id],
                                                   Depth_omn=None,
                                                   Normal_omn=None,
                                                   Extrinsic=None,
                                                   idx2d_to_3d=None)
                        Register = Registration.Registration(opt, self.sdf_func, cameraset=self.camera_set)
                        register_sucess, inlier_num_ratio, inlier_num = Register.PnP(camera_new=camera_new,
                                                                                     pointset=self.point_set,
                                                                                     if_nbv=nbv_if)
                        pnp_num_mches.append(inlier_num)
                        pnp_inlier_ratio.append(inlier_num_ratio)
                        pnp_views.append(len(Register.src_cam_id))
                        print(len(Register.src_cam_id))
                    print("---------- max inlier 3d-2d --------------")
                    print(pnp_inlier_ratio)
                    pnp_num_mches = np.array(pnp_num_mches)
                    pnp_inlier_ratio = np.array(pnp_inlier_ratio)
                    pnp_views = np.array(pnp_views)
                    pnp_score = pnp_inlier_ratio * np.clip(pnp_views, 0, 10) + pnp_num_mches / np.max(pnp_num_mches)
                    nbv = np.argmax(pnp_score)
                    print("--------- max number is {} --------------".format(pnp_num_mches[nbv]))
                    new_id = pose_graph_left[nbv]
                print(f"-------------the best view next id is {new_id}--------------")
                img_new = self.train_data.all.image[new_id:new_id + 1]
                camera_new = Camera.Camera(opt=opt,
                                           id=new_id,
                                           img_gt=img_new,
                                           pose_gt=var.poses_gt[new_id:new_id + 1],
                                           kypts2D=var.kypts[new_id],
                                           Match_mask=var.matches[new_id],
                                           Inlier_mask=var.masks[new_id],
                                           Intrinsic=self.train_data.all.intr[new_id],
                                           Depth_omn=None,
                                           Normal_omn=None,
                                           Extrinsic=None,
                                           idx2d_to_3d=None)
                print(f"Total cameras num:{len(self.camera_set.cam_ids)}")
                print(f"new cam_id: {new_id}")
                # -------------------Registration: PnP+Triangulation----------------------------------------------
                if opt.Ablate_config.tri_trad == True:
                    Register = Registration_Trad.Registration(opt, self.sdf_func, cameraset=self.camera_set)
                else:
                    Register = Registration.Registration(opt, self.sdf_func, cameraset=self.camera_set)
                register_sucess, inlier_num_ratio, inlier_num = Register.PnP(camera_new=camera_new,
                                                                             pointset=self.point_set, if_nbv=True)
                rot_error, t_error = self.camera_set.eval_poses()
                if register_sucess == False:
                    print("reconstruct fail")
                    return
                else:
                    self.camera_set.add_camera(id=camera_new.id, CameraNew=camera_new)
                rot_error, t_error = Register.eval_local_pose(camera_new)
                src_cam_id = Register.geo_init_nf(camera_new=camera_new, sdf_func=self.sdf_func,
                                                  pointset=self.point_set)
                del Register
                torch.cuda.empty_cache()
                if opt.Ablate_config.ba_trad == True:
                    # local ba
                    mode = "sfm"
                    Local_BA_id = [camera_new.id] + src_cam_id
                    Bundler = BA_Trad.BA(opt=opt,
                                         cameraset=self.camera_set,
                                         pointset=self.point_set,
                                         sdf_func=self.sdf_func,
                                         color_func=self.color_func,
                                         cam_pick_ids=Local_BA_id,
                                         mode=mode)
                    reproj_tem = Bundler.run_ba(sdf_func=self.sdf_func,
                                                color_func=self.color_func,
                                                Renderer=self.Renderer)
                    rot_error, t_error = self.camera_set.eval_poses(pick_cam_id=src_cam_id + [camera_new.id])
                    del Bundler
                    torch.cuda.empty_cache()
                    # global ba
                    Bundler = BA_Trad.BA(opt=opt,
                                         cameraset=self.camera_set,
                                         pointset=self.point_set,
                                         sdf_func=self.sdf_func,
                                         color_func=self.color_func,
                                         mode=mode
                                         )
                    reproj_tem = Bundler.run_ba(sdf_func=self.sdf_func,
                                                color_func=self.color_func,
                                                Renderer=self.Renderer)
                    rot_error, t_error = self.camera_set.eval_poses()
                    del Bundler
                    torch.cuda.empty_cache()
                else:
                    # ------------------Registration fusion-------------------------------------------------------------------
                    if opt.sfm_mode == "full":
                        reproj_tem = 100
                        iter_cycle = 0
                        mode = "sfm_refine"
                        print("-------------- reproj+rendering registration refine --------------------")
                        while (reproj_tem > 2.5):
                            if iter_cycle >= 1:
                                break
                            # -----------------------------------
                            Local_BA_id = [new_id]
                            Bundler = BA.BA(opt=opt,
                                            cameraset=self.camera_set,
                                            pointset=self.point_set,
                                            sdf_func=self.sdf_func,
                                            color_func=self.color_func,
                                            cam_pick_ids=Local_BA_id,
                                            mode=mode)
                            reproj_tem = Bundler.run_ba(sdf_func=self.sdf_func,
                                                        color_func=self.color_func,
                                                        Renderer=self.Renderer)
                            rot_error, t_error = self.camera_set.eval_poses(pick_cam_id=src_cam_id + [camera_new.id])
                            del Bundler
                            torch.cuda.empty_cache()
                            iter_cycle += 1
                    # ------------------local ba---------------------------------------------------------------------
                    reproj_tem = 100
                    iter_cycle = 0
                    mode = "sfm"
                    while (reproj_tem > 1.):
                        if iter_cycle >= 5:
                            break
                        # -----------------------------------
                        Local_BA_id = [camera_new.id] + src_cam_id
                        Bundler = BA.BA(opt=opt,
                                        cameraset=self.camera_set,
                                        pointset=self.point_set,
                                        sdf_func=self.sdf_func,
                                        color_func=self.color_func,
                                        cam_pick_ids=Local_BA_id,
                                        mode=mode)
                        reproj_tem = Bundler.run_ba(sdf_func=self.sdf_func,
                                                    color_func=self.color_func,
                                                    Renderer=self.Renderer)
                        print(f"local frames num:{len(src_cam_id + [camera_new.id])}")
                        rot_error, t_error = self.camera_set.eval_poses(pick_cam_id=src_cam_id + [camera_new.id])
                        del Bundler
                        torch.cuda.empty_cache()
                        iter_cycle += 1
                    # ------------------global ba---------------------------------------------------------------------
                    reproj_tem = 100
                    iter_cycle = 0
                    mode = "sfm"
                    while (reproj_tem > 1.):
                        if iter_cycle >= 5:
                            break
                        # -----------------global ba--------------------------------------------------------------------
                        Bundler = BA.BA(opt=opt,
                                        cameraset=self.camera_set,
                                        pointset=self.point_set,
                                        sdf_func=self.sdf_func,
                                        color_func=self.color_func,
                                        mode=mode
                                        )
                        reproj_tem = Bundler.run_ba(sdf_func=self.sdf_func,
                                                    color_func=self.color_func,
                                                    Renderer=self.Renderer)
                        rot_error, t_error = self.camera_set.eval_poses()
                        del Bundler
                        torch.cuda.empty_cache()
                        iter_cycle += 1
                    # ------------------ rendering refine -------------------------------------------------------------
                    if opt.sfm_mode == "full":
                        Refiner = rendering_refine.Refine(opt=opt,
                                                          cameraset=self.camera_set,
                                                          pointset=self.point_set,
                                                          sdf_func=self.sdf_func,
                                                          color_func=self.color_func)
                        Refiner.run(sdf_func=self.sdf_func,
                                    color_func=self.color_func,
                                    Renderer=self.Renderer
                                    )
                        del Refiner
                        torch.cuda.empty_cache()
                # # recording
                # self.tb.add_scalar("{0}/loss_{1}".format("Train", "rot_error"), rot_error, len(self.camera_set.cameras))
                # self.tb.add_scalar("{0}/loss_{1}".format("Train", "ATE"), t_error, len(self.camera_set.cameras))
                # self.tb.add_scalar("{0}/loss_{1}".format("Train", "reproj"), reproj_tem.item(),
                #                    len(self.camera_set.cameras))
                if self.it % opt.freq.ckpt == 0:
                    save_latest = False
                else:
                    save_latest = True
                self.vis_geo_rgb(opt, cameraset=self.camera_set, new_camera=camera_new, pointset=self.point_set,
                                 vis_only=False, cam_only=True)
                if (len(self.camera_set.cameras) % opt.freq.ckpt == 0) & (opt.sfm_mode == "full"):
                    self.vis_geo_rgb(opt, cameraset=self.camera_set, new_camera=camera_new, pointset=self.point_set,
                                     vis_only=False, cam_only=False)
                self.save_checkpoint(opt, ep=None, it=self.it + 1, latest=save_latest)

            # if self.it % opt.freq.ckpt == 0: self.save_checkpoint(opt, ep=None, it=self.it + 1)
            pose_graph_i += 1
            torch.cuda.empty_cache()
            # after training
            # if opt.tb:
            #     self.tb.flush()
            #     self.tb.close()
            if opt.visdom: self.vis.close()
