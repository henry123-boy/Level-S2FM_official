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
import random
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
        self.opt=opt
        Depths_omn = getattr(var, "omn_depths", None)
        Normals_omn= getattr(var, "omn_norms", None)
        cam_infos=cam_info_reloaded
        if (self.opt.data.dataset in ["TanksAndTemple","BlendedMVS","scannet","DTU","llff","ETH3D","ETH3D_sp"])&(cam_infos==None):
            # init the two view camera on a sphere
            rad_init = getattr(self.opt.data[f"{self.opt.data.scene}"],
                               "rad_init", self.opt.data.bound_max[0] / 2)
            rad = rad_init

            if self.opt.data.inside==True:
                theta_y = torch.tensor([-np.pi/4])
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
            w2c_rot=torch.inverse(R_x)@torch.inverse(R_y)@torch.inverse(R_z)
            t=w2c_rot@(torch.tensor([-rad*np.cos(theta_y)*np.cos(theta_z),-rad*np.cos(theta_y)*np.sin(theta_z),-rad*np.sin(theta_y)]).float().view(3,1))
            w2c_pose=torch.cat([w2c_rot,t],dim=-1)
            se3=camera.Lie().SE3_to_se3(w2c_pose).float()
            Extrinsic_init_cam0=se3
            # ---------------2 view initialization-----------------
            id0,id1=var.indx_init
            rel_id=id1 if id1<id0 else id1-1
            matches=var.mchs_init[0][rel_id]
            kypts0 = var.kypts_init[0][matches[:,0].astype(int)][var.inliers_init[0][rel_id]].cpu().detach().numpy()
            kypts1 = var.kypts_init[1][matches[:,1].astype(int)][var.inliers_init[0][rel_id]].cpu().detach().numpy()
            intr = var.intrs_init[0].cpu().detach().numpy()
            f, cx, cy = intr[0, 0], intr[0, 2], intr[1, 2]
            camera_colmap = pycolmap.Camera(model='SIMPLE_PINHOLE', width=self.opt.data.image_size[1],height=self.opt.data.image_size[0], params=[f, cx, cy], )
            answer = pycolmap.essential_matrix_estimation(kypts0, kypts1, camera_colmap, camera_colmap)
            two_view_rot = torch.from_numpy(pycolmap.qvec_to_rotmat(answer["qvec"]))
            scale_init=getattr(self.opt.data[f"{self.opt.data.scene}"],"scale_init",1)
            two_view_t=torch.from_numpy(answer["tvec"]).view(3,1)*scale_init
            rel_pose=torch.cat([two_view_rot,two_view_t],dim=-1).float()
            Extrinsic_init_cam1=camera.pose.compose_pair(w2c_pose,rel_pose)
            Extrinsic_init_cam1=camera.Lie().SE3_to_se3(Extrinsic_init_cam1).float()
            Extrinsic_init=[Extrinsic_init_cam0,Extrinsic_init_cam1]
            # --------- traditional triangulation ----------------------
            import cv2 as cv
            pose0=camera.Lie().se3_to_SE3(Extrinsic_init_cam0).float().squeeze()
            pose1=camera.Lie().se3_to_SE3(Extrinsic_init_cam1).float().squeeze()
            P0 = np.matmul(intr, pose0)
            P1 = np.matmul(intr, pose1)
            pts3D = cv2.triangulatePoints(P0.numpy(), P1.numpy(), kypts0.transpose(),kypts1.transpose())
            pts3D_out = pts3D[:3, :] / pts3D[3:4, :]
            self.tri_pts=pts3D_out.transpose()
        else:
            Extrinsic_init =[None,None]
        for i in range(2):
            cameraset.add_camera(id=var.indx_init[i],
                                 img_gt=var.imgs_init[i:i+1],
                                 pose_gt=var.poses_gt[var.indx_init][i:i+1],
                                 kypts2D=var.kypts_init[i],
                                 Match_mask=var.mchs_init[i],
                                 Inlier_mask=var.inliers_init[i],
                                 Intrinsic=var.intrs_init[i],
                                 Depth_omn=Depths_omn[i] if Depths_omn is not None else None,
                                 Normal_omn=Normals_omn[i] if Normals_omn is not None else None,
                                 Extrinsic=Extrinsic_init[i] if cam_infos==None else cam_infos["pose_para"][i:i+1],
                                 idx2d_to_3d=None if cam_infos==None else cam_infos["idx2d_to_3ds"][i]
                                 )
            # ------------set optimizer-------------------------
            max_iter = opt.optim.init.max_iter
            self.max_iter = 200
            lr_sdf = opt.optim.init.lr_sdf
            lr_sdf_end = opt.optim.init.lr_sdf_end
            optim = getattr(torch.optim, opt.optim.algo)
            self.optim = optim([{'params': sdf_func.parameters(), 'lr': lr_sdf}])
            sched = getattr(torch.optim.lr_scheduler, getattr(opt.optim.sched, "type"))
            self.sched = sched(self.optim, gamma=(lr_sdf_end / lr_sdf) ** (1. / max_iter))
    def run(self,
            cameraset: Camera.CameraSet,
            pointset: Point3D.Point3DSet,
            sdf_func,
            color_func,
            Renderer=None,
            ref_indx: Optional[int]=0,
            src_indx: Optional[int]=1):
        camera0=cameraset.cameras[ref_indx]
        camera1=cameraset.cameras[src_indx]
        ret = edict()
        ret0 = camera0.proj_cam_i(camera1, sdf_func, mode="kypts")
        ret1 = camera1.proj_cam_i(camera0, sdf_func, mode="kypts")
        for key in ret0.keys():
            if type(getattr(ret0, key)) is torch.Tensor:
                setattr(ret, key, torch.cat([getattr(ret0, key).squeeze(0), getattr(ret1, key).squeeze(0)], dim=0))
            elif type(getattr(ret0, key)) is np.ndarray:
                setattr(ret, key, np.concatenate([getattr(ret0, key), getattr(ret1, key)], axis=0))
        #------------update feat tracks&Point3D---------------------
        with torch.no_grad():
            save_path = self.opt.output_path + "/init_mch"
            os.makedirs(save_path, exist_ok=True)
            # pycolmap.triangulate_points()
            util_vis.draw_matches(self.opt, camera0.img_gt, camera1.img_gt, ret1.kypts_src, ret0.kypts_src,
                                  store_path=save_path + f"/{camera0.id}_{camera1.id}_org.jpg")

            kypts2d_indx=ret.kypts2d_indx.reshape(2,-1)
            feat_track=[[(i,kypts2d_indx[i,j]) for i in range(2)] for j in range(kypts2d_indx.shape[-1])]
            pts_avg=torch.from_numpy(self.tri_pts).to(self.opt.device)
            pts_avg=list(torch.split(pts_avg,1,dim=0))
            pts_indx=[]
            for pts_avg_i,feat_track_i in zip(pts_avg,feat_track):
                pts_indx.append(pointset.add_point3d(pts_avg_i,feat_track_i))
            # -------------update 2d-3d matches---------------------
            camera0.idx2d_to_3d[kypts2d_indx[0]]=pts_indx
            camera1.idx2d_to_3d[kypts2d_indx[1]] = pts_indx
        if self.opt.ba_trad==False:
            loader = tqdm.trange(self.max_iter, desc="Initialization", leave=False)
            #-------------multi-view st consistence------------------
            if pointset is not None:
                for it in loader:
                    self.optim.zero_grad()
                    rand_cam_id=random.randint(0,len(cameraset.cam_ids)-1)
                    rand_cam_id=cameraset.cam_ids[rand_cam_id]
                    cam_rand=cameraset(rand_cam_id)
                    kypts_indx=np.where(cam_rand.idx2d_to_3d!=-1)
                    pts_surface,mask_finish,sdf_surf,sample_pts=cam_rand.get_pts3D(sdf_func,kypts_indx)
                    id_3d=cam_rand.idx2d_to_3d[kypts_indx]
                    xyzs=torch.cat(pointset.get_xyzs(idxs=id_3d),dim=0)
                    tracing_loss=torch.norm(xyzs - pts_surface[0], dim=-1).mean()
                    grad=sdf_func.gradient(sample_pts).norm(dim=-1)
                    loss_total=tracing_loss+torch_F.l1_loss(sdf_surf,torch.zeros_like(sdf_surf))+torch_F.l1_loss(grad,torch.ones_like(grad))
                    loss_total.backward()
                    self.optim.step()
                    if getattr(ret,"sdfs",None) is None:
                        ret.update(edict(sdfs=sdf_surf))
                print(f"tracing_loss:{tracing_loss}")
        # --------- eval poses estimation----------------------------
        cameraset.eval_poses()

