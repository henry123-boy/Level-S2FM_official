import cv2
import numpy as np
import os, sys, time
import torch
import torch.nn.functional as torch_F
import tqdm
from utils.camera import img2cam, to_hom, cam2world, world2cam, cam2img
from easydict import EasyDict as edict
from utils import camera
from . import Camera
from . import Point3D
from typing import Optional
import pycolmap

# from models.cnn_model.encoder import SpatialEncoder
epsilon = 1e-6


class Registration():
    def __init__(self,
                 opt,
                 sdf_func,
                 cameraset: Camera.CameraSet ):
        super(Registration, self).__init__()
        self.opt=opt
        self.src_cam_id=[]
        # -----------set optimizer for geo init-------------
        self.max_iter = 200
        lr_sdf=opt.optim.geoinit.lr_sdf
        lr_sdf_end = opt.optim.geoinit.lr_sdf_end
        optim = getattr(torch.optim,opt.optim.algo)
        self.optim = optim([{'params': sdf_func.parameters(), 'lr': lr_sdf}])
        sched= getattr(torch.optim.lr_scheduler, getattr(opt.optim.sched, "type"))
        self.sched=sched(self.optim, gamma=(lr_sdf_end / lr_sdf) ** (1. / self.max_iter))
        self.sdf_func=sdf_func
        self.CameraSet=cameraset
    def get_pairs(self,
                  new_cam: Optional[Camera.Camera],
                  cameraset: Optional[Camera.CameraSet],
                  pointset: Optional[Point3D.Point3DSet]):
        """
        :param new_cam:
        :param cameraset:
        :param pointset:
        :return:
        """
        pts_3d=[]
        pts_2d=[]
        id_3d=[]
        id_2d=[]
        for cam_i in cameraset.cameras:
            id_i=cam_i.id
            rel_ind=new_cam.id if id_i>new_cam.id else new_cam.id-1
            if cam_i.inlier_msks[rel_ind].sum()<1:
                continue
            pts3d_idx=cam_i.idx2d_to_3d[cam_i.mch_msks[rel_ind][cam_i.inlier_msks[rel_ind]][...,0]]
            pts2d_idx=cam_i.mch_msks[rel_ind][cam_i.inlier_msks[rel_ind]][...,1]
            mask=pts3d_idx!=-1
            if mask.sum()==0:
                continue
            self.src_cam_id.append(id_i)
            pts_3d.append(torch.cat(pointset.get_xyzs(pts3d_idx[mask].tolist()),dim=0))
            pts_2d.append(new_cam.kypts[pts2d_idx[mask].astype(int)])
            id_3d.append(pts3d_idx[mask])
            id_2d.append(pts2d_idx[mask].astype(int))
        if len(id_2d)==0:
            return False,False,False,False
        id_2d_unique,org_ind2d=np.unique(np.concatenate(id_2d,axis=0),axis=0, return_index=True)
        id_3d_unique,org_ind3d=np.concatenate(id_3d,axis=0)[org_ind2d],org_ind2d
        pts_3d_unique=torch.cat(pts_3d,dim=0)[org_ind3d]
        pts_2d_unique = torch.cat(pts_2d, dim=0)[org_ind3d]
        return pts_3d_unique,pts_2d_unique,id_3d_unique,id_2d_unique
    def PnP(self,
            camera_new,
            pointset,
            if_nbv=False):
        p3d,p2d,id_3d,id_2d=self.get_pairs(new_cam=camera_new,
                                           cameraset=self.CameraSet,
                                           pointset=pointset)         # 得到3d to 2d的匹配点 以及 对应三维点的id号
        if p3d is False:
            return False,0,0
        if (len(p3d)<100)&(if_nbv==False):
            return False,0,len(p3d)
        print("pnp found {} 3D-2D pairs".format(len(id_3d)))
        sdfs=self.sdf_func.infer_sdf(p3d).detach().cpu()
        if self.opt.ba_trad==False:
            mask=(sdfs<0.05).squeeze()
            p3d=p3d.detach().cpu().numpy()[mask]
            p2d=p2d.detach().cpu().numpy()[mask]
        else:
            mask = (sdfs < 5000).squeeze()
            p3d = p3d.detach().cpu().numpy()[mask]
            p2d = p2d.detach().cpu().numpy()[mask]

        print("pnp found {} masked 3D-2D pairs".format(len(p3d)))
        # ---------- pnp --------------
        f, cx, cy =camera_new.f,camera_new.cx,camera_new.cy
        camera_colmap = pycolmap.Camera(model='SIMPLE_PINHOLE', width=self.opt.data.image_size[1],
                                        height=self.opt.data.image_size[0], params=[f, cx, cy], )
        answer = pycolmap.absolute_pose_estimation(p2d, p3d, camera_colmap,max_error_px=3.0)
        if answer["success"] == False:
            print("registration fail# image{}".format(camera_new.id))
            return False,0,0
        id_2d=id_2d[mask][answer['inliers']]
        id_3d=id_3d[mask][answer['inliers']]
        if (len(id_2d)<100)&(if_nbv==False):
            return False,len(id_2d)/len(p3d),len(id_2d)
        print("pnp solve {} inlier 3D-2D pairs".format(len(id_2d)))
        answer = pycolmap.pose_refinement(answer['tvec'], answer['qvec'], p2d, p3d,answer['inliers'], camera_colmap)
        # ----------update pose--------
        rot_absolute = pycolmap.qvec_to_rotmat(answer["qvec"])
        SE3 = np.concatenate([rot_absolute, answer['tvec'].reshape(3, 1)], axis=-1)
        camera_new.se3_refine.data = camera.lie.SE3_to_se3(torch.from_numpy(SE3).to(self.opt.device).unsqueeze(0)).float()
        # ----------update feature tracks---------
        feat_tracks_new=[(len(self.CameraSet), id_kyt_i) for id_kyt_i in id_2d]
        pointset.update_feat_tracks(id_3d,feat_tracks_new)
        camera_new.idx2d_to_3d[id_2d]=id_3d
        #-----------vis the pnp----------------------------
        kypts_vis = camera_new.kypts[id_2d]
        img_gt_vis = camera_new.img_gt[0].detach().permute(1, 2, 0).cpu().numpy() * 255
        img_gt_vis = cv2.cvtColor(img_gt_vis, cv2.COLOR_RGB2BGR)
        for ky0 in kypts_vis.detach().cpu().numpy().astype('int'):
            cv2.circle(img_gt_vis, ky0, radius=2, color=(0, 0, 255), thickness=2)
        os.makedirs("{0}/pnp".format(self.opt.output_path), exist_ok=True)
        cv2.imwrite("{0}/pnp".format(self.opt.output_path) + "/pnp_{}.jpg".format(len(self.CameraSet.cameras)), img_gt_vis)
        return True,len(id_2d)/len(p3d),len(id_2d)
    def eval_local_pose(self,
                        camera_new):
        rot_error,t_error=self.CameraSet.eval_poses(pick_cam_id=self.src_cam_id+[camera_new.id])
        return rot_error,t_error
    def geo_init_nf(self,
                    camera_new: Camera.Camera,
                    sdf_func,
                    pointset:Point3D.Point3DSet,
                    depth_omnidata: Optional[torch.Tensor]=None,
                    normal_omnidata: Optional[torch.Tensor]=None,
                    reproj_max: Optional[torch.Tensor]=15):
        loader = tqdm.trange(self.max_iter,desc="New Triangulation",leave=False)
        # new camera和他对应的以前的view之间的pair
        center_pairs=[]
        ray_pairs=[]
        kypts_pairs=[]
        kypts_idx_pairs=[]
        masks_new=[]
        num_ckpt=[0]
        pose_pairs=[]
        cam_pairs_id=[]
        for src_cam_i in self.src_cam_id:
            """
            mode="only_center_ray"
            返回值为: center, ray, kypts_src,kypts2d_indx_self
            """
            ret = edict()
            cam_i = self.CameraSet(src_cam_i)
            # ref to src
            ret0 = camera_new.proj_cam_i(cam_i=cam_i, sdf_func=sdf_func, mode="kypts",mode_3d="only_center_ray")
            # src to ref
            ret1 = cam_i.proj_cam_i(cam_i=camera_new, sdf_func=sdf_func, mode="kypts",mode_3d="only_center_ray")
            for key in ret0.keys():
                if type(getattr(ret0, key)) is torch.Tensor:
                    if key == "kypts_src":
                        setattr(ret, key,
                                torch.cat([getattr(ret0, key).unsqueeze(0), getattr(ret1, key).unsqueeze(0)], dim=0))
                    else:
                        setattr(ret, key,
                                torch.cat([getattr(ret0, key), getattr(ret1, key)], dim=0))

                elif type(getattr(ret0, key)) is np.ndarray:
                    setattr(ret, key,np.concatenate([getattr(ret0, key).reshape(1, -1), getattr(ret1, key).reshape(1, -1)],axis=0))
            mask_new_pts = (camera_new.idx2d_to_3d[ret.kypts2d_indx[0]] == -1)
            masks_new.append(mask_new_pts)
            center_pairs.append(ret.center.detach_())
            ray_pairs.append(ret.ray.detach_())
            kypts_pairs.append(ret.kypts_src.detach_())
            num_ckpt.append(num_ckpt[-1]+ret.kypts_src.shape[1])
            kypts_idx_pairs.append(ret.kypts2d_indx)
            pose_pairs.append([cam_i.get_pose().detach_(),camera_new.get_pose().detach_()])
            cam_pairs_id.append([self.CameraSet.cam_ids.index(camera_new.id),self.CameraSet.cam_ids.index(cam_i.id)])
        # shpere tracing 得到三维点
        center_tem = torch.cat(center_pairs, dim=1)
        ray_tem = torch.cat(ray_pairs, dim=1)
        d_surface, sdf_surf, sample_pts, mask_finish = sdf_func.sphere_tracing(center_tem, ray_tem, sdf_func)
        pts_surface_chunk=[self.triangulation(p[0],p[1],ky[0],ky[1],camera_new.intrinsic) for p,ky in zip(pose_pairs,kypts_pairs)]
        mask_finish_chunk=[mask_finish.view(2,-1,1)[:,a:b,:] for a,b in zip(num_ckpt[:-1],num_ckpt[1:])]
        pts_new=[]
        mask_finish_new=[]
        idx_2d=[]
        cam_id=[]
        for pts3d_pairs_i,finish_pairs_i,kypts_pairs_i,kypts_idx_pairs_i,mask_new_i,pose_pairs_i,cam_id_pairs_i in zip(pts_surface_chunk,mask_finish_chunk,kypts_pairs,kypts_idx_pairs,masks_new,pose_pairs,cam_pairs_id):
            """
            pts3d_pairs_i [2,N,3] new_camera对应的三维点和camera_i对应的三维点
            finish_pairs_i [2,N,1] 三维点是否收敛于表面
            kypts_pairs_i [2,N,2] 与pts3d_pair_i对应的投影后的监督
            kypts_idx_pairs_i [2,N] 与pts3d_pair_i对应的二维坐标索引
            pose_pairs_i [pts需要投影的pose对应]
            """
            mask_new_pts = mask_new_i
            pts_new.append(pts3d_pairs_i[None, mask_new_pts, :])
            mask_finish_new.append(finish_pairs_i.reshape(2, -1)[:, mask_new_pts])
            idx_2d.append(kypts_idx_pairs_i[:, mask_new_pts])
            cam_id.append(cam_id_pairs_i)
        #-------------update feat tracks-------------------
        with torch.no_grad():
            for pts_surf_i,kypts_ind_i,cam_pair_i in zip(pts_new,idx_2d,cam_id):
                pts_avg=pts_surf_i[0]
                kypts2d_indx=kypts_ind_i
                feat_track=[[(cam_pair_i[i],kypts2d_indx[i,j]) for i in range(2)] for j in range(kypts2d_indx.shape[-1])]
                pts_avg=pts_avg.detach()
                pts_avg=list(torch.split(pts_avg,1,dim=0))
                pts_indx=[]
                for pts_avg_i,feat_track_i in zip(pts_avg,feat_track):
                    pts_indx.append(pointset.add_point3d(pts_avg_i,feat_track_i))
                # -------------update 2d-3d matches---------------------
                self.CameraSet.cameras[cam_pair_i[0]].idx2d_to_3d[kypts2d_indx[0]]=pts_indx
                self.CameraSet.cameras[cam_pair_i[1]].idx2d_to_3d[kypts2d_indx[1]] = pts_indx
        if self.opt.Ablate_config.ba_trad==False:
            loader = tqdm.trange(self.max_iter, desc="Initialization", leave=False)
            #-------------multi-view st consistence------------------
            if pointset is not None:
                for it in loader:
                    import random
                    self.optim.zero_grad()
                    rand_cam_id=random.randint(0,len(self.src_cam_id)-1)
                    rand_cam_id=self.src_cam_id[rand_cam_id]
                    cam_rand=self.CameraSet(rand_cam_id)
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
        return self.src_cam_id
    def summarize_loss(self,opt,loss):
        loss_all = 0.
        assert("all" not in loss)
        # weigh losses
        for key in loss:
            assert(key in opt.loss_weight.geoinit)
            assert(loss[key].shape==())
            if opt.loss_weight.geoinit[key] is not None:
                assert not torch.isinf(loss[key]),"loss {} is Inf".format(key)
                assert not torch.isnan(loss[key]),"loss {} is NaN".format(key)
                loss_all += 10**float(opt.loss_weight.geoinit[key])*loss[key]
        loss.update(all=loss_all)
        return loss
    def triangulation(self,pose0,pose1,kypts0,kypts1,intr):
        pose0=pose0.detach().cpu().squeeze()
        pose1=pose1.detach().cpu().squeeze()
        intr=intr.detach().cpu()
        P0 = np.matmul(intr, pose0)
        P1 = np.matmul(intr, pose1)
        pts3D = cv2.triangulatePoints(P0.numpy(), P1.numpy(), kypts0.cpu().numpy().transpose(), kypts1.cpu().numpy().transpose())
        pts3D_out = pts3D[:3, :] / pts3D[3:4, :]
        tri_pts = torch.from_numpy(pts3D_out.transpose()).to(self.opt.device)
        return tri_pts