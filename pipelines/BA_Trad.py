import copy
import pdb
import numpy as np
import torch.nn as nn
import torch
import tqdm
import utils.util as util
from utils.camera import  world2cam, cam2img
from easydict import EasyDict as edict
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
                 cam_pick_ids: Optional[list]=None,
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
        self.opt=opt
        self.mode=mode
        #----------set optimizer-----------------
        max_iter = opt.optim.ba.max_iter
        if (cam_pick_ids is not None):
            if (len(cam_pick_ids) == 1):
                max_iter=int(max_iter//2)
        self.max_iter=max_iter
        lr_sdf = opt.optim.ba.lr_sdf
        lr_sdf_end = opt.optim.ba.lr_sdf_end
        lr_pose = opt.optim.ba.lr_pose
        optim = getattr(torch.optim, opt.optim.algo)
        # init_tem_para: poses
        num_cam=len(cameraset.cameras) if cam_pick_ids is None else len(cam_pick_ids)
        se3_refine = torch.zeros(num_cam, 6).to(opt.device)
        cams_opt = cameraset.cameras if cam_pick_ids is None else [cameraset(i) for i in cam_pick_ids]
        for i in range(num_cam):
            se3_refine.data[i:i + 1, :] = cams_opt[i].se3_refine.data
        self.r_paras = rot_para(opt, se3_refine[..., :3])
        self.t_paras = trans_para(opt, se3_refine[..., 3:])
        self.xyzs_paras=xyzs_para(opt,torch.cat(pointset.get_all_parameters()["xyzs"],dim=0).to(opt.device))
        cameras_para = [{'params': self.r_paras.parameters(), 'lr': opt.optim.ba.lr_pose_r},
                        {'params': self.t_paras.parameters(), 'lr': opt.optim.ba.lr_pose_t}]
        # init_tem_para: pts3d
        self.optim = optim(cameras_para+[{'params': self.xyzs_paras.parameters(), 'lr': opt.optim.lr_xyzs}])
        sched = getattr(torch.optim.lr_scheduler, getattr(opt.optim.sched, "type"))
        self.sched = sched(self.optim, gamma=(lr_sdf_end / lr_sdf) ** (1. / max_iter))

        #-----------pick the pts ---------------
        if cam_pick_ids is not None:
            self.cam_pick_ids=cam_pick_ids
            pts_pick_ids,pose_idx,kypts2D_forward = util.get_idx3d_camset(cameraset=cameraset,
                                             cam_id=self.cam_pick_ids)
            self.pts_pick_ids=np.concatenate(pts_pick_ids,axis=0)
            self.pose_idx=np.array(pose_idx)
            self.ba_name="local_ba"
        else:
            self.cam_pick_ids=[cam_i for cam_i in cameraset.cam_ids]
            pts_pick_ids, pose_idx, kypts2D_forward = util.get_idx3d_camset(cameraset=cameraset,
                                                       cam_id=self.cam_pick_ids)
            self.pose_idx=np.array(pose_idx)
            self.pts_pick_ids = np.concatenate(pts_pick_ids, axis=0)
            self.ba_name = "global_ba"
        _, _, self.rgbs_gt = cameraset.get_ren_data(sdf_func=sdf_func, color_func=color_func, cam_ids=self.cam_pick_ids,
                                                    dp_req=True)
        self.kypts2D_forward = torch.cat(kypts2D_forward, dim=0)
        self.pointset=pointset
        self.cameraset=cameraset


    def run_ba(self,
               sdf_func,
               color_func,
               Renderer):
        loader = tqdm.trange(self.max_iter,desc=self.ba_name,leave=False)
        for it in loader:
            self.xyzs_all = self.xyzs_paras()
            self.se3_refine = torch.cat([self.r_paras(), self.t_paras()], dim=1)

            self.optim.zero_grad()
            ret=edict()
            # with torch.cuda.amp.autocast(dtype=torch.float16):
            #---------------global reproj error------------------------------
            xyzs_forward=self.xyzs_all[self.pts_pick_ids]
            poses_forward=camera.lie.se3_to_SE3(self.se3_refine[self.pose_idx]).to(self.opt.device)
            # xyzs_forward [n,1,3]   poses_forward [n,3,4]    kypts2D_forward [n,2]
            xyzx_forward=world2cam(xyzs_forward.unsqueeze(1),poses_forward)     # [n,1,3]
            uvs_forward=cam2img(xyzx_forward,self.cameraset.cameras[0].intrinsic.repeat(xyzx_forward.shape[0],1,1))
            uvs_forward=(uvs_forward/(uvs_forward[...,2:]+epsilon))[...,:2].squeeze(1)    # [n,2]
            # # ---------------------
            ret.reproj_loss=0.5*((2*torch.log(1+torch.norm(uvs_forward - self.kypts2D_forward, dim=-1)**2/4)).mean())+0.5*torch.norm(uvs_forward - self.kypts2D_forward, dim=-1).mean()
            # print(mask_surf.sum()/len(mask_surf))
            if torch.isinf(ret.reproj_loss).any()==True:
                pdb.set_trace()
            if torch.isnan(ret.reproj_loss).any()==True:
                pdb.set_trace()
            # ---------------------
            reproj_loss_tem=torch.norm(uvs_forward - self.kypts2D_forward, dim=-1).mean()

            loss = self.compute_loss(ret)
            if (loss.reproj_error.item()>10):
                self.opt.loss_weight.ba.reproj_error = 1
            else:
                self.opt.loss_weight.ba.reproj_error = 0
            loss = self.summarize_loss(self.opt, loss)
            loss.all.backward()
            self.optim.step()
            self.sched.step()
            #----------------print the result--------------------------
            loader.set_postfix(it=it,loss="{:.3f}".format(loss.all))
        self.pointset.update_xyzs(self.pts_pick_ids,xyzs_forward.detach())
        for cam_idx,j in zip(self.cam_pick_ids,range(len(self.cam_pick_ids))):
            self.cameraset(cam_idx).se3_refine.data=self.se3_refine[j:j+1].to(self.opt.device)
        print("reprojection error{}".format(reproj_loss_tem))
        return torch.norm(uvs_forward - self.kypts2D_forward, dim=-1).mean()

    def compute_loss(self,ret):
        loss = edict()
        loss.reproj_error=ret.reproj_loss
        return loss

    def summarize_loss(self,opt,loss):
        loss_all = 0.
        assert("all" not in loss)
        # weigh losses
        for key in loss:
            assert(key in opt.loss_weight.ba)
            assert(loss[key].shape==())
            if opt.loss_weight.ba[key] is not None:
                assert not torch.isinf(loss[key]),"loss {} is Inf".format(key)
                assert not torch.isnan(loss[key]),"loss {} is NaN".format(key)
                loss_all += 10**float(opt.loss_weight.ba[key])*loss[key]
        loss.update(all=loss_all)
        return loss

class rot_para(nn.Module):
    def __init__(self,opt,para_in):
        super(rot_para, self).__init__()
        self.paras=nn.Parameter(copy.deepcopy(para_in).to(opt.device))
    def forward(self):
        return self.paras
class trans_para(nn.Module):
    def __init__(self,opt,para_in):
        super(trans_para, self).__init__()
        self.paras=nn.Parameter(copy.deepcopy(para_in).to(opt.device))
    def forward(self):
        return self.paras
class xyzs_para(nn.Module):
    def __init__(self,opt,para_in):
        super(xyzs_para, self).__init__()
        self.paras=nn.Parameter(copy.deepcopy(para_in).to(opt.device))
    def forward(self):
        return self.paras