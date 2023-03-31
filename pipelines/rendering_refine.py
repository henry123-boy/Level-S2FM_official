import torch.nn as nn
import torch
import torch.nn.functional as torch_F
import tqdm
from easydict import EasyDict as edict
import utils.camera as camera
from . import Camera
from . import Point3D
from typing import Optional
from utils.util_vis import extract_mesh

# from models.cnn_model.encoder import SpatialEncoder
epsilon = 1e-6
MAX_SAMPLES=1024
class Refine(nn.Module):
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
        super(Refine, self).__init__()
        self.opt=opt
        self.mode=mode
        #----------set optimizer-----------------
        max_iter = opt.optim.refine.max_iter
        self.max_iter=max_iter
        lr_sdf = opt.optim.refine.lr_sdf
        lr_sdf_end = opt.optim.refine.lr_sdf_end
        lr_color= opt.optim.refine.lr_color
        lr_color_end = opt.optim.refine.lr_color_end
        optim = getattr(torch.optim, opt.optim.algo)
        # init_tem_para: poses
        num_cam=len(cameraset.cameras) if cam_pick_ids is None else len(cam_pick_ids)
        self.optim = optim([{'params': sdf_func.parameters(), 'lr': lr_sdf},
                            {'params': color_func.parameters(), 'lr': lr_color}])
        sched = getattr(torch.optim.lr_scheduler, getattr(opt.optim.sched, "type"))
        self.sched = sched(self.optim, gamma=(lr_sdf_end / lr_sdf) ** (1. / max_iter))
        #-----------pick the pts ---------------
        if cam_pick_ids is not None:
            self.cam_pick_ids=cam_pick_ids
            self.ba_name="local_ba"
        else:
            self.cam_pick_ids=[cam_i for cam_i in cameraset.cam_ids]
            self.ba_name = "global_ba"
        _, _, self.rgbs_gt =cameraset.get_ren_data(sdf_func=sdf_func,color_func=color_func,
                                                                            cam_ids=self.cam_pick_ids,dp_req=True)
        self.pointset=pointset
        self.cameraset=cameraset

    @torch.no_grad()
    def print_loss(self, loss_dict, PSNR = None):
        if PSNR is not None:
            loss_dict_reduced = {'PSNR': PSNR.item()}
        for key in loss_dict.keys():
            loss_dict_reduced[key] = loss_dict[key].item()

        print(loss_dict_reduced)

    def run(self,
               sdf_func,
               color_func,
               Renderer
            ):
        loader = tqdm.trange(self.max_iter,desc=self.ba_name,leave=False)
        for it in loader:
            self.optim.zero_grad()
            ret=edict()
            # #---------------global rendering---------------------------------
            poses_all, poses_gt_all = self.cameraset.get_all_poses(pick_cam_id=self.cam_pick_ids)
            pose_input = poses_all
            self.cameraset.render(sdf_func=sdf_func,color_func=color_func,
                             ret=ret,cam_ids=self.cam_pick_ids,dp_req=True if self.ba_name=="local_ba" else False,
                             pose_input=pose_input,
                             rgbs_gt=self.rgbs_gt.detach_(),
                             Renderer=Renderer,
                             pointset=self.pointset)
            loss = self.compute_loss(ret)
            if it%10 == 0 or it == len(loader)-1:
                self.print_loss(loss, PSNR=ret.get('PSNR',None))
            loss = self.summarize_loss(self.opt, loss)
            loss.all.backward()
            self.optim.step()
            self.sched.step()
        return

    def compute_loss(self,ret):
        loss = edict()
        loss.eikonal_loss = torch_F.l1_loss(torch.norm(ret.normals, dim=-1),
                                            torch.ones_like(torch.norm(ret.normals, dim=-1)))
        loss.rgb = ret.rgb_loss
        loss.DC_Loss = ret.DC_loss
        loss.sdf_surf = torch_F.l1_loss(ret.sdfs, torch.zeros_like(ret.sdfs))
        loss.tracing_loss=ret.tracing_loss
        return loss

    def summarize_loss(self,opt,loss):
        loss_all = 0.
        assert("all" not in loss)
        # weigh losses
        for key in loss:
            assert(key in opt.loss_weight.refine)
            assert(loss[key].shape==())
            if opt.loss_weight.refine[key] is not None:
                assert not torch.isinf(loss[key]),"loss {} is Inf".format(key)
                assert not torch.isnan(loss[key]),"loss {} is NaN".format(key)
                loss_all += 10**float(opt.loss_weight.refine[key])*loss[key]
        loss.update(all=loss_all)
        return loss