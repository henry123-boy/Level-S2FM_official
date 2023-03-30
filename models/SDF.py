import pdb
import numpy as np
import torch.nn as nn
import torch
import utils.util as util
import utils.camera as camera
from .base import get_Embedder, Geometry

epsilon = 10e-16
from utils.custom_functions import \
    RayAABBIntersector


class SDF(nn.Module):
    def __init__(self, opt):
        super(SDF, self).__init__()
        self.opt = opt

        # hash bound config
        self.bound_max = torch.from_numpy(np.array(opt.data.bound_max)).to(  # [1,1,3]  the boundary
            opt.device)[None, None, :].float()
        self.bound_min = torch.from_numpy(np.array(opt.data.bound_min)).to(  # [1,1,3]
            opt.device)[None, None, :].float()
        self.center = (self.bound_max + self.bound_min) / 2
        self.half_size = (self.bound_max - self.bound_min) / 2

        # volsdf config
        self.rescale = opt.SDF.VolSDF.rescale
        self.beta_speed = opt.SDF.VolSDF.beta_speed
        beta_init = np.log(opt.SDF.VolSDF.beta_init) / self.beta_speed
        beta_optim = torch.from_numpy(np.array([beta_init], dtype=np.float32)).to(opt.device)
        self.beta = nn.Parameter(data=beta_optim, requires_grad=True)

        # config for st (sphere tracing)
        self.sdf_threshold = float(opt.SDF.VolSDF.sdf_threshold)  # sdf threshold for convergence of st
        self.iters_max = int(opt.SDF.VolSDF.iters_max_st)  # iter number for st
        self.scale_mlp = opt.SDF.NN_Init.scale_mlp

        # define_network
        self.define_network(opt=opt)

    def define_network(self, opt):

        # acquire the embedder
        self.embed_fn = get_Embedder(opt=opt, input_dim=3)
        input_3D_dim = self.embed_fn.out_dim

        # acquire the geometry encoder
        self.SDF_MLP = Geometry(opt=opt,
                                input_dim=input_3D_dim,
                                skip=opt.SDF.arch.skip,
                                tf_init=opt.SDF.NN_Init.tf_init,
                                layers=util.get_layer_dims(opt.SDF.arch.layers))

    def infer_sdf(self, xyz, mode="ret_sdf"):

        # get the enc from the embed fn
        enc = self.embed_fn(xyz,
                            rescale=self.rescale,
                            bound_min=self.bound_min,
                            bound_max=self.bound_max
                            )

        feat = self.SDF_MLP(enc)

        if self.opt.data.inside == True:
            sdf = feat[..., :1] / self.scale_mlp
            if self.opt.data.bg_sdf == True:  # only inside we need to do the bg sdf setting
                sdf = torch.min(sdf, self.opt.data.bg_rad - xyz.norm(dim=-1, keepdim=True))
        else:
            sdf = -feat[..., :1] / self.scale_mlp

        if mode == "ret_sdf":
            return sdf
        elif mode == "ret_feat":
            return feat
        elif mode == "ret_all":
            return sdf, feat

    def forward_ab(self):
        beta = torch.exp(self.beta * self.beta_speed)
        return 1. / beta, beta

    def sdf_to_sigma(self, sdf, alpha, beta):
        exp = 0.5 * torch.exp(-torch.abs(sdf) / beta)
        psi = torch.where(sdf >= 0, exp, 1 - exp)
        return alpha * psi

    def density(self, opt, xyzs):
        sdfs = self.infer_sdf(opt, xyzs, mode="ret_sdf")
        alpha_render, beta_render = self.forward_ab()
        density_samples = self.sdf_to_sigma(sdfs, alpha_render, beta_render)
        return density_samples

    def get_surface_pts(self, pts):
        sdf = self.infer_sdf(pts.detach(), mode="ret_sdf")
        normals = self.gradient(pts)
        normals_value = torch.norm(normals, dim=-1, keepdim=True)
        surf_pts = pts - normals / normals_value.detach() * sdf
        return surf_pts, normals_value

    def gradient(self, p):
        with torch.enable_grad():
            p.requires_grad_(True)
            y = self.infer_sdf(p, mode="ret_sdf")
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=p,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True, allow_unused=True)[0]
            return gradients

    def sphere_tracing(self, ray0, ray_direction, model, c=None,
                       tau=0.5, n_steps=[128, 129], n_secant_steps=8,
                       depth_range=[0., 2.4], max_points=3500000, rad=1.0, iter=0):

        _, hits_t, _ = RayAABBIntersector.apply(ray0.view(-1, 3), ray_direction.view(-1, 3), self.center.squeeze(0),
                                                self.half_size.squeeze(0), 1)
        depth_intersect = hits_t.squeeze(1).view([*ray_direction.shape[:2], -1])
        sphere_intersections_points = ray0[:, :, None, :] + depth_intersect.unsqueeze(-1) * ray_direction[:, :, None, :]
        # Initialize start current points
        curr_start_points = torch.zeros_like(ray0).view(-1,3).cuda().float()
        curr_start_points = sphere_intersections_points[:, :, 0, :].reshape(-1, 3)
        acc_start_dis = torch.zeros_like(depth_intersect[...,0]).squeeze().cuda().float()
        acc_start_dis = depth_intersect.reshape(-1, 2)[..., 0]

        # Initialize end current points
        curr_end_points = torch.zeros_like(ray0).view(-1,3).cuda().float()
        curr_end_points = sphere_intersections_points[:, :, 1, :].reshape(-1, 3)
        acc_end_dis = torch.zeros_like(depth_intersect[...,0]).squeeze().cuda().float()
        acc_end_dis = depth_intersect.reshape(-1, 2)[..., 1]

        # Initizliae min and max depth
        min_dis = acc_start_dis.clone()
        max_dis = acc_end_dis.clone()

        # Iterate on the rays (from both sides) till finding a surface
        iters = 0
        next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
        next_sdf_start = self.infer_sdf(curr_start_points, mode="ret_sdf")
        next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
        next_sdf_end = self.infer_sdf(curr_end_points, mode="ret_sdf")
        unfinished_mask_start = None
        unfinished_mask_end = None
        pts_track = []
        while True:
            with torch.no_grad():
                # Update sdf
                curr_sdf_start = torch.zeros_like(acc_start_dis).cuda()
                curr_sdf_start = next_sdf_start
                curr_sdf_start[abs(curr_sdf_start) <= self.sdf_threshold] = 0
                curr_sdf_end = torch.zeros_like(acc_end_dis).cuda()
                curr_sdf_end = next_sdf_end
                curr_sdf_end[abs(curr_sdf_end) <= self.sdf_threshold] = 0

                # Update masks
                if unfinished_mask_start is None:
                    unfinished_mask_start = (abs(curr_sdf_start) > self.sdf_threshold).view(-1, 1)
                    unfinished_mask_end = (abs(curr_sdf_end) > self.sdf_threshold).view(-1, 1)
                else:
                    unfinished_mask_start = unfinished_mask_start & (abs(curr_sdf_start) > self.sdf_threshold).view(-1,
                                                                                                                    1)
                    unfinished_mask_end = unfinished_mask_end & (abs(curr_sdf_end) > self.sdf_threshold).view(-1, 1)
                if (unfinished_mask_start.sum() == 0) or iters == self.iters_max:
                    break
                iters += 1

                # Make step
                # Update distance
                acc_start_dis = acc_start_dis[:, None] + 1. * curr_sdf_start
                acc_end_dis = acc_end_dis[:, None] + 1. * curr_sdf_end
                acc_start_dis = torch.where(acc_start_dis > max_dis.view(*acc_start_dis.shape),
                                            max_dis.view(*acc_start_dis.shape), acc_start_dis)
                acc_end_dis = torch.where(acc_end_dis > max_dis.view(*acc_end_dis.shape),
                                          max_dis.view(*acc_end_dis.shape), acc_end_dis)
                # Update points
                pts_track.append(curr_start_points.detach().unsqueeze(-2))
                curr_start_points = (ray0 + acc_start_dis.view([*ray_direction.shape[:-1], 1]) * ray_direction).view(-1,
                                                                                                                     3)
                curr_end_points = (ray0 + acc_end_dis.view([*ray_direction.shape[:-1], 1]) * ray_direction).view(-1, 3)
                # Fix points which wrongly crossed the surface
                if unfinished_mask_start.sum() > 0:
                    next_sdf_start[unfinished_mask_start] = \
                        self.infer_sdf(curr_start_points[unfinished_mask_start.squeeze()],
                                       mode="ret_sdf").squeeze()
                    next_sdf_start = next_sdf_start.view(-1, 1)



                if unfinished_mask_end.sum() > 0:
                    next_sdf_end[unfinished_mask_end] = \
                        self.infer_sdf(curr_end_points[unfinished_mask_end.squeeze()], mode="ret_sdf").squeeze()
                    next_sdf_end = next_sdf_end.view(-1, 1)
                acc_start_dis = acc_start_dis.squeeze(-1)
                acc_end_dis = acc_end_dis.squeeze(-1)
                unfinished_mask_start = unfinished_mask_start & (acc_start_dis < acc_end_dis).view(-1, 1)
                unfinished_mask_end = unfinished_mask_end & (acc_start_dis < acc_end_dis).view(-1, 1)
        if len(pts_track) == 0:
            pts_track = [curr_start_points.detach().unsqueeze(-2)]
        pts_tracks = torch.cat(pts_track, dim=-2)


        sdf_tracks = self.infer_sdf(pts_tracks, mode="ret_sdf")

        sdf_sum = torch.sum(sdf_tracks, dim=-2)

        d_pred = sdf_sum.view(*ray0.shape[:-1]) + min_dis.view(*ray0.shape[:-1])
        d_pred = torch.where(d_pred > max_dis.view(*d_pred.shape), max_dis.view(*d_pred.shape), d_pred)
        MAX_SAMPLES = self.opt.Res
        finish_mask = (sdf_tracks[:, -1, :].abs() < (
                self.bound_max.squeeze()[0] - self.bound_min.squeeze()[0]) / 10 / MAX_SAMPLES)

        factor_rand = torch.rand_like(d_pred)
        d_up = torch.where(1.5 * acc_end_dis.view(*d_pred.shape) > max_dis.view(*d_pred.shape),
                           max_dis.view(*d_pred.shape), 1.5 * acc_end_dis.view(*d_pred.shape))
        d_sample = (1 - factor_rand) * d_up + factor_rand * min_dis.view(*d_pred.shape)
        # mask_inside=d_sample<self.opt.RenMchFusion.depth.max_d
        sampled_pts = ray0 + d_sample[..., None] * ray_direction

        pick_ind = torch.randperm(pts_tracks.shape[0])[:4096]
        sampled_pts = torch.cat([pts_tracks[pick_ind].view(1, -1, 3), sampled_pts.view(1, -1, 3)], dim=1)

        return d_pred, sdf_tracks[:, -1, 0], sampled_pts, finish_mask
