import pdb

import torch
import torch.nn as nn
from utils.custom_functions import \
    RayAABBIntersector
from utils import camera
import numpy as np
import torch.nn.functional as torch_F


class Renderer(nn.Module):
    def __init__(self, opt):
        # options
        self.opt = opt

        # hash bound config
        self.bound_max = torch.from_numpy(np.array(opt.data.bound_max)).to(  # [1,1,3]  the boundary
            opt.device)[None, None, :].float()
        self.bound_min = torch.from_numpy(np.array(opt.data.bound_min)).to(  # [1,1,3]
            opt.device)[None, None, :].float()
        self.center = (self.bound_max + self.bound_min) / 2
        self.half_size = (self.bound_max - self.bound_min) / 2

        if getattr(self.opt.data[f"{self.opt.data.scene}"],
                   "bgcolor", None) is not None:
            bgcolor = getattr(self.opt.data[f"{self.opt.data.scene}"], "bgcolor")
        else:
            bgcolor = opt.data.bgcolor
        bgcolor = np.array(bgcolor)
        self.bgcolor = torch.from_numpy(bgcolor).to(opt.device).float()

    def composite(self, ray, rgb_samples, density_samples, depth_samples):
        ray_length = ray.norm(dim=-1, keepdim=True)  # [B,HW,1]
        # volume rendering: compute probability (using quadrature)

        depth_intv_samples = depth_samples[..., 1:, 0] - depth_samples[..., :-1, 0]  # [B,HW,N-1]

        dist_samples = depth_intv_samples * ray_length  # [B,HW,N-1]
        sigma_delta = density_samples[..., :-1] * dist_samples  # [B,HW,N-1]
        alpha = 1 - (-sigma_delta).exp_()  # [B,HW,N]
        T = (-torch.cat([torch.zeros_like(sigma_delta[..., :1]), sigma_delta], dim=2).cumsum(dim=2)).exp_()[...,
            :-1]  # [B,HW,N]
        prob = (T * alpha)[..., None]  # [B,HW,N,1]

        # integrate RGB and depth weighted by probability
        rgb = (rgb_samples[..., :-1, :] * prob).sum(dim=2)  # [B,HW,3]

        return rgb, prob

    def forward(self, opt, center, ray,
                SDF_Field, Rad_Field
                ):
        # sample the points through the ray
        depth_samples, _, _ = self.volsdf_sampling(opt, center, ray,
                                                   SDF_Field=SDF_Field)  # [B,HW,N,1]

        depth_samples = depth_samples[:, :, :, None]

        p3d_samp = camera.get_3D_points_from_depth(opt, center, ray, depth_samples,
                                                   multi_samples=True)  # [B,HW,N,3]

        alpha_render, beta_render = SDF_Field.forward_ab()  # get the alpha and beta
        sdfs, feats = SDF_Field.infer_sdf(p3d_samp, mode="ret_all")  # sdfs [...,1] [...,N+1]
        normals = SDF_Field.gradient(p3d_samp)

        # get geo enc + ray enc
        ray_enc = Rad_Field.infer_embed_v(ray_utils=ray[..., None, :].expand_as(p3d_samp))  # [B,HW,1,3]
        if opt.Ablate_config.dual_field == True:
            geo_enc = torch.cat([feats[..., 1:],
                                 Rad_Field.Geometry_feat(p3d_samp)[..., 1:]], dim=-1)
        else:
            geo_enc = feats[..., 1:]

        all_enc = torch.cat([p3d_samp, normals, ray_enc, geo_enc], dim=-1)  # [B,HW,N,C]

        # infer the apperance
        rgbs = Rad_Field.infer_app(all_enc)
        # sdfs -> density
        densitys = SDF_Field.sdf_to_sigma(sdf=sdfs,
                                          alpha=alpha_render,
                                          beta=beta_render)
        # composite the rgb and depth
        rgb, prob = self.composite(ray=ray,
                                   rgb_samples=rgbs,
                                   density_samples=densitys.squeeze(-1),
                                   depth_samples=depth_samples)

        # adding the background
        opacity = prob.sum(dim=2)  # [B,HW,1]

        # expand the dim of bgcolor to rgb
        diff_dims = len(rgb.shape) - len(self.bgcolor.shape)
        if diff_dims >= 0:
            for i in range(diff_dims):
                self.bgcolor = self.bgcolor.unsqueeze(0)
        else:
            for i in range(abs(diff_dims)):
                self.bgcolor = self.bgcolor.squeeze(0)

        rgb = rgb + ((1 - opacity) * self.bgcolor)

        depth_mlp = (depth_samples[..., :-1, :] * prob).sum(dim=2)  # [B,HW,1]
        depth_mlp = depth_mlp + (1 - opacity) * depth_samples[..., -1, :]

        normal_mlp = (normals[..., :-1, :] * prob).sum(dim=2)
        normal_mlp = normal_mlp + (1 - opacity) * normals[..., -1, :]

        # return
        ret = {"rgb": rgb,
               "sdfs_volume": sdfs,
               "normals": normals,
               "depth_mlp": depth_mlp,
               "normal_mlp": normal_mlp}

        return ret

    def sample_depth(self, opt,
                     min_d=None, max_d=None):
        depth_min, depth_max = min_d[..., None, :], max_d[..., None, :]
        # num_rays = num_rays if num_rays is not None else opt.H * opt.W
        rand_samples = 0.5
        rand_samples += torch.arange(opt.SDF.VolSDF.sample_intvs, device=opt.device)[None, None, :,
                        None].float()  # [B,HW,N,1]
        depth_samples = rand_samples / opt.SDF.VolSDF.sample_intvs * (depth_max - depth_min) + depth_min  # [B,HW,N,1]

        return depth_samples

    def sample_depth_from_opacity(self, opt, depth_sample, opacity_approx):
        '''
        :param opt:
        :param opacity_approx:  [B,HW,N,1]
        :return:
        '''
        opacity_approx = torch.cat(
            [torch.zeros_like(opacity_approx[..., :1], device=opt.device), opacity_approx], -1
        )
        grid = torch.linspace(0, 1, opt.VolSDF.final_sample_intvs + 1, device=opt.device)  # [Nf+1]
        unif = 0.5 * (grid[:-1] + grid[1:]).repeat(*opacity_approx.shape[:-1], 1)  # [B,HW,Nf]
        idx = torch.searchsorted(opacity_approx, unif, right=False)  # [B,HW,Nf] \in {1...N}

        # inverse transform sampling from CDF
        depth_bin = depth_sample
        depth_low = depth_bin.gather(dim=-1, index=(idx - 1).clamp(min=0))  # [B,HW,Nf]
        depth_high = depth_bin.gather(dim=-1, index=idx.clamp(max=opacity_approx.shape[-1] - 1))  # [B,HW,Nf]
        cdf_low = opacity_approx.gather(dim=-1, index=(idx - 1).clamp(min=0))  # [B,HW,Nf]
        cdf_high = opacity_approx.gather(dim=-1, index=idx.clamp(max=opacity_approx.shape[-1] - 1))  # [B,HW,Nf]
        # linear interpolation
        t = (unif - cdf_low) / (cdf_high - cdf_low + 1e-8)  # [B,HW,Nf]
        depth_samples = depth_low + t * (depth_high - depth_low)  # [B,HW,Nf]
        return depth_samples[..., None]  # [B,HW,Nf,1]

    def opacity_to_sample(self, opt, depth_samp, sdf, alpha, beta, B, HW):
        sigma = self.sdf_to_sigma(sdf, alpha, beta)
        delta_i = depth_samp[..., 1:] - depth_samp[..., :-1]  # NOTE: already real depth
        R_t = torch.cat(
            [torch.zeros([*sdf.shape[:-1], 1], device=opt.device), torch.cumsum(sigma[..., :-1] * delta_i, dim=-1)],
            dim=-1)[..., :-1]
        # -------------- a fresh set of \hat{O}
        opacity_approx = 1 - torch.exp(-R_t)
        fine_dvals = self.sample_depth_from_opacity(opt, depth_samp, opacity_approx)
        return fine_dvals

    def sdf_to_sigma(self, sdf, alpha, beta):
        exp = 0.5 * torch.exp(-torch.abs(sdf) / beta)
        psi = torch.where(sdf >= 0, exp, 1 - exp)
        return alpha * psi

    def volsdf_sampling(self, opt, center,
                        ray, SDF_Field=None):
        '''
        :param opt:
        :param unisamp:       uniform sample points [B,HW,N,3]
        :param center:   [B,HW,3]        the center of camera
        :param ray:      [B,HW,3]        the ray
        :return:
        '''
        _, hits_t, _ = RayAABBIntersector.apply(center.view(-1, 3), ray.view(-1, 3), self.center.squeeze(0),
                                                self.half_size.squeeze(0), 1)
        d_min_max = hits_t.squeeze(-1).view([*center.shape[:2], -1])  # get the min&max by intersect
        batch_size = center.shape[0]
        if opt.SDF.VolSDF.volsdf_sampling == False:
            depth_coarse = self.sample_depth(opt, min_d=d_min_max[..., :1],
                                             max_d=d_min_max[..., 1:]).squeeze(-1)
            return depth_coarse, depth_coarse, depth_coarse
        with torch.no_grad():
            # init beta
            max_d = d_min_max[..., 1]
            max_d[torch.all(max_d == -1)] = 0

            eps_ = torch.tensor([opt.SDF.VolSDF.eps]).to(opt.device) * torch.ones_like(center[:, :, 0])
            beta = torch.sqrt((max_d ** 2) / (4 * (opt.SDF.VolSDF.sample_intvs - 1) * torch.log(1 + eps_)))
            alpha = 1. / beta
            # uniform sampling
            depth_samples = self.sample_depth(opt, min_d=d_min_max[..., :1],
                                              max_d=d_min_max[..., 1:])  # [B,HW,N,1]
            unisamp = camera.get_3D_points_from_depth(opt, center, ray, depth_samples, multi_samples=True)  # [B,HW,N,3]
            sdf = SDF_Field.infer_sdf(unisamp, mode="ret_sdf")  # [B,HW,N,1]
            alpha_graph, beta_graph = SDF_Field.forward_ab()  # [attain the alpha and beta
            net_bounds_max = self.error_bound(depth_samples.squeeze(-1), sdf.squeeze(-1), alpha_graph, beta_graph).max(
                dim=-1).values
            mask = net_bounds_max > opt.SDF.VolSDF.eps

            # cal bounds_error by beta++
            bounds = self.error_bound(depth_samples.squeeze(-1), sdf.squeeze(-1), alpha[:, :, None], beta[:, :, None])
            bounds_masked = bounds[mask]

            final_fine_dvals = torch.zeros([*mask.shape, opt.SDF.VolSDF.final_sample_intvs]).to(
                opt.device)  # [B,HW,N_final]
            final_iter_usage = torch.zeros([*mask.shape]).to(opt.device)  # [B,HW]
            final_converge_flag = torch.zeros([*mask.shape], dtype=torch.bool).to(opt.device)  # [B,HW]
            # if the sampling is fine
            if (~mask).sum() > 0:
                final_fine_dvals[~mask] = self.opacity_to_sample(opt, depth_samples.squeeze(-1)[~mask],
                                                                 sdf.squeeze(-1)[~mask], alpha_graph, beta_graph,
                                                                 *mask.shape)[:, :, 0]
                final_iter_usage[~mask] = 0
            final_converge_flag[~mask] = True

            current_samp = depth_samples.shape[-2]
            it_algo = 0
            depth_samples = depth_samples.squeeze(-1)
            sdf = sdf.squeeze(-1)
            while it_algo < opt.SDF.VolSDF.max_upsample_iter:
                # print("sampling algo iteration{}".format(it_algo))
                it_algo += 1
                if mask.sum() > 0:
                    # intuitively, the bigger bounds error, the more weights in sampling
                    upsampled_d_vals_masked = self.sample_pdf(depth_samples[mask], bounds_masked,
                                                              opt.SDF.VolSDF.sample_intvs + 2, det=True)[..., 1:-1]
                    # upsample_depth
                    depth_samples = torch.cat([depth_samples,
                                               torch.zeros([*depth_samples.shape[:2], opt.SDF.VolSDF.sample_intvs]).to(
                                                   opt.device)], dim=-1)
                    sdf = torch.cat([sdf, torch.zeros([*sdf.shape[:2], opt.SDF.VolSDF.sample_intvs]).to(opt.device)],
                                    dim=-1)

                    depth_samples_masked = depth_samples[mask]
                    sdf_masked = sdf[mask]
                    # add the hierachical sampled depth into the new one
                    depth_samples_masked[...,
                    current_samp:current_samp + opt.SDF.VolSDF.sample_intvs] = upsampled_d_vals_masked
                    depth_samples_masked, sort_indices_masked = torch.sort(depth_samples_masked, dim=-1)
                    # add the hierachical sampled sdf into the new one
                    tem_samp_pts3D = center[mask][..., None, :] + ray[mask][..., None, :] * upsampled_d_vals_masked[...,
                                                                                            :, None]
                    sdf_masked[..., current_samp:current_samp + opt.SDF.VolSDF.sample_intvs] = SDF_Field(
                        tem_samp_pts3D, mode="ret_sdf")[:, :, 0]
                    sdf_masked = torch.gather(sdf_masked, dim=-1,
                                              index=sort_indices_masked)
                    # update depth_samples and sdf
                    depth_samples[mask] = depth_samples_masked
                    sdf[mask] = sdf_masked
                    current_samp += opt.SDF.VolSDF.sample_intvs

                    # using the beta_graph to cal the new bound error
                    net_bounds_max = self.error_bound(depth_samples, sdf, alpha_graph, beta_graph).max(dim=-1).values
                    sub_mask_of_mask = net_bounds_max[mask] > opt.SDF.VolSDF.eps
                    converged_mask = mask.clone()
                    converged_mask[mask] = ~sub_mask_of_mask

                    if (converged_mask).sum() > 0:
                        final_fine_dvals[converged_mask] = self.opacity_to_sample(opt, depth_samples[converged_mask],
                                                                                  sdf[converged_mask], alpha_graph,
                                                                                  beta_graph,
                                                                                  *converged_mask.shape).squeeze(-1)
                        final_iter_usage[converged_mask] = it_algo
                        final_converge_flag[converged_mask] = True
                    # using the bisection method to find the new beta++
                    if (sub_mask_of_mask).sum() > 0:
                        # mask-the-mask approach
                        new_mask = mask.clone()
                        new_mask[mask] = sub_mask_of_mask
                        # [Submasked, 1]
                        beta_right = beta[new_mask]
                        beta_left = beta_graph * torch.ones_like(beta_right, device=opt.device)
                        d_samp_tmp = depth_samples[new_mask]
                        sdf_tmp = sdf[new_mask]
                        # ----------------
                        # Bisection iterations
                        for _ in range(opt.VolSDF.max_bisection_itr):
                            beta_tmp = 0.5 * (beta_left + beta_right)
                            alpha_tmp = 1. / beta_tmp
                            # alpha_tmp = alpha_net
                            # [Submasked]
                            bounds_tmp_max = self.error_bound(d_samp_tmp, sdf_tmp, alpha_tmp[:, None],
                                                              beta_tmp[:, None]).max(dim=-1).values
                            beta_right[bounds_tmp_max <= opt.SDF.VolSDF.eps] = beta_tmp[
                                bounds_tmp_max <= opt.SDF.VolSDF.eps]
                            beta_left[bounds_tmp_max > opt.SDF.VolSDF.eps] = beta_tmp[
                                bounds_tmp_max > opt.SDF.VolSDF.eps]
                        # updata beta++ and alpha++
                        beta[new_mask] = beta_right
                        alpha[new_mask] = 1. / beta[new_mask]

                        # ----------------
                        # after upsample, the remained rays that not yet converged.
                        # ----------------
                        bounds_masked = self.error_bound(d_samp_tmp, sdf_tmp, alpha[new_mask][:, None],
                                                         beta[new_mask][:, None])
                        # bounds_masked = error_bound(d_vals_tmp, rays_d_tmp, sdf_tmp, alpha_net, beta[new_mask])
                        bounds_masked = torch.clamp(bounds_masked, 0, 1e5)  # NOTE: prevent INF caused NANs

                        # mask = net_bounds_max > eps   # NOTE: the same as the following
                        mask = new_mask
                    else:
                        break
                else:
                    break
            # ----------------
            # for rays that still not yet converged after max_iter, use the last beta+
            # ----------------
            if (~final_converge_flag).sum() > 0:
                # print("existing rays which did not converge")
                beta_plus = beta[~final_converge_flag]
                alpha_plus = 1. / beta_plus
                final_fine_dvals[~final_converge_flag] = self.opacity_to_sample(opt,
                                                                                depth_samples[~final_converge_flag],
                                                                                sdf[~final_converge_flag],
                                                                                alpha_plus[:, None], beta_plus[:, None],
                                                                                *final_converge_flag.shape).squeeze(-1)
                final_iter_usage[~final_converge_flag] = -1
            beta[final_converge_flag] = beta_graph
            depth_coarse = self.sample_depth(opt, min_d=d_min_max[..., :1],
                                             max_d=d_min_max[..., 1:]).squeeze(-1)
            depth_sample_final = torch.cat([final_fine_dvals, depth_coarse], dim=-1)
            depth_sample_final, _ = torch.sort(depth_sample_final, dim=-1)
            return depth_sample_final, beta, final_iter_usage

    def error_bound(self, d_vals, sdf, alpha, beta):
        """
        @ Bound on the opacity approximation error
        mentioned in paper, bounds error is calculated between xi-1 and xi
        Args:
            d_vals: [..., N_pts]
            sdf:    [..., N_pts]
        Return:
            bounds: [..., N_pts-1]
        """
        device = sdf.device
        sigma = self.sdf_to_sigma(sdf, alpha, beta)  # [..., N_pts]
        sdf_abs_i = torch.abs(sdf)  # [..., N_pts-1]
        # delta_i = (d_vals[..., 1:] - d_vals[..., :-1]) * rays_d.norm(dim=-1)[..., None]
        delta_i = d_vals[..., 1:] - d_vals[..., :-1]  # NOTE: already real depth
        # [..., N_pts-1]. R(t_k) of the starting point of the interval.
        R_t = torch.cat(
            [
                torch.zeros([*sdf.shape[:-1], 1], device=device),
                torch.cumsum(sigma[..., :-1] * delta_i, dim=-1)
            ], dim=-1)[..., :-1]  # [..., N_pts-1]
        d_i_star = torch.clamp_min(0.5 * (sdf_abs_i[..., :-1] + sdf_abs_i[..., 1:] - delta_i), 0.)  # [..., N_pts-1]
        errors = alpha / (4 * beta) * (delta_i ** 2) * torch.exp(
            -d_i_star / beta)  # [..., N_pts-1]. E(t_{k+1}) of the ending point of the interval.
        errors_t = torch.cumsum(errors, dim=-1)  # [..., N_pts-1]
        bounds = torch.exp(-R_t) * (torch.exp(errors_t) - 1.)
        # TODO: better solution
        #     # NOTE: nan comes from 0 * inf
        #     # NOTE: every situation where nan appears will also appears c * inf = "true" inf, so below solution is acceptable
        bounds[torch.isnan(bounds)] = np.inf
        return bounds

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        # device = weights.get_device()
        device = weights.device
        # Get pdf
        weights = weights + 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat(
            [torch.zeros_like(cdf[..., :1], device=device), cdf], -1
        )  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0.0, 1.0, steps=N_importance, device=device)
            u = u.expand(list(cdf.shape[:-1]) + [N_importance])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)
        u = u.contiguous()

        # Invert CDF
        inds = torch.searchsorted(cdf.detach(), u, right=False)

        below = torch.clamp_min(inds - 1, 0)
        above = torch.clamp_max(inds, cdf.shape[-1] - 1)
        # (batch, N_importance, 2) ==> (B, batch, N_importance, 2)
        inds_g = torch.stack([below, above], -1)

        matched_shape = [*inds_g.shape[:-1], cdf.shape[-1]]  # fix prefix shape

        cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
        bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)  # fix prefix shape

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom[denom < eps] = 1
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples
