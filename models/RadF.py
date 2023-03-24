import pdb

import torch
import torch.nn as nn
import numpy as np
from .base import get_Embedder, Geometry, Radiance
import utils.util as util


class RadF(nn.Module):  # radiance field
    def __init__(self, opt):
        super(RadF, self).__init__()

        # opt config
        self.opt = opt

        # hash bound config
        self.bound_max = torch.from_numpy(np.array(opt.data.bound_max)).to(  # [1,1,3]  the boundary
            opt.device)[None, None, :].float()
        self.bound_min = torch.from_numpy(np.array(opt.data.bound_min)).to(  # [1,1,3]
            opt.device)[None, None, :].float()
        self.center = (self.bound_max + self.bound_min) / 2
        self.half_size = (self.bound_max - self.bound_min) / 2

        # volsdf rad config
        self.rescale = opt.SDF.VolSDF.rescale
        self.define_network(opt)

    def define_network(self, opt):

        if self.opt.Ablate_config.dual_field == True:
            # use another field to represent radiance

            # acquire the embedder
            self.embed_fn = get_Embedder(opt=opt,
                                         input_dim=3,
                                         input_choice="Hash"
                                         )
            input_3D_dim = self.embed_fn.out_dim

            # acquire the geometry encoder
            self.Geo_enc = Geometry(opt=opt,
                                    input_dim=input_3D_dim,
                                    skip=opt.SDF.arch.skip,
                                    tf_init=opt.SDF.NN_Init.tf_init,
                                    layers=util.get_layer_dims(opt.SDF.arch.layers))

        self.embed_fn_v = get_Embedder(opt=opt,
                                       input_dim=3,
                                       input_choice="Fourier"
                                       )

        # the radiance decoder for color
        input_enc_dim = 3 + self.embed_fn_v.out_dim + 3 \
                        + util.get_layer_dims(opt.SDF.arch.layers)[-1][-1]  # 3 point + geo_enc + normal 3 + view_emb

        if self.opt.Ablate_config.dual_field == True:
            input_enc_dim += util.get_layer_dims(opt.SDF.arch.layers)[-1][-1]  # extra geo_enc

        self.Rad_dec = Radiance(opt=opt,
                                input_dim=input_enc_dim,
                                skip=opt.SDF.arch.skip,
                                tf_init=opt.SDF.NN_Init.tf_init,
                                layers=util.get_layer_dims(opt.RadF.arch.layers))

    def Geometry_feat(self, xyz):
        # get the enc from the embed fn
        enc = self.embed_fn(xyz,
                            rescale=self.rescale,
                            bound_min=self.bound_min,
                            bound_max=self.bound_max
                            )

        feat = self.Geo_enc(enc)

        return feat

    def infer_embed_v(self, ray_utils):
        return self.embed_fn_v(ray_utils)

    def infer_app(self, geo_enc):
        """
        rgbs: [...,3]
        """
        rgbs = self.Rad_dec(geo_enc)
        return rgbs
