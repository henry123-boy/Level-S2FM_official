import pdb

import torch
import torch.nn as nn
import tinycudann as tcnn
import json
import numpy as np
import utils.util as util


# --------------- Hash encoding for the geometry -------------------------------
class Embedder_Hash(nn.Module):
    def __init__(self, kwargs,
                 include_input=True,
                 input_dim=3):
        super(Embedder_Hash, self).__init__()
        self.embedder_obj = tcnn.Encoding(n_input_dims=input_dim, encoding_config=kwargs)
        self.input_dim = input_dim
        self.out_dim = self.embedder_obj.n_output_dims
        self.out_dim += self.input_dim
        self.include_input=include_input

    def forward(self,
                input: torch.Tensor,
                bound_min: torch.Tensor,
                bound_max: torch.Tensor,
                rescale: float = 1.0):
        """
        input: the xyz position
        rescale: control the initialization of the SDF's sphere
        """
        # normalized into [0,1]
        assert (input.shape[-1] == self.input_dim)

        norm_input = (input - bound_min.to(input.device)) / ((bound_max - bound_min).to(input.device))
        org_shape=input.shape
        out = self.embedder_obj(norm_input.view(-1,3))
        if self.include_input:
            out = torch.cat([input / rescale, out.view([*org_shape[:-1],-1])], dim=-1)
        return out


class Embedder_Fourier(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder_Fourier, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self,
                input: torch.Tensor,
                bound_min: torch.Tensor=None,
                bound_max: torch.Tensor=None,
                rescale: float = 1.0):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input/rescale)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


# ---------------- create Embedder ---------------------------------------------
def get_Embedder(opt,
                 input_dim=3,
                 input_choice="Hash",
                 choices=('Hash', 'Fourier', 'SH')):
    if input_choice not in choices:
        raise ValueError(f'Invalid input option. Valid choices are: {choices}')

    if input_choice == "Hash":
        """
        config file structure
        "encoding"{
        "otype": "HashGrid",
        "n_levels": ,
        "n_features_per_level": ,
        "log2_hashmap_size": ,
        "base_resolution": ,
        "per_level_scale":
        }
        """
        embed_config = opt.SDF.Hash_config.config_file
        with open(embed_config) as config_file:
            config_surface = json.load(config_file)
        enc_cfg = config_surface["encoding"]
        L = enc_cfg["n_levels"]
        F = enc_cfg["n_features_per_level"]
        log2_T = enc_cfg["log2_hashmap_size"]
        N_min = enc_cfg["base_resolution"]
        scale = (opt.data.bound_max[0] - opt.data.bound_min[0]) / 2
        b_ = np.exp(np.log(2048 * scale / N_min) / (L - 1))
        kwargs = {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": L,
            "n_features_per_level": F,
            "log2_hashmap_size": log2_T,
            "base_resolution": N_min,
            "per_level_scale": b_,
            "interpolation": "Linear"
        }
        embed_obj = Embedder_Hash(kwargs=kwargs, input_dim=input_dim)

    elif input_choice=="Fourier":
        embed_kwargs = {
            "include_input": True,  # needs to be True for ray_bending to work properly
            "input_dim": input_dim,
            "max_freq_log2": 4 - 1,
            "N_freqs": 4,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }
        embed_obj = Embedder_Fourier(**embed_kwargs)

    elif input_choice=="SH":
        embed_config = opt.RadF.Hash_config.config_file
        with open(embed_config) as config_file:
            config_surface = json.load(config_file)
        enc_cfg = config_surface["encoding"]
        embed_obj = tcnn.Encoding(n_input_dims=input_dim, encoding_config=enc_cfg)

    return embed_obj


# --------------- Make MLP and Init for Geometry -------------------------------
class Geometry(nn.Module):
    def __init__(self,
                 opt,
                 input_dim,
                 layers,
                 skip=[],
                 tf_init=True
                 ):
        super(Geometry, self).__init__()
        self.mlp = torch.nn.ModuleList()
        L_surface = layers
        bias = opt.SDF.NN_Init.bias

        self.skip = skip
        # create surface
        for li, (k_in, k_out) in enumerate(L_surface):
            if li == 0: k_in = input_dim
            if li in skip: k_in += input_dim
            if li == len(L_surface) - 1: k_out += 1  # 256+1   1 is for SDF, and 256 is the dim of position feature
            linear = torch.nn.Linear(k_in, k_out)
            if tf_init:
                if li == len(L_surface) - 1:
                    torch.nn.init.normal_(linear.weight, mean=np.sqrt(np.pi) / np.sqrt(L_surface[li][0]), std=0.0001)
                    torch.nn.init.constant_(linear.bias, -bias)
                elif li == 0:
                    torch.nn.init.constant_(linear.bias, 0.0)
                    torch.nn.init.constant_(linear.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(linear.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(k_out))
                elif li in opt.SDF.arch.skip:
                    torch.nn.init.constant_(linear.bias, 0.0)
                    torch.nn.init.normal_(linear.weight, 0.0, np.sqrt(2) / np.sqrt(k_out))
                    torch.nn.init.constant_(linear.weight[:, -(input_dim - 3):],
                                            0.0)  # NOTE: this contrains the concat order to be  [h, x_embed]
                else:
                    torch.nn.init.constant_(linear.bias, 0.0)
                    torch.nn.init.normal_(linear.weight, 0.0, np.sqrt(2) / np.sqrt(k_out))
            linear = nn.utils.weight_norm(linear)
            self.mlp.append(linear)
        # define activate
        self.softplus = nn.Softplus(beta=100, threshold=20)
        self.relu = nn.ReLU()

    def forward(self,
                points_enc):
        """
        points_enc: the encoding after the positional encode
        """
        feat=points_enc
        for li, layer in enumerate(self.mlp):
            if li in self.skip: feat = torch.cat([feat, points_enc], dim=-1) / np.sqrt(2)
            feat = layer(feat)
            if li <= len(self.mlp) - 2:
                feat = self.softplus(feat)
        return feat

# --------------- Make MLP and Init for Radiance -------------------------------

class Radiance(nn.Module):
    def __init__(self,
                 opt,
                 input_dim,
                 layers,
                 skip=[],
                 tf_init=True
                 ):
        super(Radiance, self).__init__()
        self.mlp = torch.nn.ModuleList()
        L_radiance = layers
        self.skip = skip
        self.opt=opt

        # create surface
        self.mlp_radiance = torch.nn.ModuleList()
        for li, (k_in, k_out) in enumerate(L_radiance):
            if li == 0: k_in = input_dim
            linear = torch.nn.Linear(k_in, k_out)
            if tf_init:
                linear = nn.utils.weight_norm(linear)
            self.mlp_radiance.append(linear)

        # define activate
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self,
                geo_enc):
        """
        geo_enc: the encoding after the geometry network
        """
        feat=geo_enc
        for li, layer in enumerate(self.mlp_radiance):
            feat = layer(feat)
            if li <= len(self.mlp) - 2:
                feat = self.relu(feat)
        out = self.sigmoid(feat)

        return out

