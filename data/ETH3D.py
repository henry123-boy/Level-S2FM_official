import os
import json
import pdb

import torch
import numpy as np
from tqdm import tqdm
import cv2
from . import base
import utils.camera as camera
import torch.nn.functional as torch_F
# from utils.io_util import load_mask, load_rgb, glob_imgs
# from utils.rend_util import rot_to_quat, load_K_Rt_from_P
class Dataset(base.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,opt,split="train",subset=None):
        self.opt=opt
        self.root = opt.data.root or "data/ETH3D"
        self.root=os.path.join(self.root,opt.data.scene)
        assert os.path.exists(self.root), "Data directory is empty"

        image_dir = '{0}/images'.format(self.root)
        # image_dir = '{0}/pose'.format(self.instance_dir)`
        image_list = os.listdir(image_dir)
        # image_list = [id for id in image_list if id.endswith("0.jpg")]
        image_list.sort(key=lambda _: int((_.split('.')[0]).split("_")[1]))
        self.n_images = len(image_list)
        self.img_names=image_list
        cam_center_norms = []
        center=[0,0,0]
        n_cam=0
        self.intrinsics_all = []
        self.c2w_all = []
        self.rgb_images = []
        self.raw_H,self.raw_W = 4134, 6204
        height_,width_=opt.data.image_size[0],opt.data.image_size[1]
        factor_y=self.raw_H/height_
        factor_x=self.raw_W/width_
        intrinsics = np.loadtxt(f'{self.root}/intrinsics.txt')
        intrinsics[0,0]/=factor_x
        intrinsics[1, 1] /= factor_y
        intrinsics[0, 2] /= factor_x
        intrinsics[1, 2] /= factor_y
        self.factor_x = factor_x
        self.factor_y = factor_y
        self.downscale=min(factor_x,factor_y)


        for imgname in tqdm(image_list, desc='loading dataset...'):
            c2w = camera.pose.invert(torch.from_numpy(np.loadtxt(f'{self.root}/pose/{imgname[:-4]}.txt'))[:3,:]).squeeze().numpy()
            cam_center_norms.append(np.linalg.norm(c2w[:3, 3]))
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float()[:3,:3])

            if self.opt.data.init:     # init by colmap
                if self.opt.model=="barf":
                    colmap_root=f"{self.root}/cam"
                else:
                    colmap_root = f"{self.root}/rec_3rd/rec_model/cam"
                if os.path.exists(colmap_root + "/" + imgname[:-4]+".cam")==False:
                    continue
                with open(colmap_root + "/" + imgname[:-4]+".cam", "r") as f:
                    lines = f.readlines()
                    int_lines = [float(a.strip("\n")) for a in lines[0].split(" ")]
                    t_=np.array(int_lines[:3]).reshape(1, 3, -1)
                    rot_=np.array(int_lines[3:]).reshape(1, 3, 3)
                    f.close()
                pose=np.concatenate([rot_,t_],axis=-1)
                c2w=camera.pose.invert(torch.from_numpy(pose)).squeeze().numpy()
            center+=c2w[:3,3]
            n_cam+=1
            self.c2w_all.append(torch.from_numpy(c2w).float())
            rgb = cv2.imread(f'{image_dir}/{imgname[:-4]}.jpg')
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = (rgb.astype(np.float32) / 255).transpose(2, 0, 1)
            _, self.H, self.W = rgb.shape
            # rgb = rgb.reshape(3, -1).transpose(1, 0)
            rgb_tensor=torch_F.interpolate(torch.from_numpy(rgb).float().unsqueeze(0),size=[height_,width_]).squeeze(0)
            self.rgb_images.append(rgb_tensor)
        if self.opt.data.center:
            scale_radius = self.opt.rad
            self.center = center / n_cam
            cam_center_norms = []
            for cam_i in self.c2w_all:
                cam_i[:3, 3] -= self.center
                cam_center_norms.append(torch.norm(cam_i[:3, 3]))
            max_cam_norm = max(cam_center_norms)
            scale = (scale_radius / max_cam_norm / 1.1)
            for cam_i in self.c2w_all:
                cam_i[:3, 3] *= scale
            print(scale_radius, max_cam_norm, scale)
            ## test
            cam_test = {}
            import util
            for id,cam_i,intr in zip(range(len(self.c2w_all)),self.c2w_all,self.intrinsics_all):
                camera_i = {"{}".format(id): {"K": util.intr2list(intr[:3,:3]),
                                                    "W2C": util.pose2list(camera.pose.invert(cam_i[:3,:])),
                                                    "img_size": opt.data.image_size}}
                cam_test.update(camera_i)
            util.dict2json(os.path.join("./", 'cam{:08d}.json'.format(0)), cam_test)



    def __len__(self):
        return len(self.c2w_all)
    def get_all_camera_poses(self,opt):
        # poses are unknown, so just return some dummy poses (identity transform)
        poses_ref=[pose[:3, :].unsqueeze(0) for pose in self.c2w_all]
        poses_ref=camera.pose.invert(torch.cat(poses_ref,dim=0))
        return poses_ref
    def __getitem__(self, idx):
        opt = self.opt
        sample = dict(idx=idx)
        image = self.rgb_images[idx]
        intr, pose = self.intrinsics_all[idx][:3,:3],camera.pose.invert(self.c2w_all[idx][:3,:])
        sample.update(
            image=image,
            intr=intr,
            pose=pose,
            img_name=self.img_names[idx]
        )
        return sample
    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])
    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)



