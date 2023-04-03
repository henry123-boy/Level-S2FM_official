import copy
import pdb

import numpy as np
import os,sys,time
import torch

import imageio
from tqdm import tqdm
import pickle
import cv2
import skimage
from skimage.transform import rescale
from . import base
from utils import camera

class Dataset(base.Dataset):
    def __init__(self,opt,split="train",subset=None):

        self.raw_H,self.raw_W = 1200,1600
        height_,width_=opt.data.image_size[0],opt.data.image_size[1]
        factor_y=self.raw_H/height_
        factor_x=self.raw_W/width_
        self.factor_x=factor_x
        self.factor_y=factor_y

        super().__init__(opt,split)
        self.root = opt.data.root or "data/DTU"
        self.path = "{}/{}".format(self.root,opt.data.scene)
        if os.path.exists("{0}/{1}".format(self.path,opt.data.scene))==True:
            root_data="{0}/{1}".format(self.path,opt.data.scene)
        else:
            root_data = self.path
        self.path_image = "{}/images".format(root_data)
        self.path_cam = '{}/cameras.npz'.format(root_data)
        image_fnames = sorted(os.listdir(self.path_image))
        image_fnames=[os.path.join(self.path_image,img_f_i) for img_f_i in  image_fnames]
        self.n_images = len(image_fnames)
        # loading the pose
        camera_dict = np.load(self.path_cam)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        self.intrinsics_all = []
        self.c2w_all = []
        cam_center_norms = []
        # ------------ save pose files ---------------------
        if os.path.exists(f"{self.path}/pose")==False:
            os.makedirs(f"{self.path}/pose",exist_ok=True)
        img_names=sorted(os.listdir(self.path_image))
        for scale_mat, world_mat,file_name_i in zip(scale_mats, world_mats,img_names):
            P = world_mat @ scale_mat                      # c2w
            P = P[:3, :4]
            intrinsics, pose = self.load_K_Rt_from_P(P)
            cam_center_norms.append(np.linalg.norm(pose[:3, 3]))
            # downscale intrinsics
            intrinsics[0, 2] /= factor_x
            intrinsics[1, 2] /= factor_y
            intrinsics[0, 0] /= factor_x
            intrinsics[1, 1] /= factor_y
            if os.path.exists(f"{self.path}/intrinsics.txt")==False:
                np.savetxt(f"{self.path}/intrinsics.txt", intrinsics)
            # intrinsics[0, 1] /= downscale # skew is a ratio, do not scale
            if os.path.exists(f"{self.path}/pose/{file_name_i[:-4]}.txt") == False:
                intrinsics_org, pose_org = self.load_K_Rt_from_P(world_mat[:3, :4])
                np.savetxt(f"{self.path}/pose/{file_name_i[:-4]}.txt", pose_org)
            self.intrinsics_all.append(torch.from_numpy(intrinsics[:3,:3]).float())
            self.c2w_all.append(torch.from_numpy(pose[:3]).float())

        self.rgb_images = []
        self.list = image_fnames
        image_fnames = image_fnames
        self.n_images =len(image_fnames)
        self.c2w_all = self.c2w_all
        self.intrinsics_all = self.intrinsics_all
        self.list = image_fnames
        downscale=min(factor_x,factor_y)
        for path in tqdm(image_fnames, desc='loading images...'):
            rgb = self.load_rgb(path, downscale)
            imgname=path.split("/")[-1]
            rgb_tensor=torch.from_numpy(rgb).float()
            # rgb_tensor[:,torch.all(rgb_tensor <= 0.1, dim=0)] = 1.0
            self.rgb_images.append(rgb_tensor)
        self.object_masks = []

    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])
    def load_rgb(self,path, downscale=1):
        img = imageio.imread(path)
        img = skimage.img_as_float32(img)
        if downscale != 1:
            img = rescale(img, 1. / downscale, anti_aliasing=False, multichannel=True)

        # NOTE: pixel values between [-1,1]
        # img -= 0.5
        # img *= 2.
        img = img.transpose(2, 0, 1)
        return img

    def rotmat2qvec(self,R):
        Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
        K = np.array([
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
        eigvals, eigvecs = np.linalg.eigh(K)
        qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
        if qvec[0] < 0:
            qvec *= -1
        return qvec
    def load_mask(self,path, downscale=1):
        alpha = imageio.imread(path, as_gray=True)
        alpha = skimage.img_as_float32(alpha)
        if downscale != 1:
            alpha = rescale(alpha, 1. / downscale, anti_aliasing=False, multichannel=False)
        object_mask = alpha > 127.5

        return object_mask

    def load_K_Rt_from_P(self,P):
        """
        modified from IDR https://github.com/lioryariv/idr
        """
        out = cv2.decomposeProjectionMatrix(P)

        K = out[0]
        R = out[1]
        t = out[2]

        K = K / K[2, 2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]

        return intrinsics, pose

    def __getitem__(self,idx):
        opt = self.opt
        sample = dict(idx=idx)
        image = self.rgb_images[idx]
        intr, pose = self.intrinsics_all[idx][:3,:3], camera.pose.invert(self.c2w_all[idx][:3,:])
        sample.update(
            image=image,
            intr=intr,
            pose=pose
        )
        return sample

