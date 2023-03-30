import os
import json
import pdb

import torch
import numpy as np
from tqdm import tqdm
import cv2
from . import base
import camera
import torch.nn.functional as torch_F
# from utils.io_util import load_mask, load_rgb, glob_imgs
# from utils.rend_util import rot_to_quat, load_K_Rt_from_P
class Dataset(base.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,opt,split="train",subset=None):
        self.opt=opt
        self.root = opt.data.root or "data/scannet"
        self.root=os.path.join(self.root,opt.data.scene)
        assert os.path.exists(self.root), "Data directory is empty"
        self.downscale = opt.data.down_factor

        image_dir = '{0}/images'.format(self.root)
        # image_dir = '{0}/pose'.format(self.instance_dir)`
        image_list = os.listdir(image_dir)
        image_list = [id for id in image_list if int(id.split(".")[0])%opt.freq_frame==0]
        image_list.sort(key=lambda _: int(_.split('.')[0]))
        self.n_images = len(image_list)

        cam_center_norms = []
        center=[0,0,0]
        n_cam=0
        self.intrinsics_all = []
        self.c2w_all = []
        self.rgb_images = []
        self.depth_gt=[]
        self.norm_omnidata=[]
        self.depth_omnidata=[]
        self.raw_H,self.raw_W = 968, 1296
        height_,width_=opt.data.image_size[0],opt.data.image_size[1]
        factor_y=self.raw_H/height_
        factor_x=self.raw_W/width_
        intrinsics = np.loadtxt(f'{self.root}/intrinsic/intrinsic_color.txt')
        intrinsics[0,0]/=factor_x
        intrinsics[1, 1] /= factor_y
        intrinsics[0, 2] /= factor_x
        intrinsics[1, 2] /= factor_y
        self.factor_x = factor_x
        self.factor_y = factor_y
        if split!="train":
            image_list=image_list[:3]
        for imgname in tqdm(image_list, desc='loading dataset...'):
            c2w = np.loadtxt(f'{self.root}/pose/{imgname[:-4]}.txt')
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            if self.opt.data.init:     # init by colmap
                pdb.set_trace()
                colmap_root=f"{self.root}/cam"
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
            cam_center_norms.append(np.linalg.norm(c2w[:3, 3]))
            self.c2w_all.append(torch.from_numpy(c2w).float())
            rgb = cv2.imread(f'{image_dir}/{imgname[:-4]}.jpg')
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = (rgb.astype(np.float32) / 255).transpose(2, 0, 1)
            depth = cv2.imread(f'{self.root}/depth/{imgname[:-4]}.png', -1) / 1000
            self.depth_gt.append(depth)
            norm_=np.load(f'{self.root}/omnidata_norm/{imgname[:-4]}_normal.npy')
            depth_=np.load(f'{self.root}/omnidata_depth/{imgname[:-4]}_depth.npy')
            self.norm_omnidata.append(norm_)
            self.depth_omnidata.append(depth_.squeeze(0))
            _, self.H, self.W = rgb.shape
            # rgb = rgb.reshape(3, -1).transpose(1, 0)
            rgb_tensor=torch_F.interpolate(torch.from_numpy(rgb).float().unsqueeze(0),size=[height_,width_]).squeeze(0)
            self.rgb_images.append(rgb_tensor)
        if self.opt.data.center:
            scale_radius=self.opt.rad
            self.center=center/n_cam
            cam_center_norms=[]
            for cam_i in self.c2w_all:
                cam_i[:3,3]-=self.center
                cam_center_norms.append(torch.norm(cam_i[:3,3]))
            max_cam_norm = max(cam_center_norms)
            scale = (scale_radius / max_cam_norm / 1.1)
            for cam_i in self.c2w_all:
                cam_i[:3,3]*=scale
            print(scale_radius, max_cam_norm, scale)
            # ## test
            # cam_test = {}
            # import util
            # for id,cam_i,intr in zip(range(len(self.c2w_all)),self.c2w_all,self.intrinsics_all):
            #     camera_i = {"{}".format(id): {"K": util.intr2list(intr[:3,:3]),
            #                                         "W2C": util.pose2list(camera.pose.invert(cam_i)),
            #                                         "img_size": opt.data.image_size}}
            #     cam_test.update(camera_i)
            # pdb.set_trace()
            # util.dict2json(os.path.join("./", 'cam{:08d}.json'.format(0)), cam_test)
            # pdb.set_trace()

        # max_cam_norm = max(cam_center_norms)
        # scale = (scale_radius / max_cam_norm / 1.1)
        # print(scale_radius, max_cam_norm, scale)

        # if scale_radius > 0:
        #     for i in range(len(self.c2w_all)):
        #         self.c2w_all[i][:3, 3] *= scale
        #         self.depth[i] *= scale

    def __len__(self):
        return len(self.c2w_all)

    def __getitem__(self, idx):
        opt = self.opt
        sample = dict(idx=idx)
        image = self.rgb_images[idx]
        intr, pose = self.intrinsics_all[idx][:3,:3],camera.pose.invert(self.c2w_all[idx][:3,:])
        depth_gt=self.depth_gt[idx]
        norm_omnidata=self.norm_omnidata[idx]
        depth_omnidata=self.depth_omnidata[idx]
        #intr, pose = self.intrinsics_all[idx][:3,:3],camera.pose.invert(self.c2w_all[idx][:3,:])
        sample.update(
            image=image,
            intr=intr,
            pose=pose,
            depth_gt=depth_gt,
            norm_omnidata=norm_omnidata,
            depth_omnidata=depth_omnidata
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

    def get_gt_pose(self, scaled=True):
        # Load gt pose without normalization to unit sphere
        camera_dict = json.load(open(self.cam_file))

        c2w_all = []
        for imgname, v in camera_dict.items():
            world_mat = np.array(v['P'], dtype=np.float32).reshape(4, 4)
            if scaled and 'SCALE' in v:
                scale_mat = np.array(v['SCALE'], dtype=np.float32).reshape(4, 4)
                P = world_mat @ scale_mat
            else:
                P = world_mat
            _, c2w = load_K_Rt_from_P(P[:3, :4])
            c2w_all.append(torch.from_numpy(c2w).float())

        return torch.cat([p.float().unsqueeze(0) for p in c2w_all], 0)
    def get_all_camera_poses(self,opt):
        # poses are unknown, so just return some dummy poses (identity transform)
        pose_ref=camera.pose.invert(self.c2w_all[:3,:])
        return camera.pose(t=torch.zeros(len(self),3))


if __name__ == "__main__":
    # dataset = SceneDataset(False, './data/taxi/black')
    dataset = SceneDataset(False, './data/taxi/blue')
    c2w = dataset.get_gt_pose(scaled=True).data.cpu().numpy()
    extrinsics = np.linalg.inv(c2w)  # camera extrinsics are w2c matrix
    camera_matrix = next(iter(dataset))[1]['intrinsics'].data.cpu().numpy()

    from tools.vis_camera import visualize

    visualize(camera_matrix, extrinsics)