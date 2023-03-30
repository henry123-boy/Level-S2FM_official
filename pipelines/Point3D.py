import os, sys, time
import torch

epsilon = 1e-6
from typing import Optional


class Point3D():
    def __init__(self,
                 opt,
                 id: Optional[int],
                 x: Optional[float] = None,
                 y: Optional[float] = None,
                 z: Optional[float] = None,
                 xyz: Optional[torch.Tensor] = None):
        super(Point3D, self).__init__()
        """
        :param opt:
        :param id:
        :param x:
        :param y:
        :param z:
        """
        self.opt = opt,
        self.id = id,
        if xyz is None:
            self.xyz = torch.tensor([[x, y, z]]).to(opt.device)  # [1,3]
        else:
            self.xyz = xyz
        self.feat_track = []

    def update_xyz(self,
                   xyz_new
                   ):
        self.xyz = xyz_new

    def update_feat_track(self,
                          track_i: Optional[list]):
        """
        :param track_i:   [(camera_idx,kypts_idx)]   第一项为track上的相机id, 第二项为track相机上对应kypts的id
        :return:
        """
        self.feat_track += track_i


class Point3DSet():
    def __init__(self, opt):
        super(Point3DSet, self).__init__()
        self.opt = opt
        self.pointset = []  # Point3Ds

    def __len__(self):
        return len(self.pointset)

    def add_point3d(self,
                    xyz,  # xyz [1,3]
                    feat_tracks: list):
        idx = self.__len__()  # 当前point的一个编号
        point3d_i = Point3D(self.opt, id=idx, xyz=xyz)
        point3d_i.update_feat_track(feat_tracks)  # 加入当前点的feat_track
        self.pointset.append(point3d_i)
        return idx

    def get_all_parameters(self):
        xyzs = []
        feat_tracks = []
        for pts_i in self.pointset:
            xyzs.append(pts_i.xyz)
            feat_tracks.append(pts_i.feat_track)
        paras = {"xyzs": xyzs,
                 "feat_tracks": feat_tracks}
        return paras

    def update_feat_tracks(self,
                           idxs: list,
                           feat_tracks: list):
        for id, feat_track_i in zip(idxs, feat_tracks):
            self.pointset[id].update_feat_track([feat_track_i])

    def update_xyzs(self,
                    idxs: list,
                    xyzs_new: list):
        update_xyz = lambda idx, xyz_new: self.pointset[idx].update_xyz(xyz_new)
        map(update_xyz, idxs, xyzs_new)

    def get_xyzs(self,
                 idxs: Optional[list]):
        get_xyz_i = lambda i: self.pointset[i].xyz
        return list(map(get_xyz_i, idxs))

    def get_feat_tracks(self,
                        idxs: Optional[list]):
        get_xyz_i = lambda i: self.pointset[i].feat_track
        return list(map(get_xyz_i, idxs))
