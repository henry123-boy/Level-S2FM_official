import os, sys, time
import shutil
import datetime
import torch
import torch.nn.functional as torch_F
import ipdb
import termcolor
import socket
import contextlib
from easydict import EasyDict as edict
import numpy as np
from tqdm import tqdm
import plyfile
import skimage
import open3d as o3d
# convert to colored strings
def red(message, **kwargs): return termcolor.colored(str(message), color="red",
                                                     attrs=[k for k, v in kwargs.items() if v is True])


def green(message, **kwargs): return termcolor.colored(str(message), color="green",
                                                       attrs=[k for k, v in kwargs.items() if v is True])


def blue(message, **kwargs): return termcolor.colored(str(message), color="blue",
                                                      attrs=[k for k, v in kwargs.items() if v is True])


def cyan(message, **kwargs): return termcolor.colored(str(message), color="cyan",
                                                      attrs=[k for k, v in kwargs.items() if v is True])


def yellow(message, **kwargs): return termcolor.colored(str(message), color="yellow",
                                                        attrs=[k for k, v in kwargs.items() if v is True])


def magenta(message, **kwargs): return termcolor.colored(str(message), color="magenta",
                                                         attrs=[k for k, v in kwargs.items() if v is True])


def grey(message, **kwargs): return termcolor.colored(str(message), color="grey",
                                                      attrs=[k for k, v in kwargs.items() if v is True])


def get_time(sec):
    d = int(sec // (24 * 60 * 60))
    h = int(sec // (60 * 60) % 24)
    m = int((sec // 60) % 60)
    s = int(sec % 60)
    return d, h, m, s


def add_datetime(func):
    def wrapper(*args, **kwargs):
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(grey("[{}] ".format(datetime_str), bold=True), end="")
        return func(*args, **kwargs)

    return wrapper


def add_functionname(func):
    def wrapper(*args, **kwargs):
        print(grey("[{}] ".format(func.__name__), bold=True))
        return func(*args, **kwargs)

    return wrapper


def pre_post_actions(pre=None, post=None):
    def func_decorator(func):
        def wrapper(*args, **kwargs):
            if pre: pre()
            retval = func(*args, **kwargs)
            if post: post()
            return retval

        return wrapper

    return func_decorator


debug = ipdb.set_trace

def draw_pcd(pts3D,output_path=None,color=None):
    if type(pts3D)==torch.tensor:
        pts3D=pts3D.detach().cpu().numpy()
    if type(color)==torch.tensor:
        color=color.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3D)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(output_path, pcd)

class Log:
    def __init__(self):
        pass

    def process(self, pid):
        print(grey("Process ID: {}".format(pid), bold=True))

    def title(self, message):
        print(yellow(message, bold=True, underline=True))

    def info(self, message):
        print(magenta(message, bold=True))

    def options(self, opt, level=0):
        for key, value in sorted(opt.items()):
            if isinstance(value, (dict, edict)):
                print("   " * level + cyan("* ") + green(key) + ":")
                self.options(value, level + 1)
            else:
                print("   " * level + cyan("* ") + green(key) + ":", yellow(value))

    def loss_train(self, opt, ep, lr, loss, timer):
        if not opt.max_epoch: return
        message = grey("[train] ", bold=True)
        message += "epoch {}/{}".format(cyan(ep, bold=True), opt.max_epoch)
        message += ", lr:{}".format(yellow("{:.2e}".format(lr), bold=True))
        message += ", loss:{}".format(red("{:.3e}".format(loss), bold=True))
        message += ", time:{}".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.elapsed)), bold=True))
        message += " (ETA:{})".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.arrival))))
        print(message)

    def loss_val(self, opt, loss):
        message = grey("[val] ", bold=True)
        message += "loss:{}".format(red("{:.3e}".format(loss), bold=True))
        print(message)


log = Log()


def update_timer(opt, timer, ep, it_per_ep):
    if not opt.max_epoch: return
    momentum = 0.99
    timer.elapsed = time.time() - timer.start
    timer.it = timer.it_end - timer.it_start
    # compute speed with moving average
    timer.it_mean = timer.it_mean * momentum + timer.it * (1 - momentum) if timer.it_mean is not None else timer.it
    timer.arrival = timer.it_mean * it_per_ep * (opt.max_epoch - ep)


# move tensors to device in-place
def move_to_device(X, device):
    if isinstance(X, dict):
        for k, v in X.items():
            X[k] = move_to_device(v, device)
    elif isinstance(X, list):
        for i, e in enumerate(X):
            X[i] = move_to_device(e, device)
    elif isinstance(X, tuple) and hasattr(X, "_fields"):  # collections.namedtuple
        dd = X._asdict()
        dd = move_to_device(dd, device)
        return type(X)(**dd)
    elif isinstance(X, torch.Tensor):
        return X.to(device=device)
    return X


def to_dict(D, dict_type=dict):
    D = dict_type(D)
    for k, v in D.items():
        if isinstance(v, dict):
            D[k] = to_dict(v, dict_type)
    return D


def get_child_state_dict(state_dict, key):
    return {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("{}.".format(key))}


def restore_checkpoint(opt, model, load_name=None, resume=False):
    assert ((load_name is None) == (resume is not False))  # resume can be True/False or epoch numbers
    if resume:
        load_name = "{0}/model.ckpt".format(opt.output_path) if resume is True else \
            "{0}/model/{1}.ckpt".format(opt.output_path, resume)
    checkpoint = torch.load(load_name, map_location=opt.device)
    # load individual (possibly partial) children modules
    for name, child in model.graph.named_children():
        child_state_dict = get_child_state_dict(checkpoint["graph"], name)
        if child_state_dict:
            print("restoring {}...".format(name))
            child.load_state_dict(child_state_dict)  # 加载自己的特定epoch的参数
    for key in model.__dict__:
        if key.split("_")[0] in ["optim", "sched"] and key in checkpoint and resume:
            print("restoring {}...".format(key))
            getattr(model, key).load_state_dict(checkpoint[key])
    if resume:
        ep, it = checkpoint["epoch"], checkpoint["iter"]
        if resume is not True: assert (resume == (ep or it))
        print("resuming from epoch {0} (iteration {1})".format(ep, it))
    else:
        ep, it = None, None
    return ep, it
def restore_checkpoint_sfm(opt, model, load_name=None, resume=False):
    assert ((load_name is None) == (resume is not False))  # resume can be True/False or epoch numbers
    if resume:
        load_name = "{0}/model.ckpt".format(opt.output_path) if resume is True else \
            "{0}/model/{1}.ckpt".format(opt.output_path, resume)
    checkpoint = torch.load(load_name, map_location=opt.device)
    model.sdf_func.load_state_dict(checkpoint["sdf_func"],False)  # loading sdf function
    model.color_func.load_state_dict(checkpoint["color_func"])
    model.cam_info_reloaded=checkpoint["cam_info"]
    model.pts_info_reloaded = checkpoint["pts3d_info"]
    for key in model.__dict__:
        if key.split("_")[0] in ["optim", "sched"] and key in checkpoint and resume:
            print("restoring {}...".format(key))
            getattr(model, key).load_state_dict(checkpoint[key])
    if resume:
        ep, it = checkpoint["epoch"], checkpoint["iter"]
        if resume is not True: assert (resume == (ep or it))
        print("resuming from epoch {0} (iteration {1})".format(ep, it))
    else:
        ep, it = None, None
    return ep, it

def save_checkpoint(opt, model, ep, it, latest=False, children=None):
    os.makedirs("{0}/model".format(opt.output_path), exist_ok=True)
    if children is not None:
        graph_state_dict = {k: v for k, v in model.graph.state_dict().items() if k.startswith(children)}
    else:
        graph_state_dict = model.graph.state_dict()
    checkpoint = dict(
        epoch=ep,
        iter=it,
        graph=graph_state_dict,
    )
    for key in model.__dict__:
        if key.split("_")[0] in ["optim", "sched"]:
            checkpoint.update({key: getattr(model, key).state_dict()})
    torch.save(checkpoint, "{0}/model.ckpt".format(opt.output_path))
    if not latest:
        shutil.copy("{0}/model.ckpt".format(opt.output_path),
                    "{0}/model/{1}.ckpt".format(opt.output_path, ep or it))  # if ep is None, track it instead

def save_checkpoint_sfm(opt, model, ep, it, latest=False):
    os.makedirs("{0}/model".format(opt.output_path), exist_ok=True)
    sdf_func_state_dict = model.sdf_func.state_dict()
    color_func_state_dict = model.color_func.state_dict()
    cam_info=model.camera_set.get_all_parameters()
    pts3d_info=model.point_set.get_all_parameters()
    checkpoint = dict(
        epoch=ep,
        iter=it,
        sdf_func=sdf_func_state_dict,
        color_func=color_func_state_dict,
        cam_info=cam_info,
        pts3d_info=pts3d_info
    )
    for key in model.__dict__:
        if key.split("_")[0] in ["optim", "sched"]:
            checkpoint.update({key: getattr(model, key).state_dict()})
    torch.save(checkpoint, "{0}/model.ckpt".format(opt.output_path))
    if not latest:
        shutil.copy("{0}/model.ckpt".format(opt.output_path),
                    "{0}/model/{1}.ckpt".format(opt.output_path, ep or it))  # if ep is None, track it instead

def check_socket_open(hostname, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_open = False
    try:
        s.bind((hostname, port))
    except socket.error:
        is_open = True
    finally:
        s.close()
    return is_open


def get_layer_dims(layers):
    # return a list of tuples (k_in,k_out)
    return list(zip(layers[:-1], layers[1:]))


@contextlib.contextmanager
def suppress(stdout=False, stderr=False):
    with open(os.devnull, "w") as devnull:
        if stdout: old_stdout, sys.stdout = sys.stdout, devnull
        if stderr: old_stderr, sys.stderr = sys.stderr, devnull
        try:
            yield
        finally:
            if stdout: sys.stdout = old_stdout
            if stderr: sys.stderr = old_stderr


def colorcode_to_number(code):
    ords = [ord(c) for c in code[1:]]
    ords = [n - 48 if n < 58 else n - 87 for n in ords]
    rgb = (ords[0] * 16 + ords[1], ords[2] * 16 + ords[3], ords[4] * 16 + ords[5])
    return rgb


def intr2list(intrics):
    intrics = intrics.reshape(-1, 1).squeeze().detach().cpu().numpy().tolist()
    fx, fy, cx, cy = intrics[0], intrics[4], intrics[2], intrics[5]
    return [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def pose2list(pose):
    pose_list = pose.view(-1, 1).squeeze().detach().cpu().numpy().tolist()
    return pose_list + [0, 0, 0, 1]


import json


def dict2json(file_name, the_dict):
    '''
    将字典文件写如到json文件中
    :param file_name: 要写入的json文件名(需要有.json后缀),str类型
    :param the_dict: 要写入的数据，dict类型
    :return: 1代表写入成功,0代表写入失败
    '''
    try:
        json_str = json.dumps(the_dict, indent=4)
        with open(file_name, 'w') as json_file:
            json_file.write(json_str)
        return 1
    except:
        return 0
def convert_sigma_samples_to_ply(
        input_3d_sigma_array: np.ndarray,
        voxel_grid_origin,
        volume_size,
        ply_filename_out,
        log,
        level=5.0,
        offset=None,
        scale=None, ):
    """
    Convert sdf samples to .ply

    :param input_3d_sdf_array: a float array of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :volume_size: a list of three floats
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()
    verts, faces, normals, values = skimage.measure.marching_cubes(
        input_3d_sigma_array, level=level, spacing=volume_size
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    # mesh_points = np.matmul(mesh_points, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
    # mesh_points = np.matmul(mesh_points, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    log.info("saving mesh to %s" % str(ply_filename_out))
    ply_data.write(ply_filename_out)

    log.info(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
def extract_mesh(implicit_surface, log,volume_size=2.0, level=0.0, N=512, filepath='./surface.ply', show_progress=True,
                 chunk=16 * 1024,bound_max=None,bound_min=None,extra_info=None):

    s = volume_size
    voxel_grid_origin = [-s / 2., -s / 2., -s / 2.]
    volume_size = [s, s, s]
    if bound_max is not None:
        volume_size=[i-j for i,j in zip(bound_max,bound_min)]
        voxel_grid_origin=bound_min
    overall_index = np.arange(0, N ** 3, 1).astype(np.int)
    xyz = np.zeros([N ** 3, 3])

    # transform first 3 columns
    # to be the x, y, z index
    xyz[:, 2] = overall_index % N
    xyz[:, 1] = (overall_index / N) % N
    xyz[:, 0] = ((overall_index / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    xyz[:, 0] = (xyz[:, 0] * (s / (N - 1))) + voxel_grid_origin[2]
    xyz[:, 1] = (xyz[:, 1] * (s / (N - 1))) + voxel_grid_origin[1]
    xyz[:, 2] = (xyz[:, 2] * (s / (N - 1))) + voxel_grid_origin[0]
    def batchify(query_fn,inputs: torch.Tensor,extra_info=None,chunk=chunk):
        out = []
        for i in tqdm(range(0, inputs.shape[0], chunk), disable=not show_progress):
            if extra_info is not None:
                out_i = query_fn(torch.from_numpy(inputs[i:i + chunk]).float().cuda(),Base_Field=extra_info).data.cpu().numpy()
            else:
                out_i = query_fn(torch.from_numpy(inputs[i:i + chunk]).float().cuda()).data.cpu().numpy()
            out.append(out_i)
        out = np.concatenate(out, axis=0)
        return out
    if extra_info is not None:
        out = batchify(implicit_surface.infer_sdf, xyz,extra_info=extra_info)
    else:
        out = batchify(implicit_surface.infer_sdf, xyz)
    out = out.reshape([N, N, N])
    convert_sigma_samples_to_ply(out, voxel_grid_origin, [float(v) / N for v in volume_size], filepath, level=level,log=log)

def pad_omn2org(omnidata, img, mode="depth"):
    '''
    :param omnidata: [B,C,H,H]
    :param img: [B,C,H,W]
    :return:
    '''
    H_omn, W_omn = omnidata.shape[2], omnidata.shape[3]
    H_org, W_org = img.shape[2], img.shape[3]
    if mode == "depth":
        omnidata = torch_F.interpolate(omnidata, scale_factor=H_org / H_omn, mode="nearest")
    elif mode=="normal":
        omnidata = torch_F.interpolate(omnidata, scale_factor=H_org / H_omn, mode="bilinear")
    # padding          B C H W
    omnidata = torch_F.pad(omnidata,(int((W_org - H_org) / 2), W_org - H_org - int((W_org - H_org) / 2), 0, 0, 0, 0, 0, 0),value=-10.)
    return omnidata

from typing import Optional
# --------------------------geo util--------------------------------------
def get_idx3d_camset(cameraset,
                     cam_id,
                     mode="full"):
    pose_idx=[]
    kypts=[]
    pts_id=[]
    i=int(0)
    for id_i in cam_id:
        cam_i=cameraset(id_i)
        mask=(cam_i.idx2d_to_3d!=-1)
        pose_idx+=[i for j in range(mask.sum())]
        kypts.append(cam_i.kypts[mask])
        pts_id.append(cam_i.idx2d_to_3d[mask])
        i+=1
    return pts_id,pose_idx,kypts


