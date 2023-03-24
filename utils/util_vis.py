import pdb

import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import PIL
import imageio
from easydict import EasyDict as edict


import torch.optim as optim
from torch import autograd
import trimesh
from skimage.morphology import binary_dilation, disk

from . import camera

@torch.no_grad()
def tb_image(opt,tb,step,group,name,images,num_vis=None,from_range=(0,1),cmap="gray"):
    images = preprocess_vis_image(opt,images,from_range=from_range,cmap=cmap)
    num_H,num_W = num_vis or opt.tb.num_images
    images = images[:num_H*num_W]
    image_grid = torchvision.utils.make_grid(images[:,:3],nrow=num_W,pad_value=1.)
    if images.shape[1]==4:
        mask_grid = torchvision.utils.make_grid(images[:,3:],nrow=num_W,pad_value=1.)[:1]
        image_grid = torch.cat([image_grid,mask_grid],dim=0)
    tag = "{0}/{1}".format(group,name)
    tb.add_image(tag,image_grid,step)

def preprocess_vis_image(opt,images,from_range=(0,1),cmap="gray"):
    min,max = from_range
    images = (images-min)/(max-min)
    images = images.clamp(min=0,max=1).cpu()
    if images.shape[1]==1:
        images = get_heatmap(opt,images[:,0].cpu(),cmap=cmap)
    return images

def dump_images(opt,idx,name,images,masks=None,from_range=(0,1),cmap="gray"):
    images = preprocess_vis_image(opt,images,masks=masks,from_range=from_range,cmap=cmap) # [B,3,H,W]
    images = images.cpu().permute(0,2,3,1).numpy() # [B,H,W,3]
    for i,img in zip(idx,images):
        fname = "{}/dump/{}_{}.png".format(opt.output_path,i,name)
        img_uint8 = (img*255).astype(np.uint8)
        imageio.imsave(fname,img_uint8)

def get_heatmap(opt,gray,cmap): # [N,H,W]
    color = plt.get_cmap(cmap)(gray.numpy())
    color = torch.from_numpy(color[...,:3]).permute(0,3,1,2).float() # [N,3,H,W]
    return color

def color_border(images,colors,width=3):
    images_pad = []
    for i,image in enumerate(images):
        image_pad = torch.ones(3,image.shape[1]+width*2,image.shape[2]+width*2)*(colors[i,:,None,None]/255.0)
        image_pad[:,width:-width,width:-width] = image
        images_pad.append(image_pad)
    images_pad = torch.stack(images_pad,dim=0)
    return images_pad

@torch.no_grad()
def vis_cameras(opt,vis,step,poses=[],colors=["blue","magenta"],plot_dist=True):
    win_name = "{}/{}".format(opt.group,opt.name)
    data = []
    # set up plots
    centers = []
    for pose,color in zip(poses,colors):
        pose = pose.detach().cpu()
        vertices,faces,wireframe = get_camera_mesh(pose,depth=opt.visdom.cam_depth)
        center = vertices[:,-1]
        centers.append(center)
        # camera centers
        data.append(dict(
            type="scatter3d",
            x=[float(n) for n in center[:,0]],
            y=[float(n) for n in center[:,1]],
            z=[float(n) for n in center[:,2]],
            mode="markers",
            marker=dict(color=color,size=3),
        ))
        # colored camera mesh
        vertices_merged,faces_merged = merge_meshes(vertices,faces)
        data.append(dict(
            type="mesh3d",
            x=[float(n) for n in vertices_merged[:,0]],
            y=[float(n) for n in vertices_merged[:,1]],
            z=[float(n) for n in vertices_merged[:,2]],
            i=[int(n) for n in faces_merged[:,0]],
            j=[int(n) for n in faces_merged[:,1]],
            k=[int(n) for n in faces_merged[:,2]],
            flatshading=True,
            color=color,
            opacity=0.05,
        ))
        # camera wireframe
        wireframe_merged = merge_wireframes(wireframe)
        data.append(dict(
            type="scatter3d",
            x=wireframe_merged[0],
            y=wireframe_merged[1],
            z=wireframe_merged[2],
            mode="lines",
            line=dict(color=color,),
            opacity=0.3,
        ))
    if plot_dist:
        # distance between two poses (camera centers)
        center_merged = merge_centers(centers[:2])
        data.append(dict(
            type="scatter3d",
            x=center_merged[0],
            y=center_merged[1],
            z=center_merged[2],
            mode="lines",
            line=dict(color="red",width=4,),
        ))
        if len(centers)==4:
            center_merged = merge_centers(centers[2:4])
            data.append(dict(
                type="scatter3d",
                x=center_merged[0],
                y=center_merged[1],
                z=center_merged[2],
                mode="lines",
                line=dict(color="red",width=4,),
            ))
    # send data to visdom
    vis._send(dict(
        data=data,
        win="poses",
        eid=win_name,
        layout=dict(
            title="({})".format(step),
            autosize=True,
            margin=dict(l=30,r=30,b=30,t=30,),
            showlegend=False,
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            )
        ),
        opts=dict(title="{} poses ({})".format(win_name,step),),
    ))

def get_camera_mesh(pose,depth=1):
    vertices = torch.tensor([[-0.5,-0.5,1],
                             [0.5,-0.5,1],
                             [0.5,0.5,1],
                             [-0.5,0.5,1],
                             [0,0,0]])*depth
    faces = torch.tensor([[0,1,2],
                          [0,2,3],
                          [0,1,4],
                          [1,2,4],
                          [2,3,4],
                          [3,0,4]])
    vertices = camera.cam2world(vertices[None],pose)
    wireframe = vertices[:,[0,1,2,3,0,4,1,2,4,3]]
    return vertices,faces,wireframe

def merge_wireframes(wireframe):
    wireframe_merged = [[],[],[]]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:,0]]+[None]
        wireframe_merged[1] += [float(n) for n in w[:,1]]+[None]
        wireframe_merged[2] += [float(n) for n in w[:,2]]+[None]
    return wireframe_merged
def merge_meshes(vertices,faces):
    mesh_N,vertex_N = vertices.shape[:2]
    faces_merged = torch.cat([faces+i*vertex_N for i in range(mesh_N)],dim=0)
    vertices_merged = vertices.view(-1,vertices.shape[-1])
    return vertices_merged,faces_merged
def merge_centers(centers):
    center_merged = [[],[],[]]
    for c1,c2 in zip(*centers):
        center_merged[0] += [float(c1[0]),float(c2[0]),None]
        center_merged[1] += [float(c1[1]),float(c2[1]),None]
        center_merged[2] += [float(c1[2]),float(c2[2]),None]
    return center_merged

def plot_save_poses(opt,fig,pose,pose_ref=None,path=None,ep=None):
    # get the camera meshes
    _,_,cam = get_camera_mesh(pose,depth=opt.visdom.cam_depth)
    cam = cam.numpy()
    if pose_ref is not None:
        _,_,cam_ref = get_camera_mesh(pose_ref,depth=opt.visdom.cam_depth)
        cam_ref = cam_ref.numpy()
    # set up plot window(s)
    plt.title("epoch {}".format(ep))
    ax1 = fig.add_subplot(121,projection="3d")
    ax2 = fig.add_subplot(122,projection="3d")
    setup_3D_plot(ax1,elev=-90,azim=-90,lim=edict(x=(-1,1),y=(-1,1),z=(-1,1)))
    setup_3D_plot(ax2,elev=0,azim=-90,lim=edict(x=(-1,1),y=(-1,1),z=(-1,1)))
    ax1.set_title("forward-facing view",pad=0)
    ax2.set_title("top-down view",pad=0)
    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0,hspace=0)
    plt.margins(tight=True,x=0,y=0)
    # plot the cameras
    N = len(cam)
    color = plt.get_cmap("gist_rainbow")
    for i in range(N):
        if pose_ref is not None:
            ax1.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=(0.3,0.3,0.3),linewidth=1)
            ax2.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=(0.3,0.3,0.3),linewidth=1)
            ax1.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=(0.3,0.3,0.3),s=40)
            ax2.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=(0.3,0.3,0.3),s=40)
        c = np.array(color(float(i)/N))*0.8
        ax1.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=c)
        ax2.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=c)
        ax1.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=c,s=40)
        ax2.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=c,s=40)
    png_fname = "{}/{}.png".format(path,ep)
    plt.savefig(png_fname,dpi=75)
    # clean up
    plt.clf()

def plot_save_poses_blender(opt,fig,pose,pose_ref=None,path=None,ep=None):
    # get the camera meshes
    _,_,cam = get_camera_mesh(pose,depth=opt.visdom.cam_depth)
    cam = cam.numpy()
    if pose_ref is not None:
        _,_,cam_ref = get_camera_mesh(pose_ref,depth=opt.visdom.cam_depth)
        cam_ref = cam_ref.numpy()
    # set up plot window(s)
    ax = fig.add_subplot(111,projection="3d")
    ax.set_title("epoch {}".format(ep),pad=0)
    setup_3D_plot(ax,elev=45,azim=35,lim=edict(x=(-3,3),y=(-3,3),z=(-3,2.4)))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0,hspace=0)
    plt.margins(tight=True,x=0,y=0)
    # plot the cameras
    N = len(cam)
    ref_color = (0.7,0.2,0.7)
    pred_color = (0,0.6,0.7)
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam_ref],alpha=0.2,facecolor=ref_color))
    for i in range(N):
        ax.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=ref_color,linewidth=0.5)
        ax.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=ref_color,s=20)
    if ep==0:
        png_fname = "{}/GT.png".format(path)
        plt.savefig(png_fname,dpi=75)
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam],alpha=0.2,facecolor=pred_color))
    for i in range(N):
        ax.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=pred_color,linewidth=1)
        ax.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=pred_color,s=20)
    for i in range(N):
        ax.plot([cam[i,5,0],cam_ref[i,5,0]],
                [cam[i,5,1],cam_ref[i,5,1]],
                [cam[i,5,2],cam_ref[i,5,2]],color=(1,0,0),linewidth=3)
    png_fname = "{}/{}.png".format(path,ep)
    plt.savefig(png_fname,dpi=75)
    # clean up
    plt.clf()

def setup_3D_plot(ax,elev,azim,lim=None):
    ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.zaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.xaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.yaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.zaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    ax.set_xlabel("X",fontsize=16)
    ax.set_ylabel("Y",fontsize=16)
    ax.set_zlabel("Z",fontsize=16)
    ax.set_xlim(lim.x[0],lim.x[1])
    ax.set_ylim(lim.y[0],lim.y[1])
    ax.set_zlim(lim.z[0],lim.z[1])
    ax.view_init(elev=elev,azim=azim)
def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


# TODO Output masking yes or no
class Extractor3D(object):
    '''  Mesh extractor class for Occupancies

    The class contains functions for exctracting the meshes from a occupancy field

    Args:
        model (nn.Module): trained model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        refine_max_faces (int): max number of faces which are used as batch
            size for refinement process (we added this functionality in this
            work)
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1,
                 refine_max_faces=10000):
        if model is not None:
            self.model = model.to(device)
        else:
            self.model = None
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.padding = padding
        self.refine_max_faces = refine_max_faces

    def generate_mesh(self, data=None, return_stats=True, mask_loader=None):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        # inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        kwargs = {}

        # c = self.model.encode_inputs(inputs)
        mesh = self.generate_from_latent(None, stats_dict=stats_dict,
                                         data=None, mask_loader=mask_loader, **kwargs)

        return mesh, stats_dict

    def generate_from_latent(self, c=None, stats_dict={}, data=None,
                             mask_loader=None, **kwargs):
        ''' Generates mesh from latent.

        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 2 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,) * 3, (0.5,) * 3, (nx,) * 3
            )
            values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()

            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(self.device)
                # Normalize to bounding box
                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)
                # Evaluate model and update
                values = self.eval_points(
                    pointsf, c, **kwargs).cpu().numpy()

                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()
            if mask_loader is not None:
                pointsf2 = box_size * make_3d_grid((-0.5,) * 3, (0.5,) * 3, (value_grid.shape[0],) * 3)
                it = 0
                # for data in mask_loader:
                occ = filter_points(pointsf2, mask_loader) > 0.5
                value_grid[~occ.reshape(value_grid.shape)] = -30.0
                print("Masking Iteration: %03d" % it)
                it += 1
        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh = self.extract_mesh(value_grid, c, stats_dict=stats_dict)
        return mesh

    def eval_points(self, p, c=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.model(pi, None, return_logits=True, **kwargs).squeeze(-1)

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

    def extract_mesh(self, occ_hat, c=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 2 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
        vertices = box_size * (vertices - 0.5)

        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, c)
            stats_dict['time (normals)'] = time.time() - t0
        else:
            normals = None
        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, c=None):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c = c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decoder(vi, None, only_occupancy=True).squeeze(-1)
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c=None):
        ''' Refines the predicted mesh.

        Args:
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert (n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces)

        # detach c; otherwise graph needs to be retained
        # caused by new Pytorch version?
        # c = c.detach()

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-5)

        # Dataset
        ds_faces = TensorDataset(faces)
        dataloader = DataLoader(ds_faces, batch_size=self.refine_max_faces,
                                shuffle=True)

        # We updated the refinement algorithm to subsample faces; this is
        # usefull when using a high extraction resolution / when working on
        # small GPUs
        it_r = 0
        while it_r < self.refinement_step:
            for f_it in dataloader:
                f_it = f_it[0].to(self.device)
                optimizer.zero_grad()

                # Loss
                face_vertex = v[f_it]
                eps = np.random.dirichlet((0.5, 0.5, 0.5), size=f_it.shape[0])
                eps = torch.FloatTensor(eps).to(self.device)
                face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

                face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
                face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
                face_normal = torch.cross(face_v1, face_v2)
                face_normal = face_normal / \
                              (face_normal.norm(dim=1, keepdim=True) + 1e-10)

                face_value = torch.cat([
                    # torch.sigmoid(self.model.decode(p_split, c).logits)
                    self.model.decoder(p_split, None, only_occupancy=True).squeeze(-1)
                    for p_split in torch.split(
                        face_point.unsqueeze(0), 20000, dim=1)], dim=1)

                normal_target = -autograd.grad(
                    [face_value.sum()], [face_point], create_graph=True)[0]

                normal_target = \
                    normal_target / \
                    (normal_target.norm(dim=1, keepdim=True) + 1e-10)
                loss_target = (face_value - threshold).pow(2).mean()
                loss_normal = \
                    (face_normal - normal_target).pow(2).sum(dim=1).mean()

                loss = loss_target + 0.01 * loss_normal

                # Update
                loss.backward()
                optimizer.step()

                # Update it_r
                it_r += 1

                if it_r >= self.refinement_step:
                    break

        mesh.vertices = v.data.cpu().numpy()
        return mesh


def filter_points(p, mask_loader):
    # p = torch.from_numpy(p)
    p = p.cpu()
    n_p = p.shape[0]
    inside_mask = np.ones((n_p,), dtype=np.bool)
    inside_img = np.zeros((n_p,), dtype=np.bool)
    # for i in trange(n_images):
    # get data
    for data in mask_loader:
        datai = data
        maski_in = datai.get('img.mask')[0]

        # Apply binary dilation to account for errors in the mask
        maski = torch.from_numpy(binary_dilation(maski_in, disk(12))).float()

        # h, w = maski.shape
        h, w = maski.shape
        w_mat = datai.get('img.world_mat')
        c_mat = datai.get('img.camera_mat')
        s_mat = datai.get('img.scale_mat')

        # project points into image
        phom = torch.cat([p, torch.ones(n_p, 1)], dim=-1).transpose(1, 0)
        proj = c_mat @ w_mat @ s_mat @ phom
        proj = proj[0]
        proj = (proj[:2] / proj[-2].unsqueeze(0)).transpose(1, 0)

        # check which points are inside image; by our definition,
        # the coordinates have to be in [-1, 1]
        mask_p_inside = ((proj[:, 0] >= -1) &
                         (proj[:, 1] >= -1) &
                         (proj[:, 0] <= 1) &
                         (proj[:, 1] <= 1)
                         )
        inside_img |= mask_p_inside.cpu().numpy()

        # get image coordinates
        proj[:, 0] = (proj[:, 0] + 1) * (w - 1) / 2.
        proj[:, 1] = (proj[:, 1] + 1) * (h - 1) / 2.
        proj = proj.long()

        # fill occupancy values
        proj = proj[mask_p_inside]
        occ = torch.ones(n_p)
        occ[mask_p_inside] = maski[proj[:, 1], proj[:, 0]]
        inside_mask &= (occ.cpu().numpy() >= 0.5)

    occ_out = np.zeros((n_p,))
    occ_out[inside_img & inside_mask] = 1.
    return occ_out
import skimage
import plyfile
from tqdm import tqdm
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
def extract_mesh_occuNet(opt,var,graph,implicit_surface, log,volume_size=2.0, level=0.0, N=512, filepath='./surface.ply', show_progress=True,
                 chunk=16 * 1024):
    s = volume_size
    voxel_grid_origin = [-s / 2., -s / 2., -s / 2.]
    #voxel_grid_origin = [-s / 20., -s / 20., -s / 20.]
    volume_size = [s, s, s]

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

    def batchify(query_fn, inputs: torch.Tensor, chunk=chunk):
        out = []
        for i in tqdm(range(0, inputs.shape[0], chunk), disable=not show_progress):

            out_i = query_fn(opt,var,None, None, None,graph,torch.from_numpy(inputs[i:i + chunk]).float().cuda(),only_occupancy=True).data.cpu().numpy()
            out.append(out_i)
        out = np.concatenate(out, axis=0)
        return out

    out = batchify(implicit_surface.forward_samples, xyz)
    out = out.reshape([N, N, N])
    convert_sigma_samples_to_ply(out, voxel_grid_origin, [float(v) / N for v in volume_size], filepath, level=level,log=log)


import open3d as o3d
import numpy as np


def get_camera_frustum(img_size, K, W2C, frustum_length=0.5, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset

def visualize_cameras(colored_camera_dicts, sphere_radius, camera_size=0.1, geometry_file=None, geometry_type='mesh'):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw = [sphere, coord_frame]

    idx = 0
    for color, camera_dict in colored_camera_dicts:
        idx += 1

        cnt = 0
        frustums = []
        for img_name in sorted(camera_dict.keys()):
            K = np.array(camera_dict[img_name]['K']).reshape((4, 4))
            W2C = np.array(camera_dict[img_name]['W2C']).reshape((4, 4))
            C2W = np.linalg.inv(W2C)
            img_size = camera_dict[img_name]['img_size']
            frustums.append(get_camera_frustum(img_size, K, W2C, frustum_length=camera_size, color=color))
            cnt += 1
        cameras = frustums2lineset(frustums)
        things_to_draw.append(cameras)

    if geometry_file is not None:
        if geometry_type == 'mesh':
            geometry = o3d.io.read_triangle_mesh(geometry_file)
            geometry.compute_vertex_normals()
        elif geometry_type == 'pointcloud':
            geometry = o3d.io.read_point_cloud(geometry_file)
        else:
            raise Exception('Unknown geometry_type: ', geometry_type)

        things_to_draw.append(geometry)
    pdb.set_trace()
    o3d.visualization.draw_geometries(things_to_draw)


def visualize_camera_with_mesh(opt,camera_train_dict=None,mesh_path=None):
    root_dir=opt.output_path
    colored_camera_dicts = [([0, 1, 0], camera_train_dict)]
    sphere_radius=opt.rad
    camera_size = 0.1
    geometry_file = os.path.join(root_dir, mesh_path)
    geometry_type = 'mesh'
    visualize_cameras(colored_camera_dicts=colored_camera_dicts,sphere_radius=sphere_radius,camera_size=camera_size,geometry_file=geometry_file,geometry_type=geometry_type)

import cv2
def tensor2opencv(opt,tensor,H=None,W=None):
    if H is None:
        H,W=opt.H,opt.W
    if tensor.shape[-1]==3:
        type=="color"
    else:
        type == "gray"
    tensor = tensor.view(-1, H, W, tensor.shape[-1]).permute(0, 3, 1, 2)
    tensor = tensor[0].detach().permute(1, 2, 0).cpu().numpy()
    tensor *= 255
    if type=="color":
        tensor = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
    return tensor

def draw_matches(opt,img0,img1,pts0,pts1,vis_num=100,store_path=None,color=(255, 0, 0)):
    '''
    :param img0: B C H W
    :param img1: B C H W
    :param pts0: N 2
    :param pts1: N 2
    :return:
    '''
    img0_vis = img0
    img1_vis = img1
    img0_vis=img0_vis[0].detach().permute(1, 2, 0).cpu().numpy()
    img1_vis=img1_vis[0].detach().permute(1, 2, 0).cpu().numpy()
    img0_vis*= 255
    img1_vis *= 255
    img0_vis = cv2.cvtColor(img0_vis, cv2.COLOR_RGB2BGR)
    img1_vis = cv2.cvtColor(img1_vis, cv2.COLOR_RGB2BGR)
    img3 = np.concatenate([img0_vis, img1_vis], axis=1)
    if pts0 is None:
        if store_path is None:
            os.makedirs(f"{opt.output_path}/matches_vis", exist_ok=True)
            cv2.imwrite(f"{opt.output_path}/matches_vis/matche.jpg", img3)
        else:
            cv2.imwrite(store_path, img3)
        return 
    # vis the matches
    src_ky2d = pts1.detach().cpu().numpy().astype('int')
    kypts_2d_ref = pts0.detach().cpu().numpy().astype('int')
    select_inds=torch.randperm(src_ky2d.shape[0])[:vis_num]
    for ky0, ky1 in zip(kypts_2d_ref[select_inds], src_ky2d[select_inds]):
        ky1[0] += img1_vis.shape[1]
        cv2.line(img3, ky0, ky1, color, thickness=2)
        cv2.circle(img3,ky0,radius=2,color=(0, 0, 255),thickness=2)
        cv2.circle(img3, ky1, radius=2, color=(0, 0, 255), thickness=2)
    if store_path is None:
        os.makedirs(f"{opt.output_path}/matches_vis", exist_ok=True)
        cv2.imwrite(f"{opt.output_path}/matches_vis/matche.jpg", img3)
    else:
        cv2.imwrite(store_path, img3)
@torch.no_grad()
def generate_videos_synthesis(opt,):
    pose_pred, pose_GT = self.graph.get_pose(opt,var),var.poses_gt
    poses = pose_pred if opt.model == "GeoSDF_SfM" else pose_GT
    if opt.model == "GeoSDF_SfM" and opt.data.dataset == "llff":
        _, sim3 = self.prealign_cameras(opt, pose_pred, pose_GT)
        scale = sim3.s1 / sim3.s0
    else:
        scale = 1
    # rotate novel views around the "center" camera of all poses
    idx_center = (poses - poses.mean(dim=0, keepdim=True))[..., 3].norm(dim=-1).argmin()
    pose_novel = camera.get_novel_view_poses(opt, poses[idx_center], N=60, scale=scale).to(opt.device)
    # render the novel views
    novel_path = "{}/novel_view".format(opt.output_path)
    os.makedirs(novel_path, exist_ok=True)
    pose_novel_tqdm = tqdm.tqdm(pose_novel, desc="rendering novel views", leave=False)
    intr = edict(next(iter(self.test_loader))).intr[:1].to(opt.device)  # grab intrinsics

    for i, pose in enumerate(pose_novel_tqdm):
        rays_idx = self.graph.patch_sample(opt)
        rays_idx = rays_idx.reshape(1, -1).squeeze()
        pts_patch, sampled_pts, d_pts, sdf_surf,_ = self.graph.render_depth(opt, pose.unsqueeze(0), var, rays_idx=rays_idx,
                                                                    ref_intr=var.ref_intr, src_intrs=var.src_intrs,
                                                                    mode="full")
        normal_vis=self.graph.RenMchFusion.gradient(pts_patch)
        depth = d_pts.view(1, opt.data.image_size[0], opt.data.image_size[1], -1)
        normal_vis=normal_vis.view(1, opt.data.image_size[0], opt.data.image_size[1], -1)
        depth = depth.permute(0, 3, 1, 2)  # B 1 H W
        normal_vis=normal_vis.permute(0, 3, 1, 2).abs()
        normal_vis/=torch.norm(normal_vis,dim=1,keepdim=True)
        normal_vis/=1.2
        inv_depth=1/depth
        torchvision_F.to_pil_image(inv_depth.cpu()[0]).save("{}/depth_{}.png".format(novel_path, i))
        torchvision_F.to_pil_image(normal_vis.cpu()[0]).save("{}/normal_{}.png".format(novel_path, i))
    # write videos
    print("writing videos...")
    rgb_vid_fname = "{}/novel_view_rgb.mp4".format(opt.output_path)
    depth_vid_fname = "{}/novel_view_depth.mp4".format(opt.output_path)
    normal_vid_fname = "{}/novel_view_normal.mp4".format(opt.output_path)
    depth_vid_final_fname = "{}/novel_view_depth_final.mp4".format(opt.output_path)
    normal_vid_final_fname = "{}/novel_view_normal_final.mp4".format(opt.output_path)
    # os.system(
    #     "ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(novel_path,
    #                                                                                             rgb_vid_fname))
    img_tem=cv2.imread(f"{novel_path}/depth_0.png")
    imgInfo=img_tem.shape
    size = (imgInfo[1], imgInfo[0])
    f = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    videoWrite = cv2.VideoWriter(depth_vid_fname, f, 30, size)     #  文件名     编码器   帧率   图片大小
    for i in range(1, 60):
        fileName = "depth_" + str(i) + ".png"
        img_path=f"{novel_path}/{fileName}"
        img = cv2.imread(img_path)
        videoWrite.write(img)
    print("end!")

    normal_tem = cv2.imread(f"{novel_path}/normal_0.png")
    normalInfo = normal_tem.shape
    size = (normalInfo[1], normalInfo[0])
    f_norm = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    videoWrite_norm = cv2.VideoWriter(normal_vid_fname, f_norm, 30, size)  # 文件名     编码器   帧率   图片大小
    for i in range(1, 60):
        fileName = "normal_" + str(i) + ".png"
        img_path = f"{novel_path}/{fileName}"
        img = cv2.imread(img_path)
        videoWrite_norm.write(img)
    print("end!")

    for i, pose in enumerate(pose_novel_tqdm):
        pose_vis = pose.unsqueeze(0)
        depth_vis, normal_vis, rbg_vis = self.graph.render_slices(opt, pose_vis, self.train_data.all, self.train_data.all.intr[:1])
        depth_vis = depth_vis.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)
        inv_depth_vis = 1 / depth_vis
        normal_vis = normal_vis.view(-1, opt.H, opt.W, 3).permute(0, 3, 1, 2).abs()
        normal_vis /= torch.norm(normal_vis, dim=1, keepdim=True)
        normal_vis /= 1.2
        rbg_vis = rbg_vis.view(-1, opt.H, opt.W, 3).permute(0, 3, 1, 2)
        torchvision_F.to_pil_image(inv_depth_vis.cpu()[0]).save("{}/depth_final{}.png".format(novel_path, i))
        torchvision_F.to_pil_image(normal_vis.cpu()[0]).save("{}/normal_final{}.png".format(novel_path, i))
        torchvision_F.to_pil_image(rbg_vis.cpu()[0]).save("{}/rgb_final{}.png".format(novel_path, i))

    img_tem = cv2.imread(f"{novel_path}/rgb_final0.png")
    imgInfo = img_tem.shape
    size = (imgInfo[1], imgInfo[0])
    f = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    videoWrite = cv2.VideoWriter(rgb_vid_fname, f, 30, size)  # 文件名     编码器   帧率   图片大小
    for i in range(1, 60):
        fileName = "rgb_final" + str(i) + ".png"
        img_path = f"{novel_path}/{fileName}"
        img = cv2.imread(img_path)
        videoWrite.write(img)
    print("end!")

    img_tem = cv2.imread(f"{novel_path}/normal_final0.png")
    imgInfo = img_tem.shape
    size = (imgInfo[1], imgInfo[0])
    f = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    videoWrite = cv2.VideoWriter(normal_vid_final_fname, f, 30, size)  # 文件名     编码器   帧率   图片大小
    for i in range(1, 60):
        fileName = "normal_final" + str(i) + ".png"
        img_path = f"{novel_path}/{fileName}"
        img = cv2.imread(img_path)
        videoWrite.write(img)
    print("end!")

    img_tem = cv2.imread(f"{novel_path}/depth_final0.png")
    imgInfo = img_tem.shape
    size = (imgInfo[1], imgInfo[0])
    f = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    videoWrite = cv2.VideoWriter(depth_vid_final_fname, f, 30, size)  # 文件名     编码器   帧率   图片大小
    for i in range(1, 60):
        fileName = "depth_final" + str(i) + ".png"
        img_path = f"{novel_path}/{fileName}"
        img = cv2.imread(img_path)
        videoWrite.write(img)
    print("end!")

    os.system(
        "ffmpeg -y -framerate 30 -i {0}/depth_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(novel_path,
                                                                                                          depth_vid_fname))
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

def vis_by_wis3d_pcd(wis3d,pcd_path,name):
    wis3d.add_point_cloud(pcd_path, name=name)
    pcd = trimesh.load_mesh(pcd_path)
    wis3d.add_point_cloud(pcd.vertices, pcd.colors, name=name)

def vis_by_wis3d_mesh(wis3d,mesh_path,name):
    wis3d.add_point_cloud(mesh_path, name=name)
    mesh = trimesh.load_mesh(mesh_path)
    wis3d.add_mesh(mesh.vertices, mesh.faces,
                   mesh.visual.vertex_colors[:, :3], name=name)