import trimesh
import argparse
import os 
import numpy as np 
import copy 
import cv2
import trimesh.viewer
from scipy.spatial.transform import Rotation
from PIL import Image


import os
import json
import time
import logging as log
import argparse
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


class PipelineView:
    def __init__(self, filesdict, vfov=60,  **callbacks):
        self.vfov = vfov
        
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window('Level S^2fM', 1280, 720)
        
        self.pcdview = gui.SceneWidget()
        self.pcdview.enable_scene_caching(
            True)  # makes UI _much_ more responsive
        self.pcdview.scene = rendering.Open3DScene(self.window.renderer)
        self.pcdview.scene.set_background([1, 1, 1, 1])  # White background
        self.pcdview.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, [0, -6, 0])
        # Point cloud bounds, depends on the sensor range
        self.pcd_bounds = o3d.geometry.AxisAlignedBoundingBox([-3, -3, 0],
                                                              [3, 3, 6])
        self.camera_view()  # Initially look from the camera

        self.window.add_child(self.pcdview)
        self.filesdict = filesdict 
# viewer = PipelineView()
        # gui.Application.instance.run()
        self.flag_gui_init = False
        self.flag_exit = False

    def update(self, frame_elements):
        if not self.flag_gui_init:
            # Set dummy point cloud to allocate graphics memory
            dummy_pcd = o3d.t.geometry.PointCloud({
                'positions':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32),
                'colors':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32),
                'normals':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32)
            })
            if self.pcdview.scene.has_geometry('pcd'):
                self.pcdview.scene.remove_geometry('pcd')

            # self.pcd_material.shader = "normals" if self.flag_normals else "defaultLit"
            self.pcdview.scene.add_geometry('pcd', dummy_pcd)
            self.flag_gui_init = True

            self.pcdview.force_redraw()

    def camera_view(self):
        """Callback to reset point cloud view to the camera"""
        self.pcdview.setup_camera(self.vfov, self.pcd_bounds, [0, 0, 0])
        # Look at [0, 0, 1] from camera placed at [0, 0, 0] with Y axis
        # pointing at [0, -1, 0]
        self.pcdview.scene.camera.look_at([0, 0, 1], [0, 0, 0], [0, -1, 0])

    def birds_eye_view(self):
        """Callback to reset point cloud view to birds eye (overhead) view"""
        self.pcdview.setup_camera(self.vfov, self.pcd_bounds, [0, 0, 0])
        self.pcdview.scene.camera.look_at([0, 0, 1.5], [0, 3, 1.5], [0, -1, 0])

    def on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        """Callback on window initialize / resize"""
        frame = self.window.content_rect
        self.pcdview.frame = frame
        panel_size = self.panel.calc_preferred_size(layout_context,
                                                    self.panel.Constraints())
        self.panel.frame = gui.Rect(frame.get_right() - panel_size.width,
                                    frame.y, panel_size.width,
                                    panel_size.height)


class PipelineController:
    def __init__(self, filesdict):
        self.view = PipelineView(filesdict)
        # self.view.window.set_on_layout(self.view.on_layout)
        # self.view.window.show()
        # self.view.window.run()

        # self.view.window.on_close = self.on_close
        gui.Application.instance.run()

    def update_view(self, frame_elements):
        """Updates view with new data. May be called from any thread.

        Args:
            frame_elements (dict): Display elements (point cloud and images)
                from the new frame to be shown.
        """
        gui.Application.instance.post_to_main_thread(
            self.pipeline_view.window,
            lambda: self.pipeline_view.update(frame_elements))
def parse_args():
    parser = argparse.ArgumentParser(description='Incremental Structure from Motion')
    parser.add_argument('--root', type=str, required=True, help='root directory of the data')

    opt = parser.parse_args()
    return opt

def read_camera_json(filename):
    import json
    with open(filename) as f:
        data = json.load(f)

    for k in data.keys():
        data[k]['K'] = np.array(data[k]['K']).reshape(4,4)[:3,:3]
        data[k]['W2C'] = np.array(data[k]['W2C']).reshape(4,4)
    
    return data

def create_frustum(K, w2c=None):
    near = 0.2 # near clipping plane
    far = 1000.0 # far clipping plane

    width = K[0,2]*2
    height = K[1,2]*2
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    x = np.array([0, width, width, 0])
    y = np.array([0, 0, height, height])
    z = np.array([1, 1, 1, 1]) # homogeneous coordinates
    x = (x - cx) * near / fx
    y = (y - cy) * near / fy
    z = z*near
    corners = np.stack([x, y, z]).T
    corners = np.concatenate((corners, np.array([[0,0,0]])), axis=0)
    
    line_set = corners[
        [[0,4],
        [1,4],
        [2,4],
        [3,4],
        [0,1],
        [1,2],
        [2,3],
        [3,0],]
    ]

    frustum = trimesh.load_path(line_set)
    if w2c is not None:
        frustum.apply_transform(np.linalg.inv(w2c))

    return frustum

def update_frustums(Scene, frustums):
    for j, frustum in enumerate(frustums):
        if f'cam_{j}' in Scene.geometry:
            Scene.geometry[f'cam_{j}'] = frustum
        else:
            color_frustum = copy.deepcopy(frustum)
            color_frustum.colors = [[255,0,0]]*4
            
            Scene.add_geometry({f'cam_{j}':color_frustum})

def main():
    opt = parse_args()

    # plyfiles = [f for f in os.listdir(opt.root) if f.endswith('.ply')]
    camfiles = [f for f in os.listdir(opt.root) if f.endswith('.json') and 'gt' not in f]
    camfiles.sort()

    view_ids = [(int(cam[3:-5]),cam[3:-5]) for cam in camfiles]

    filesdict = []
    
    for pair in view_ids:
        # filesdict[pair[0]] = pair[1]
        pcd_path = os.path.join(opt.root, f'{pair[0]}_pointcloud.ply')
        cam_path = os.path.join(opt.root, f'cam{pair[1]}.json')
        mesh_path = os.path.join(opt.root, f'{pair[1]}.ply')
        filesdict.append({'pcd': pcd_path, 'cam': cam_path, 'mesh': mesh_path})
    from tqdm import tqdm 
    # mesh_list = [trimesh.load(f['mesh']) for f in tqdm(filesdict.values())]
    mesh = trimesh.load(filesdict[-1]['mesh'])
    cameras = [read_camera_json(f['cam']) for f in tqdm(filesdict)]
    camera_orders = list(cameras[-1].keys())

    # viewer = PipelineView(filesdict)

    app = PipelineController(filesdict)
        
    

    

if __name__ == "__main__":
    main()