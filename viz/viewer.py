from collections import deque, OrderedDict
import easydict
import argparse
import os 
import numpy as np 
import copy 
import cv2
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
import matplotlib.cm as cm 

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
    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector([[0,1], [1,2], [2,3], [3,0], [4,0], [4,1], [4,2], [4,3]])
    )
    if w2c is not None:
        frustum.transform(np.linalg.inv(w2c))
        
    return frustum

class PipelineView:
    def __init__(self, 
                    filesdict, 
                    vfov=60, 
                    width=1024,
                    height=1024,
                    dest='teaser',
                    render=True,
                    **callbacks):
        self.vfov = vfov
        self.width = width
        self.height = height
        self.dest = dest
        os.makedirs(self.dest, exist_ok=True)
        
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window('Level S^2fM', self.width, self.height)
        self.window.set_on_layout(self.on_layout)
        self.window.set_on_close(callbacks['on_window_close'])  # callback
        self.pcd_material = o3d.visualization.rendering.MaterialRecord()
        self.pcd_material.shader = "defaultLit"
        self.pcd_material.base_roughness = 0.15
        self.pcd_material.base_reflectance = 0.72
        self.pcd_material.point_size = 2

        self.pcdview = gui.SceneWidget()
        self.window.add_child(self.pcdview)
        self.pcdview.enable_scene_caching(
            True)  # makes UI _much_ more responsive
        self.pcdview.scene = rendering.Open3DScene(self.window.renderer)
        self.pcdview.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, [0, -10, 0])
        self.pcdview.scene.set_background([1, 1, 1, 1])  # White background

        self.point_material = o3d.visualization.rendering.MaterialRecord()
        self.point_material.shader = "defaultLit"
        self.point_material.point_size = 2.0
        # self.point_material.base_color = [1.0, 0.00, 0.0, 1.0]

        self.cam_material = o3d.visualization.rendering.MaterialRecord()
        self.cam_material.shader = 'unlitLine'
        self.cam_material.line_width = 1.0
        self.cam_material.base_color = [0.0, 0.0, 0.0, 1.0]

        self.ccam_material = o3d.visualization.rendering.MaterialRecord()
        self.ccam_material.shader = 'unlitLine'
        self.ccam_material.line_width = 2.0
        self.ccam_material.base_color = [1.0, 0.0, 0.0, 1.0]
                                         
        # self.camera_view()  # Initially look from the camera
        em = self.window.theme.font_size

        # Options panel
        self.panel = gui.Vert(em, gui.Margins(em, em, em, em))
        self.panel.preferred_width = int(100 * self.window.scaling)
        self.window.add_child(self.panel)
        toggles = gui.Vert(em)
        self.panel.add_child(toggles)

        
        toggle_capture = gui.Button("Next View")
        toggle_capture.is_on = False
        toggle_capture.set_on_clicked(
            callbacks['on_toggle_capture'])  # callback
        toggles.add_child(toggle_capture)

        self.fit_view = gui.Button("Fit View")
        self.fit_view.is_on = False
        self.fit_view.set_on_clicked(
            callbacks['on_toggle_fitview'])
        toggles.add_child(self.fit_view)

        self.window.add_child(self.pcdview)
        self.filesdict = filesdict 

        self.flag_gui_init = False
        self.flag_exit = False

        self.max_pcd_vertices = 1000000

        self.queue = deque()
        for file_info in self.filesdict:
            self.queue.append(file_info)

        temp = self.load_pcd(self.filesdict[-1]['pcd'])
        frustums = self.load_cameras(self.filesdict[-1]['cam'])
        points = np.array(temp.points)
        # for fru in frustums:
            # points = np.concatenate((points,np.array(fru.frustum.points)),axis=0)
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(points))
        self.pcd_bounds = obb.get_axis_aligned_bounding_box()
        self.cnt = 0

        if not render:
            self.update(all=True)

    @staticmethod
    def load_mesh(mesh_file):
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        vertices = np.array(mesh.vertices)
        vertices_color = np.array([[192,192,192]]*vertices.shape[0])/255
        # mesh.vertex_color = o3d.utility.Vector3dVector(vertices_color)
        mesh.paint_uniform_color([0.6, 0.6, 0.6])
        
        return mesh
    
    @staticmethod
    def load_pcd(pcd_file):
        pcd = o3d.io.read_point_cloud(pcd_file)
        points = np.array(pcd.points)
        colors = np.linalg.norm(points,axis=1)
        colors = colors/np.max(colors)
        rgb = cm.rainbow(colors)[:,:3]
        # colors = colors/np.linalg.norm(colors,axis=1,keepdims=True)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        
        return pcd

    @staticmethod
    def load_cameras(cam_file):
        with open(cam_file) as f:
            cameras = json.load(f)

        cameras_dict = []
        for camkey, camval in cameras.items():
            K = np.array(camval['K']).reshape(4,4)[:3,:3]
            W2C = np.array(camval['W2C']).reshape(4,4)
            frustum = create_frustum(K, W2C)
            camera_obj = easydict.EasyDict(
                id=camkey,
                K=K,
                W2C=W2C,
                frustum=frustum,
                width = camval['img_size'][0],
                height = camval['img_size'][1]
            )
            cameras_dict.append(camera_obj)

        return cameras_dict


    def toggle_fitview(self, is_on):
        # if is_on:
            
        # print('toggle')
        info = self.queue[0]
        cameras = self.load_cameras(info['cam'])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=self.width,height=self.height, fx=cameras[0].K[0,0], fy=cameras[0].K[1,1], cx=cameras[0].K[0,2], cy=cameras[0].K[1,2])
        o3d_mat = np.array(
                [[0,1,0,0],
                [-1,0,0,0],
                [0,0,1,0],
                [0,0,0,1]
                ]
            )
        if is_on:
            self.current_extrinsic = self.pcdview.scene.camera.get_projection_matrix()
            
            
            self.pcdview.setup_camera(intrinsic, o3d_mat@cameras[-1].W2C, self.pcd_bounds)
 
    def update(self, all=None):
        if len(self.queue) == 0:
            return 
        info = self.queue[0] if not all else self.queue[-1]
        pcd = self.load_pcd(info['pcd'])
        cameras = self.load_cameras(info['cam'])
        
        print(len(self.queue))
        focal_length = 1000
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=self.width,height=self.height, fx=focal_length,fy=focal_length, cx=self.width//2, cy=self.height*0.75)
        o3d_mat = np.array(
            [[0,1,0,0],
             [-1,0,0,0],
             [0,0,1,0],
             [0,0,0,1]
            ]
        )
        RT = cameras[-1].W2C

        
        T_new = np.eye(4)
        T_new[:3,3] = [0,0,np.linalg.norm(RT[:3,3])*2]
        T_new[:3,:3] = Rotation.from_euler('xyz',[0,30,0],degrees=True).as_matrix()

        w2c = o3d_mat@T_new@RT
        
        # self.pcdview.setup_camera(intrinsic, o3d_mat@cameras[-1].W2C, self.pcd_bounds)
        self.pcdview.setup_camera(intrinsic, w2c, self.pcd_bounds)
        # self.pcdview.scene.camera.look_at([0, 0, 1.5], [0, 3, 1.5], [0, -1, 0])
        for i, camera in enumerate(cameras):
            if self.pcdview.scene.has_geometry(f'cam-{camera.id}'):
                self.pcdview.scene.remove_geometry(f'cam-{camera.id}')

                self.pcdview.scene.add_geometry(f'cam-{camera.id}', camera.frustum, self.cam_material)
            else:
                self.pcdview.scene.add_geometry(f'cam-{camera.id}', camera.frustum, self.ccam_material if not all else self.cam_material)


        if self.pcdview.scene.has_geometry('mesh'):
            self.pcdview.scene.remove_geometry('mesh')
        
        # if self.cnt % 5 == 0:
        mesh = self.load_mesh(info['mesh'])
        self.pcdview.scene.add_geometry('mesh', mesh,self.pcd_material)

        if self.pcdview.scene.has_geometry('pcd'):
            self.pcdview.scene.remove_geometry('pcd')
        pcd = self.load_pcd(info['pcd'])
        self.pcdview.scene.add_geometry('pcd', pcd, self.point_material)

        self.flag_gui_init = True


        def on_image(image):
            img = image
            quality = 85
            path = f'{self.dest}/{self.cnt:06d}.jpg'
            o3d.io.write_image(path, img, quality)
        
        self.pcdview.force_redraw()
        self.pcdview.scene.scene.render_to_image(on_image)
        
        self.queue.popleft()
        self.cnt+=1


    def on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        # """Callback on window initialize / resize"""
        frame = self.window.content_rect
        self.pcdview.frame = frame
        panel_size = self.panel.calc_preferred_size(layout_context,
                                                    self.panel.Constraints())
        self.panel.frame = gui.Rect(frame.get_right() - panel_size.width,
                                    frame.y, panel_size.width,
                                    panel_size.height)
