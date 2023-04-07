import trimesh
import argparse
import os 
import numpy as np 
import copy 
import cv2
import trimesh.viewer
from scipy.spatial.transform import Rotation
from PIL import Image
import open3d as o3d

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
    
    frustum = o3d.geometry.LineSet(o3d.utility.Vector3dVector(corners),o3d.utility.Vector2iVector([[0,4],[1,4],[2,4],[3,4],[0,1],[1,2],[2,3],[3,0]]))
    if w2c is not None:
        frustum.transform(np.linalg.inv(w2c))

    return frustum

def update_frustums(visualizer, frustums):
    for j, frustum in enumerate(frustums):
        visualizer.update_geometry(frustum)
        # if f'cam_{j}' in Scene.geometry:
        #     Scene.geometry[f'cam_{j}'] = frustum
        # else:
        #     color_frustum = copy.deepcopy(frustum)
        #     color_frustum.colors = [[255,0,0]]*4
            
        #     Scene.add_geometry({f'cam_{j}':color_frustum})

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
    # mesh = trimesh.load(filesdict[-1]['mesh'])
    mesh = o3d.io.read_triangle_mesh(filesdict[-1]['mesh'])
    mesh.compute_vertex_normals()
    
    cameras = [read_camera_json(f['cam']) for f in tqdm(filesdict)]
    camera_orders = list(cameras[-1].keys())

    # frustums_final = [create_frustum(cameras[0]['18']['K'], cameras[0]['18']['W2C'])
    frustums_all = [[create_frustum(values['K'],values['W2C']) for values in cams.values()] for cams in cameras]
        
    # Scene.show()
    # Scene = trimesh.Scene(geometry={'mesh':mesh})
    # viewer = Viewer(Scene, None,start_loop=False)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    
    for i, frustums in enumerate(frustums_all):
        for j, fru in enumerate(frustums):
            if j < i:
                vis.add_geometry(fru)
            else:
                vis.update_geometry(fru)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f'./outs/{i}.png')
        vis.run()
        # vis.update_geometry(frustums)
        # o3d.visualization.draw_geometries([mesh]+frustums)
        
        # for j, frustum in enumerate(frustums):
        #     if f'cam_{j}' in Scene.geometry:
        #         Scene.geometry[f'cam_{j}'] = frustum
        #     else:
        #         Scene.add_geometry({f'cam_{j}':frustum})
        # Scene.add_geometry({f'cam_{i}':frustum})
        # mesh = trimesh.load(filesdict[i]['pcd'])
        # w2c = cameras[-1][camera_orders[i]]['W2C']#[:3,:3]
        import pdb; pdb.set_trace()
    # Scene.camera.look_at(np.array([[0,0,0]]))
    rendered_images = []
    for i, frustums in enumerate(frustums_all):
        # for j, frustum in enumerate(frustums):
        #     if f'cam_{j}' in Scene.geometry:
        #         Scene.geometry[f'cam_{j}'] = frustum
        #     else:
        #         Scene.add_geometry({f'cam_{j}':frustum})
        # Scene.add_geometry({f'cam_{i}':frustum})
        mesh = trimesh.load(filesdict[i]['pcd'])
        w2c = cameras[-1][camera_orders[i]]['W2C']#[:3,:3]
        # viewer.scene.camera_transform = w2c
        
        # import pdb; pdb.set_trace()
        # Scene.camera_transform = w2c
        # w2c = np.linalg.inv(w2c)
        # rotation = w2c[:3,:3]
        # center = -rotation.T@w2c[:3,3].reshape(3)
        # matg = copy.deepcopy(Scene.camera_transform)
        # matg[:3,:3] = rotation.T
        # Scene.camera_transform = matg

        # euler = Rotation.from_matrix(rotation.T).as_euler('xyz', degrees=False)
        # euler = trimesh.transformations.euler_from_matrix(rotation.T, axes='rxyz')
        # Scene.set_camera(angles=euler)
        # Scene.show()
        # Scene.camera.look_at(w2c[:3,3].reshape(1,3),center=np.zeros((3)))
        
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # # center = center.reshape(3)
        # # rotation_4x4 = np.eye(4)
        # # rotation_4x4[:3,:3] = rotation
        # # Scene.camera.look_at(center)#, rotation=rotation_4x4)#,center=center[:,0])
        
        Scene.geometry['mesh'] = mesh
        # Scene.show()
        # import pdb; pdb.set_trace()
        update_frustums(Scene, frustums)
        Scene.apply_transform(np.linalg.inv(w2c))
        # Scene.show()
        
        png = Scene.save_image(resolution=(640,480))
        Scene.apply_transform(w2c)
        with open(f'outs/{i:06d}.png', 'wb') as f:
            f.write(png)
            f.close()
        rendered_images.append(Image.open(f'outs/{i:06d}.png'))
    rendered_images[0].save('out.gif', save_all=True, append_images=rendered_images[1:], optimize=False, duration=100, loop=0)
    
    
        
    

    

if __name__ == "__main__":
    main()