import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from controller import PipelineController

def parse_args():
    parser = argparse.ArgumentParser(description='Incremental Structure from Motion')
    parser.add_argument('--root', type=str, required=True, help='root directory of the data')
    parser.add_argument('--width', type=int, default=1024, help='width of the window')
    parser.add_argument('--height', type=int, default=512, help='height of the window')
    parser.add_argument('--output', type=str, default='teaser', help='output path')
    parser.add_argument('--skip-render', action='store_true', help='skip rendering')

    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()
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

    app = PipelineController(filesdict,width=1024,height=512,dest=opt.output, render=not opt.skip_render)
        
    

    

if __name__ == "__main__":
    main()