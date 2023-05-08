import pdb

import tqdm

# import sfm_cal
import sfm_utils
import argparse
from typing import List, Tuple, Dict
from pathlib import Path
import numpy as np
import os
from database import COLMAPDatabase, blob_to_array, pair_id_to_image_ids
import pycolmap
from read_write_model import read_points3D_binary


def read_matches_from_db(database_path: Path
                         ) -> Tuple[List[Tuple[str]], List[np.ndarray]]:
    db = COLMAPDatabase.connect(database_path)
    id2name = db.image_id_to_name()
    desc = {}
    for image_id, r, c, data in db.execute("SELECT * FROM descriptors"):
        d = blob_to_array(data, np.uint8, (-1, c))
        desc[image_id] = d / np.linalg.norm(d, axis=1, keepdims=True)
    # only compute scores if descriptors are in database
    compute_scores = (len(desc) > 0)
    scores = [] if compute_scores else None
    pairs = []
    matches = []
    for pair_id, data in db.execute("SELECT pair_id, data FROM matches"):
        id1, id2 = pair_id_to_image_ids(pair_id)
        name1, name2 = id2name[id1], id2name[id2]
        if data is None:
            continue
        pairs.append((name1, name2))
        match = blob_to_array(data, np.uint32, (-1, 2))
        matches.append(match)
        if compute_scores:
            d1, d2 = desc[id1][match[:, 0]], desc[id2][match[:, 1]]
            scores.append(np.einsum('nd,nd->n', d1, d2))
    db.close()
    return pairs, matches, scores


def read_image_id_to_name_from_db(database_path: Path) -> Dict[int, str]:
    db = COLMAPDatabase.connect(database_path)
    id2name = db.image_id_to_name()
    db.close()
    return id2name


def read_keypoints_from_db(database_path: Path, as_cpp_map: bool = True,
                           ) -> Dict[str, np.ndarray]:
    keypoints_dict = {}
    db = COLMAPDatabase.connect(str(database_path))
    id2name = db.image_id_to_name()
    for image_id, rows, cols, data in db.execute("SELECT * FROM keypoints"):
        keypoints = blob_to_array(data, np.float32, (rows, cols))
        keypoints = keypoints.astype(np.float64)[:, :2]  # keep only xy
        keypoints_dict[id2name[image_id]] = keypoints
    db.close()
    return keypoints_dict


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ETH3D')
    parser.add_argument('--scene', type=str, default='room')
    parser.add_argument('--dataroot', type=str, default='./data')
    parser.add_argument('--clean', action='store_true', default=False)
    parser.add_argument('--ignore-intrinsic', action='store_true', default=False)


    args = parser.parse_args()

    working_path = f"{args.dataroot}/{args.data}/{args.scene}"

    if args.clean and os.path.exists(f"{working_path}/database.db"):
        os.remove(f"{working_path}/database.db")

    # reading the K matrix from intrinsics.txt
    if not os.path.exists(f"{working_path}/intrinsics.txt"):
        K = None
    else:
        with open(f"{working_path}/intrinsics.txt", 'r') as f:
            K = np.array([[float(x) for x in line.split()] for line in f])
    
    # run feature_extractor
    # if K is None or args.ignore_intrinsic:
    cmd = f'colmap feature_extractor --database_path {working_path}/database.db --image_path {working_path}/images --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_PINHOLE'
    # else:
        # cmd = f'colmap feature_extractor --database_path {working_path}/database.db --image_path {working_path}/images --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE --ImageReader.camera_params {K[0,0]},{K[1,1]},{K[0,2]},{K[1,2]}'
    
    os.system(cmd)
    
    # run exaustive_matcher
    cmd = f'colmap exhaustive_matcher --database_path {working_path}/database.db'
    os.system(cmd)

    keypoints_dict = read_keypoints_from_db(f"{working_path}/database.db")
    pairs, matches, scores = read_matches_from_db(f"{working_path}/database.db")
    db = COLMAPDatabase.connect(f"{working_path}/database.db")

    os.makedirs(f"{working_path}/sparse", exist_ok=True)
    # run mapper to get the view order
    cmd = f'colmap mapper --database_path {working_path}/database.db --image_path {working_path}/images --output_path {working_path}/sparse > {working_path}/log.txt'
    os.system(cmd)

    K=None
    for camera_id, model, width, height,params,f in db.execute("SELECT  camera_id, model, width, height,params,prior_focal_length FROM cameras"):
        f,cx,cy=blob_to_array(params,np.float64)
        K=np.array([[f,0,cx],[0,f,cy],[0,0,1]])

    if args.data == 'blender':
        img_files=sorted(os.listdir(f"{working_path}/images"),key=lambda x:int((str(x).split('.')[0]).split("_")[-1]))
    elif args.data in ['TanksAndTemple', 'BlendedMVS']:
        img_files=sorted(os.listdir(f"{working_path}/images"),key=lambda x:int((str(x).split('.')[0])))
    else:
        img_files=sorted(os.listdir(f"{working_path}/images"),key=lambda x:int((str(x).split('.')[0]).split("_")[-1]))
    
    
    pairs_full = []
    matches_full = []

    pose_graph_images = f'{working_path}/sparse/0'

    reconstruction = pycolmap.Reconstruction(pose_graph_images)
    img_dict = {}
    
    os.makedirs(f'{working_path}/pose', exist_ok=True)

    for fname in img_files:
        with open(f'{working_path}/pose/{fname[:-4]}.txt', 'w') as f:
            G = np.eye(4)
            # write G to f
            for i in range(4):
                for j in range(4):
                    f.write(str(G[i,j]) + ' ')
                f.write('\n')

    for image_id, image in reconstruction.images.items():
        img_dict[image_id] = image.name

    
    init="Initializing with image pair"
    reg="Registering image #"
    sequence=[]
    import re
    with open(f"{working_path}/log.txt","r") as f:
        lines=f.readlines()
        init_indx=[]
        for (l,indx) in zip(lines,range(len(lines))):
            print(l)
            if init in l:
                sequence=[]
                img_id0,img_id1=re.findall("\d+",l)
                sequence.append(img_id0)
                sequence.append(img_id1)
            elif reg in l:
                img_idx=re.findall("\d+",l)[0]
                ind_sequence=int(re.findall("\d+",l)[1])
                if ind_sequence-1==len(sequence):
                    sequence.append(img_idx)
                else:
                    # overlapping the frame that could not be registered
                    sequence[ind_sequence-1]=img_idx
                print(ind_sequence)

    while len(sequence)>len(img_dict):
        sequence.pop()
    
    img_name_sequenc=[img_dict[int(i)] for i in sequence]
    pose_graph=[img_files.index(n_i) for n_i in img_name_sequenc]
    np.save(f"{working_path}/pose_graph.npy", pose_graph)

    for idx_i in range(len(img_files)):
        ref_img_name = img_files[idx_i]
        print(ref_img_name)
        for idx_ij in range(len(img_files)-idx_i-1):
            src_img_name = img_files[idx_i+idx_ij+1]
            pairs_full.append((ref_img_name, src_img_name))
            inverse = False
            try:
                index_find = pairs.index((ref_img_name, src_img_name))
            except:
                try: 
                    index_find = pairs.index((src_img_name, ref_img_name))
                    inverse = True
                except:
                    index_find = -1
            print(index_find)
            if (index_find == -1) | (len(matches[index_find]) < 50):
                matches_full.append(np.uint32(np.array([])))
            else:
                if inverse == False:
                    matches_full.append(matches[index_find])
                else:
                    matches_full.append(matches[index_find][:,[1,0]])
                    # matches_full.append(np.concatenate([matches[indx_find][:,1:],matches[indx_find][:,:1]],axis=-1))
    two_view_dict = dict()
    mask_inlier = []
    for img_pair,match in zip(pairs_full,matches_full):
        print(img_pair)
        left_img_name,right_img_name=img_pair[0],img_pair[1]
        if len(match)!=0:
            ky_pts0,ky_pts1=keypoints_dict[left_img_name],keypoints_dict[right_img_name]
            pair=(ky_pts0[match[:,0]],ky_pts1[match[:,1]])
            fx, cx, cy = K[0, 0], K[0, 2], K[1, 2]
            camera_colmap = pycolmap.Camera(model='SIMPLE_PINHOLE', width=int(cx*2),height=int(cy*2), params=[fx, cx, cy], )
            answer = pycolmap.essential_matrix_estimation(pair[0],pair[1], camera_colmap, camera_colmap)
            pose1,pose2,mask=sfm_utils.recover_pose_(pair[0],pair[1],K)
            mask=np.array( answer["inliers"])
            print(f"ratio:{mask.sum()/len(mask)}")
            pts_triang=sfm_utils.trangle_3Dpts(pose1,pose2,K,K,pair[0],pair[1])
            pts_triang_mask=pts_triang[:,mask].transpose(1,0)
            content_dict={"paired_kypts":pair,"pose1":pose1,"pose2":pose2,"inlier_msk":mask,"intr":K}
            tem_dict = {img_pair: content_dict}
            mask_inlier.append(mask)
            two_view_dict.update(tem_dict)
        else:
            content_dict={"paired_kypts":[],"pose1":[],"pose2":[],"inlier_msk":[],"intr":K}
            tem_dict = {img_pair: content_dict}
            if len(mask_inlier)!=0:
                mask_inlier.append(np.zeros_like(mask_inlier[-1]))
            else:
                mask_inlier.append(np.array([0,0,0]))
            two_view_dict.update(tem_dict)


    mask_inlier = np.array(mask_inlier)
    N_views_list = []
    iter_ind=0
    for ky_name in img_files:
        other_indx=[j for j in range(len(keypoints_dict.keys())) if j!=iter_ind]
        n=len(keypoints_dict.keys())
        tem_indx=[int((2*n-iter_ind-1)*iter_ind/2+j-iter_ind-1) if j>iter_ind else int((2*n-j-1)*j/2+iter_ind-j-1) for j in other_indx]
        print(tem_indx)
        mask_=mask_inlier[tem_indx]
        matches_full=np.array(matches_full)
        matches_new=[matches_full[tem_indx[i]] if (i>=iter_ind)|(len(matches_full[tem_indx[i]])==0) else np.concatenate([matches_full[tem_indx[i]][:,1:],matches_full[tem_indx[i]][:,:1]],axis=-1) for i in range(len(tem_indx))]
        tem_dict=dict({"kypts":keypoints_dict[ky_name],"indxes": matches_new,"mask":mask_})
        N_views_list.append(tem_dict)
        iter_ind+=1

    np.save(f"{working_path}/two_view.npy", two_view_dict)
    np.save(f"{working_path}/n_views.npy", N_views_list)



if __name__ == "__main__":
    main()