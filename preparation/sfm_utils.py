import os,sys,pdb
import cv2
import torch
import numpy as np

def sift_matches(img1,img2,detector='sift'):
    if detector.startswith('si'):
        print("sift detector......")
        sift = cv2.SIFT_create()
    else:
        print("surf detector......")
        sift = cv2.xfeatures2d.SURF_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    # get matched 2D pairs
    pts1=[]
    pts2=[]
    for i in range(len(good)):
        pts1.append(kp1[good[i].queryIdx].pt)
        pts2.append(kp2[good[i].trainIdx].pt)

    # print("the number of good matches is {}".format(len(good)))
    # cv2.drawMatchesKnn expects list of lists as matches.
    print(
        "the number of matched pairs is {}".format(len(pts1))
    )
    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    # cv2.imwrite("./matched.jpg",img3)
    return [np.array(pts1),np.array(pts2)]

def recover_pose(model,pts1,pts2,K):
    E,mask=cv2.findEssentialMat(pts1,pts2,K,method=cv2.RANSAC,prob=0.8,threshold=3,maxIters=5)  # threshold is for the distance between point and epipolar
    _,R,T,mask=cv2.recoverPose(E,pts1,pts2,K,mask=mask)
    print("the inliers ratio is {:.2f}".format(mask.sum() / len(mask)))
    model.R2=R
    model.T2=T
    model.R1=np.eye(3)
    model.T1 = np.zeros_like(T)
    model.pose1=np.concatenate((model.R1,model.T1),axis=-1)
    model.pose2 = np.concatenate((model.R2, model.T2), axis=-1)
    mask=mask[:,0].astype(np.bool)
    model.pairs=[np.array(pts1[mask]),np.array(pts2[mask])]
    return R,T,mask


def recover_pose_(pts1,pts2,K):
    E,mask=cv2.findEssentialMat(pts1,pts2,K,method=cv2.RANSAC,prob=0.7,threshold=20,maxIters=100)  # threshold is for the distance between point and epipolar
    _,R,T,mask=cv2.recoverPose(E,pts1,pts2,K,mask=mask)
    print("---------------------------------------------")
    print("the inliers ratio is {:.2f}".format(mask.sum() / len(mask)))
    print("the num of the matches is {}".format(len(mask)))
    print("---------------------------------------------")
    # if (mask.sum() / len(mask))<0.10:
    #     pdb.set_trace()
    R2=R
    T2=T
    R1=np.eye(3)
    T1 = np.zeros_like(T)
    pose1=np.concatenate((R1,T1),axis=-1)
    pose2 = np.concatenate((R2, T2), axis=-1)
    mask=mask[:,0].astype(np.bool)
    #pairs=[np.array(pts1[mask]),np.array(pts2[mask])]
    return pose1,pose2,mask

def trangle_3Dpts(pose1,pose2,K1,K2,pts1,pts2):
    P1=np.matmul(K1,pose1)
    P2=np.matmul(K2,pose2)
    pts3D=cv2.triangulatePoints(P1,P2,pts1.transpose(),pts2.transpose())
    pts3D_out=pts3D[:3,:]/pts3D[3:4,:]
    return pts3D_out



# img1_path=r"../datasets/ETH3D/DSC_0259.JPG"
# img2_path=r"../datasets/ETH3D/DSC_0260.JPG"
# #3410.68 3409.64 3115.69 2063.73
# K=np.array([[3410.68,0,3115.69],[0,3409.64,2063.73],[0,0,1]])
# img1=cv2.imread(img1_path)
# img2=cv2.imread(img2_path)
# pairs=sift_matches(img1,img2)
# R,T=recover_pose(pairs[0],pairs[1],K)
# Points_3D=trangle_3Dpts(np.concatenate((np.ones_like(R),np.zeros_like(T)),axis=-1),np.concatenate((R,T),axis=-1),K,K,pairs[0],pairs[1])