import matplotlib.pyplot as plt
import cv2
import numpy as np
import open3d as o3d
from ultralytics import SAM
from ultralytics import RTDETR
import copy
import time


def estimate_pose_for_mask(pts,color_img,mask,DOWN_SAMPLE_SIZE,ref_pcd):
    """
    Estimate the pose for an object in the point cloud.
    - pts: the point cloud with shape (height,width,coord) with type np.float32
    - color_img: the colors for the pts points with shape (height,width,3)
    - mask: mask for the object with shape (height,width) with type np.bool

    Returns a tuple:
        tvec - translation
        rvec - Rodrigues vector for rotation
        reg_p2p - the result object from Open3D (open3d.pipelines.registration.RegistrationResult)
    """
    objpts = pts[mask,:]
    color = color_img[mask,:]

    #remove NaNs (e.g. wrong depth pixels)
    verts = objpts.reshape((-1,3))
    idx = ~np.isnan(verts).any(axis=1)
    verts = verts[idx,:]
    color = color[idx,:]

    # create PCD
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(color/255)

    # downsample the PCD
    pcd_ds = pcd.voxel_down_sample(DOWN_SAMPLE_SIZE)

    # default transformation is around the mean of the object, with identity rotation
    pts_mean = np.mean(np.asarray(pcd_ds.points),axis=0)
    initial_transform = np.block([[np.identity(3), np.asmatrix(pts_mean).T],[0,0,0,1]])

    # run pose estimation with different outlier margins
    reg_p2p = o3d.pipelines.registration.registration_icp(
        ref_pcd, pcd_ds, 1, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))

    reg_p2p = o3d.pipelines.registration.registration_icp(
        ref_pcd, pcd_ds, 0.01, reg_p2p.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))

    # create rvec and tvec
    tvec = reg_p2p.transformation[0:3,3]
    rvec,_ = cv2.Rodrigues(reg_p2p.transformation[0:3,0:3])
    rvec = rvec.T[0,:]
    return tvec,rvec,reg_p2p

def estimate_pose(color_img, depth_img, Kdepth, det_model, cls_idxs, sam, DOWN_SAMPLE_SIZE, ref_pcd):
    # create point cloud
    pts = cv2.rgbd.depthTo3d(depth_img, Kdepth)
    ## detection
    results = det_model(color_img, classes = cls_idxs)
    det_result = results[0]
    if len(det_result.boxes) == 0:
        return []
    sam_result = sam.predict(color_img, bboxes = det_result.boxes.xyxy)[0]
    ## pose calculation
    results = [estimate_pose_for_mask(pts,color_img,mask.cpu().numpy(),DOWN_SAMPLE_SIZE,ref_pcd) for mask in sam_result.masks.data]
    ## Filter the results by fitness score. Fitness ranges from 0 to 1,
    #  and shows the inlier proportion. For an object, even 0.5 can be
    #  a successful match, since the object can be seen from one side.
    FITNESS_LIMIT = 0.4
    filtered_results = list(filter(lambda x: x[2].fitness > FITNESS_LIMIT, results))

    return results

def estimation_setup():
    det_model = RTDETR('rtdetr-x.pt')
    cls_idxs = [id for id,name in det_model.names.items() if name in ['bottle','cup']]
    sam = SAM('mobile_sam.pt')
    DOWN_SAMPLE_SIZE = 4e-3 # downsample the point clouds to 4 mm in order to speed up ICP process
    # ref_pcd = o3d.io.read_point_cloud("./asset/madara_bottle_simplified.pcd")
    ref_pcd = o3d.io.read_point_cloud("./utils_testing/madara_white_scaled_inverted.pcd")
    cnt = np.asarray(ref_pcd.points).shape[0]
    ref_pcd.colors = o3d.utility.Vector3dVector(np.repeat([[1,0,0]],cnt,axis = 0).astype(np.float32))
    ref_pcd = ref_pcd.voxel_down_sample(DOWN_SAMPLE_SIZE)
    return det_model,cls_idxs,sam,DOWN_SAMPLE_SIZE,ref_pcd


if __name__ == '__main__':
    # color_img = cv2.imread('./asset/example_image.png')
    # depth_img = cv2.imread('./asset/example_depth.png', cv2.IMREAD_ANYDEPTH)
    color_img = cv2.imread('./testing/color.png')
    depth_img = cv2.imread('./testing/depth.png', cv2.IMREAD_ANYDEPTH)
    # Kdepth = np.array([[637.22601318,   0.        , 644.54681396],
    #                     [  0.        , 636.76959229, 363.35079956],
    #                     [  0.        ,   0.        ,   1.        ]])
    Kdepth = np.array([[     612.82,           0.,      320.63],
                        [          0.,      612.95,      241.23],
                        [          0.,           0.,           1.]])

    det_model,cls_idxs,sam,DOWN_SAMPLE_SIZE,ref_pcd = estimation_setup()

    start_time = time.time()
    results = estimate_pose(color_img, depth_img, Kdepth, det_model, cls_idxs, sam, DOWN_SAMPLE_SIZE, ref_pcd)
    exec_time = time.time() - start_time

    print(results)
    print(f"Execution time: {exec_time:.9f} seconds")
