import cv2
import numpy as np
import open3d as o3d
from ultralytics import SAM
from ultralytics import RTDETR
import yaml
import os


class Estimator:
    def __init__(self):
        self.det_model = RTDETR('rtdetr-x.pt')
        self.cls_idxs = [id for id,name in self.det_model.names.items() if name in ['bottle','cup']]
        self.sam = SAM('mobile_sam.pt')
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = curr_dir + "/../config/estimator.yml"
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        ref_pdc_path = config["ref_pcd_path"] # must be absolute path
        self.down_smaple_size = config["down_sample_size"]
        ref_pcd = o3d.io.read_point_cloud(ref_pdc_path)
        cnt = np.asarray(ref_pcd.points).shape[0]
        ref_pcd.colors = o3d.utility.Vector3dVector(np.repeat([[1,0,0]],cnt,axis = 0).astype(np.float32))
        self.ref_pcd = ref_pcd.voxel_down_sample(self.down_smaple_size)
        self.fitness_limit = config["fitness_limit"]

    def estimate(self, color_img, depth_img, camera_matrix, do_viz):
        pts = cv2.rgbd.depthTo3d(depth_img, camera_matrix)
        results = self.det_model(color_img, classes = self.cls_idxs)
        det_result = results[0]
        if len(det_result.boxes) == 0:
            return []
        sam_result = self.sam.predict(color_img, bboxes = det_result.boxes.xyxy)[0]
        if do_viz:
            sam_rez_img = np.array(sam_result.plot())
            cv2.imshow("SAM result", sam_rez_img)
            cv2.waitKey()
        results = [self.estimate_pose_for_mask(pts, color_img, mask.cpu().numpy()) for mask in sam_result.masks.data]
        ## Filter the results by fitness score. Fitness ranges from 0 to 1,
        #  and shows the inlier proportion. For an object, even 0.5 can be
        #  a successful match, since the object can be seen from one side.
        filtered_results = list(filter(lambda x: x[2].fitness > self.fitness_limit, results))

        return results

    def estimate_pose_for_mask(self, pts, color_img, mask):
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
        pcd_ds = pcd.voxel_down_sample(self.down_smaple_size)

        # default transformation is around the mean of the object, with identity rotation
        pts_mean = np.mean(np.asarray(pcd_ds.points),axis=0)
        initial_transform = np.block([[np.identity(3), np.asmatrix(pts_mean).T],[0,0,0,1]])

        # run pose estimation with different outlier margins
        reg_p2p = o3d.pipelines.registration.registration_icp(
            self.ref_pcd, pcd_ds, 1, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))

        reg_p2p = o3d.pipelines.registration.registration_icp(
            self.ref_pcd, pcd_ds, 0.01, reg_p2p.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))

        # create rvec and tvec
        tvec = reg_p2p.transformation[0:3,3]
        rvec,_ = cv2.Rodrigues(reg_p2p.transformation[0:3,0:3])
        rvec = rvec.T[0,:]
        return tvec,rvec,reg_p2p
