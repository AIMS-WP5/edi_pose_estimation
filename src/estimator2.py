import os

import cv2
import numpy as np
import yaml

from rotation_seg import SAMSegmentEstimator


class Estimator:
    """Wrapper that uses SAMSegmentEstimator (rotation_seg_fill) for pose estimation."""

    def __init__(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(curr_dir, "../config/estimator.yml")
        cfg_yaml = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as config_file:
                cfg_yaml = yaml.safe_load(config_file) or {}

        # Build config for SAMSegmentEstimator with sensible defaults for the bag.
        self.estimator_cfg = {
            "seg_model_type": "yolo",
            "seg_model_path": os.path.join(curr_dir, "rotation.pt"),
            "seg_conf": cfg_yaml.get("seg_conf", 0.01),
            "seg_iou": cfg_yaml.get("seg_iou", 0.7),
            "seg_imgsz": cfg_yaml.get("seg_imgsz", 640),
            "model_path": os.path.join(curr_dir, "../asset/madara_white.obj"),
            "model_scale": 112.0 / 135.0,
            # Keep units consistent with RealSense Z16 in mm for rosbag frame 80.
            "depth_scale": cfg_yaml.get("depth_scale", 1.0),
            # Relaxed thresholds to avoid early rejection on sparse frames.
            "min_points_for_pose": cfg_yaml.get("min_points_for_pose", 10),
            "min_fitness": cfg_yaml.get("min_fitness", -1.0),
            "max_rmse": cfg_yaml.get("max_rmse", 1e9),
            "min_proj_iou": cfg_yaml.get("min_proj_iou", 0.0),
        }
        self.seg_estimator = SAMSegmentEstimator(self.estimator_cfg)

    def estimate(self, color_img, depth_img, camera_matrix, do_viz=False):
        """
        Estimate pose(s) using rotation_seg_fill.

        Returns a list with a single dict:
            {"pose": 4x4, "tvec": 3, "rvec": 3, "raw": result_dict}
        or empty list if no candidate.
        """
        depth_f32 = depth_img.astype(np.float32)
        res = self.seg_estimator.estimate(color_img, depth_f32, camera_matrix)
        if res is None:
            if do_viz:
                cv2.imshow("rotation_seg_fill", color_img)
                cv2.waitKey(1)
            return []

        pose = res["pose"]
        tvec = pose[:3, 3]
        rvec, _ = cv2.Rodrigues(pose[:3, :3])
        rvec = rvec.reshape(-1)

        if do_viz and res.get("mask") is not None:
            mask = res["mask"].astype(np.uint8)
            overlay = color_img.copy()
            overlay[mask == 1] = (0.5 * overlay[mask == 1] + 0.5 * np.array([0,255,0])).astype(np.uint8)
            cv2.imshow("rotation_seg_fill", overlay)
            cv2.waitKey(1)

        return [{"pose": pose, "tvec": tvec, "rvec": rvec, "raw": res}]
