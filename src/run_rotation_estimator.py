import os
import numpy as np
import cv2

# from rotation_seg import SAMSegmentEstimator
from estimator2 import Estimator

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import scipy

import pyrealsense2 as rs


def camera_setup():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    # config.enable_device_from_file("/home/arnis/aims/rotation_predict/20260217_161517.bag")
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    # Get intrinsic matrix of camera
    intr = pipeline_profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    fx = float(intr.fx) # Focal length of x
    fy = float(intr.fy) # Focal length of y
    ppx = float(intr.ppx) # Principle Point Offsey of x (aka. cx)
    ppy = float(intr.ppy) # Principle Point Offsey of y (aka. cy)
    axs = 0.0 # Axis skew
    camera_matrix = np.array([
        [fx, axs, ppx],
        [0.0, fy, ppy],
        [0.0, 0.0, 1.0]])
    print("Camera matrix:", camera_matrix)
    dist_coeffs = np.asanyarray(intr.coeffs)
    # create align object
    align_to = rs.stream.infrared # for d405, for other cameras use rs.stream.color
    align = rs.align(align_to)
    # Start streaming
    pipeline.start(config)

    depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
    print("Depth scale:", depth_scale)

    return pipeline, camera_matrix, align, depth_scale, dist_coeffs

def main():

    node = rclpy.create_node("realsense_pose_estimation")
    pose_pub = Node.create_publisher(node, PoseStamped, "/bottle_pose", 10)

    pipeline, K, align, depth_scale, dist_coeffs = camera_setup()
    try:
        sam_cfg = {
            "seg_model_type": "yolo",
            "seg_model_path": os.path.join("rotation.pt"),
            "model_path": os.path.join("../asset/madara_white.obj"),
            "model_scale": 112.0 / 135.0,
            "depth_scale": 1.0,  # keep mm
            "min_points_for_pose": 10,
            "min_fitness": -1.0,
            "max_rmse": 1e9,
            "min_proj_iou": 0.0,
        }
        
        est = Estimator()

        while True:
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_raw = np.asanyarray(depth_frame.get_data())
            color = np.asanyarray(color_frame.get_data())
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

            # Run Estimator wrapper.
            res_est_list = est.estimate(color, depth_raw, K, do_viz=False)
            res_est = res_est_list[0] if res_est_list else None

            print(f"Estimator result: {'None' if res_est is None else 'OK'}")
            if res_est:
                print("pose (estimator):\n", res_est['pose'])
                position_rez = depth_scale * np.array(res_est["tvec"]) # lai parveidotu uz m
                est_pose_rvec = np.array(res_est["rvec"])
                est_pose_rmat, _ = cv2.Rodrigues(est_pose_rvec)
                est_pose_quat = scipy.spatial.transform.Rotation.from_matrix(est_pose_rmat).as_quat() # [x y z w]

                
                pose_msg = PoseStamped()
                pose_msg.pose.position.x = position_rez[0]
                pose_msg.pose.position.y = position_rez[1]
                pose_msg.pose.position.z = position_rez[2]
                pose_msg.pose.orientation.x = est_pose_quat[0]
                pose_msg.pose.orientation.y = est_pose_quat[1]
                pose_msg.pose.orientation.z = est_pose_quat[2]
                pose_msg.pose.orientation.w = est_pose_quat[3]
                pose_pub.publish(pose_msg)

                axes_img = color.copy()
                cv2.drawFrameAxes(axes_img, K, dist_coeffs, res_est["rvec"], position_rez, 0.05, 1)
                cv2.imshow("Est Position", axes_img)

            else:
                cv2.imshow("Est Position", color)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    rclpy.init()
    main()
    rclpy.shutdown()
