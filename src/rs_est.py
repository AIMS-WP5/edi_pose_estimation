import pyrealsense2 as rs
import numpy as np
import cv2
import estimator
import open3d as o3d
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import scipy
import copy

do_viz = False
use_gpd = False

def main():
    node = rclpy.create_node("realsense_pose_estimation")
    pose_pub = Node.create_publisher(node, Pose, "/bottle_pose", 10)

    ## estimator setup
    sam_seg_estimator = estimator.Estimator()

    ## camera setup
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    # Get intrinsic matrix of camera
    intr = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx = float(intr.fx) # Focal length of x
    fy = float(intr.fy) # Focal length of y
    ppx = float(intr.ppx) # Principle Point Offsey of x (aka. cx)
    ppy = float(intr.ppy) # Principle Point Offsey of y (aka. cy)
    axs = 0.0 # Axis skew
    camera_matrix = np.array([
        [fx, axs, ppx],
        [0.0, fy, ppy],
        [0.0, 0.0, 1.0]])
    # print("Camera matrix:", camera_matrix)
    dist_coeffs = np.asanyarray(intr.coeffs)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
    # create align object
    align_to = rs.stream.color
    align = rs.align(align_to)
    # Start streaming
    pipeline.start(config)

    depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

    try:
        while True:
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Wait for both frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = depth_scale * depth_image # convert to m

            if do_viz:
                pts = cv2.rgbd.depthTo3d(depth_image, camera_matrix)
                verts = pts.reshape((-1,3))
                idx = ~np.isnan(verts).any(axis=1)
                verts = verts[idx,:]
                color_image_1 = cv2.cvtColor(color_image, cv2.COLOR_RGBA2BGR)
                color = color_image_1.reshape((-1,3))
                color = color[idx,:]

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(verts)
                pcd.colors = o3d.utility.Vector3dVector(color/255)
                o3d.visualization.draw_geometries(
                    [pcd])

            # cv2.imwrite("/home/arnis/aitools/pose_estimation/tmp/realsense_color.png", color_image)
            # cv2.imwrite("/home/arnis/aitools/pose_estimation/tmp/realsense_depth.png", depth_image)

            result_poses = sam_seg_estimator.estimate(color_image, depth_image, camera_matrix, do_viz)
            print("Detected poses:",result_poses)
            if result_poses == []:
                cv2.imshow("Pose estimation", color_image)
                continue
            axes_img = color_image.copy()
            for pose in result_poses:
                cv2.drawFrameAxes(axes_img, camera_matrix, dist_coeffs, pose[1], pose[0], 0.05, 1)

            if use_gpd:
                # write a shifted pcd file for grasp pose estimation using GPD
                # estimated pose coordinates is the new origin point
                est_pose_tvec = np.array(result_poses[0][0])
                shift_vec = -1 * est_pose_tvec
                pts = cv2.rgbd.depthTo3d(depth_image, camera_matrix)
                verts = pts.reshape((-1,3))
                idx = ~np.isnan(verts).any(axis=1)
                verts = verts[idx,:]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(verts)
                pcd = pcd.translate(shift_vec)
                pcd_file_path = "/home/arnis/aitools/pose_estimation/tmp/depth.pcd" # todo: make a config param
                o3d.io.write_point_cloud(pcd_file_path, pcd)

            if do_viz:
                ref_pcd_tmp = copy.deepcopy(sam_seg_estimator.ref_pcd)
                pcd_best = ref_pcd_tmp.transform(result_poses[0][2].transformation)
                o3d.visualization.draw_geometries(
                    [pcd, pcd_best])

            # convert rotation vector to quaternion
            est_pose_rvec = np.array(result_poses[0][1])
            est_pose_rmat, _ = cv2.Rodrigues(est_pose_rvec)
            est_pose_quat = scipy.spatial.transform.Rotation.from_matrix(est_pose_rmat).as_quat() # [x y z w]

            # publish ros msg with estimated pose
            pose_msg = Pose()
            pose_msg.position.x = result_poses[0][0][0]
            pose_msg.position.y = result_poses[0][0][1]
            pose_msg.position.z = result_poses[0][0][2]
            pose_msg.orientation.x = est_pose_quat[0]
            pose_msg.orientation.y = est_pose_quat[1]
            pose_msg.orientation.z = est_pose_quat[2]
            pose_msg.orientation.w = est_pose_quat[3]
            pose_pub.publish(pose_msg)

            cv2.imshow("Pose estimation", axes_img)
            if do_viz:
                cv2.waitKey()

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rclpy.init()
    main()
    rclpy.shutdown()
