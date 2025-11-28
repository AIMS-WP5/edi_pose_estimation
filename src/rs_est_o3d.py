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
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    # extrinsic = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]] # to roatate pcd around for better viewing
    extrinsic = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] # identity
    dist_coeffs = np.asanyarray(intr.coeffs)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # create align object
    align_to = rs.stream.color # rs.stream.infrared for d405
    align = rs.align(align_to)
    # Start streaming
    pipeline.start(config)

    depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()


    def convert_rs_frames_to_pointcloud(rs_frames):
        aligned_frames = align.process(rs_frames)
        rs_depth_frame = aligned_frames.get_depth_frame()
        np_depth = np.asanyarray(rs_depth_frame.get_data())
        o3d_depth = o3d.geometry.Image(np_depth)

        rs_color_frame = aligned_frames.get_color_frame()
        np_color = np.asanyarray(rs_color_frame.get_data())
        np_color = cv2.cvtColor(np_color, cv2.COLOR_RGBA2BGR)
        o3d_color = o3d.geometry.Image(np_color)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1/depth_scale, convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, pinhole_camera_intrinsic, extrinsic)

        return pcd


    try:
        # setup vis window
        rs_frames = pipeline.wait_for_frames()
        pcd = convert_rs_frames_to_pointcloud(rs_frames)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud Visualizer",
                        width=800, height=800)
        vis.add_geometry(pcd)
        pcd_red_bottle = copy.deepcopy(sam_seg_estimator.ref_pcd)
        vis.add_geometry(pcd_red_bottle)
        render_opt = vis.get_render_option()
        render_opt.point_size = 2

        while True:
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Wait for both frames: depth and color
            frames = pipeline.wait_for_frames()

            pcd_new = convert_rs_frames_to_pointcloud(frames)
            pcd.points = pcd_new.points
            pcd.colors = pcd_new.colors
            vis.update_geometry(pcd)
            if vis.poll_events():
                vis.update_renderer()
            else:
                break

            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = depth_scale * depth_image # convert to m

            result_poses = sam_seg_estimator.estimate(color_image, depth_image, camera_matrix, True)
            print("Detected poses:",result_poses)
            if result_poses == []:
                cv2.imshow("Pose estimation", color_image)
                pcd_red_bottle.points = o3d.utility.Vector3dVector([])
                vis.update_geometry(pcd_red_bottle)
                if vis.poll_events():
                    vis.update_renderer()
                else:
                    break
                continue
            axes_img = color_image.copy()
            for pose in result_poses:
                cv2.drawFrameAxes(axes_img, camera_matrix, dist_coeffs, pose[1], pose[0], 0.05, 1)


            ref_pcd_tmp = copy.deepcopy(sam_seg_estimator.ref_pcd)
            pcd_best = ref_pcd_tmp.transform(result_poses[0][2].transformation)
            pcd_red_bottle.points = pcd_best.points
            pcd_red_bottle.colors = pcd_best.colors
            vis.update_geometry(pcd_red_bottle)
            if vis.poll_events():
                vis.update_renderer()
            else:
                break

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

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()

if __name__ == '__main__':
    rclpy.init()
    main()
    rclpy.shutdown()
