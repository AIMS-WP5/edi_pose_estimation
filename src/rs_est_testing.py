import pyrealsense2 as rs
import numpy as np
import cv2
import time
import estimator
import csv
import datetime
import os
import yaml
import subprocess
import open3d as o3d

record_log = False
use_aruco = True

def main():
    if record_log:
        ## setup csv file for data recording/logging
        t = datetime.datetime.now()
        timestamp = f"{t.year}_{t.month}_{t.day}_{t.hour}_{t.minute}_{t.second}"
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        log_file_path = curr_dir + f"/../testing/{timestamp}.csv"
        log_file = open(log_file_path, "w", newline="")
        field_names = [
            "True T x", "True T y", "True T z", "True R x", "True R y", "True R z", 
            "T x", "T y", "T z", "R x", "R y", "R z", 
            "Fitness", "Inlier rmse", "Correspondence set size", 
            "Detection time", "Total time"
        ]
        writer = csv.DictWriter(log_file, fieldnames=field_names)
        writer.writeheader()
        use_aruco = True # otherwise true pose cannot be determined

    if use_aruco:
        ## ArUco setup
        aruco_config_path = curr_dir + "/../config/aruco.yml"
        with open(aruco_config_path, "r") as config_file:
            aruco_config = yaml.safe_load(config_file)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()
        marker_length = aruco_config["marker_length"] # m
        marker_separation = aruco_config["marker_separation"] # m
        board = cv2.aruco.GridBoard((5,7), marker_length, marker_separation, aruco_dict)
        # "ground truth" bottle position relative to aruco board, meters and radians
        offset_tvec_dict = aruco_config["board_to_bottle_tvec"]
        offset_rvec_dict = aruco_config["board_to_bottle_rvec"]
        BOARD_BOTTLE_TVEC = np.array([offset_tvec_dict['x'], offset_tvec_dict['y'], offset_tvec_dict['z']])
        BOARD_BOTTLE_RVEC = np.array([offset_rvec_dict['x'], offset_rvec_dict['y'], offset_rvec_dict['z']])
        # BOARD_BOTTLE_TVEC = np.array([-0.037, 1.50, -0.039])
        # BOARD_BOTTLE_RVEC = np.array([0.00, -np.pi/2, 0.0])
        # BOARD_BOTTLE_RVEC = np.array([np.pi/np.sqrt(2), 0.00, np.pi/np.sqrt(2)])

    ## estimator setup
    sam_seg_estimator = estimator.Estimator()

    ## camera setup
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    # device_product_line = str(device.get_info(rs.camera_info.product_line))
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
    dist_coeffs = np.asanyarray(intr.coeffs)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
    # config.enable_device_from_file("/home/arnis/AIMS/aitools/pose_estimation/rs_bags/20250210_154204.bag")
    # create align object
    align_to = rs.stream.color
    align = rs.align(align_to)
    # Start streaming
    pipeline.start(config)

    try:
        prev_time = time.time()
        while True:
            # Wait for both frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # cv2.imwrite("./utils_testing/color.png", color_image)
            # cv2.imwrite("./utils_testing/depth.png", depth_image)

            detection_start = time.time()
            result_poses = sam_seg_estimator.estimate(color_image, depth_image, camera_matrix)
            detection_time = time.time() - detection_start
            print("Detected poses:",result_poses)
            if result_poses == []:
                continue
            axes_img = color_image.copy()
            for pose in result_poses:
                cv2.drawFrameAxes(axes_img, camera_matrix, dist_coeffs, pose[1], pose[0], 0.05, 1)

            if use_aruco:
                # detect board position
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                corners, ids, rejected, recovered_ids = cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejected, camera_matrix, dist_coeffs)
                rvec = None
                tvec = None
                if ids is not None:
                    retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                        corners, ids, board, camera_matrix, dist_coeffs, rvec, tvec, False
                    )
                    R_board, _ = cv2.Rodrigues(rvec)
                    R_bottle, _ = cv2.Rodrigues(BOARD_BOTTLE_RVEC)
                    R = R_board.dot(R_bottle)
                    bottle_rvec, _ = cv2.Rodrigues(R)
                    bottle_tvec = tvec + R_board.dot(BOARD_BOTTLE_TVEC.reshape(3,1))
                    cv2.drawFrameAxes(axes_img, camera_matrix, dist_coeffs, bottle_rvec, bottle_tvec, 0.05, 2)
                else:
                    print("Board not detected, dropping results")
                    continue
            
            # write results to log file
            curr_time = time.time()
            total_time = curr_time - prev_time
            prev_time = curr_time
            if record_log:
                row = {
                    "True T x": bottle_tvec[0][0],
                    "True T y": bottle_tvec[1][0],
                    "True T z": bottle_tvec[2][0],
                    "True R x": bottle_rvec[0][0],
                    "True R y": bottle_rvec[1][0],
                    "True R z": bottle_rvec[2][0],
                    "T x": result_poses[0][0][0],
                    "T y": result_poses[0][0][1],
                    "T z": result_poses[0][0][2],
                    "R x": result_poses[0][1][0],
                    "R y": result_poses[0][1][1],
                    "R z": result_poses[0][1][2],
                    "Fitness": result_poses[0][2].fitness,
                    "Inlier rmse": result_poses[0][2].inlier_rmse,
                    "Correspondence set size": len(result_poses[0][2].correspondence_set),
                    "Detection time": detection_time,
                    "Total time": total_time
                }
                writer.writerow(row)

            # grasp pose estimation
            est_pose_tvec = np.array(result_poses[0][0])
            est_pose_tvec *= -1
            pts = cv2.rgbd.depthTo3d(depth_image, camera_matrix)
            verts = pts.reshape((-1,3))
            idx = ~np.isnan(verts).any(axis=1)
            verts = verts[idx,:]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(verts)
            pcd = pcd.translate(est_pose_tvec)
            o3d.io.write_point_cloud("/home/arnis/AIMS/aitools/pose_estimation/testing/depth.pcd", pcd)
            subprocess.run(["/home/arnis/gpd/build/detect_grasps", "/home/arnis/gpd/cfg/eigen_params.cfg", "/home/arnis/AIMS/aitools/pose_estimation/testing/depth.pcd", str(est_pose_tvec[0]), str(est_pose_tvec[1]), str(est_pose_tvec[2])])

            cv2.imshow("Pose estimation", axes_img)
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            print("Waiting 5s...")
            time.sleep(5)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if record_log:
            log_file.close()


if __name__ == '__main__':
    main()
