import numpy as np
import cv2
import time
import estimator
import csv
import datetime
import os
import yaml
import sys

def main():
    ## setup csv file for data recording/logging
    if len(sys.argv) < 2:
        print("Provide test directory!")
        exit(1)
    dir_path = sys.argv[1]
    folder_name = dir_path.split('/')[-2]
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = curr_dir + f"/../testing/zivid/{folder_name}.csv"
    log_file = open(log_file_path, "w", newline="")
    field_names = [
        "Case",
        "True T x", "True T y", "True T z", "True R x", "True R y", "True R z", 
        "T x", "T y", "T z", "R x", "R y", "R z", 
        "Fitness", "Inlier rmse", "Correspondence set size", 
        "Detection time",
        "cos", "pose error"
    ]
    writer = csv.DictWriter(log_file, fieldnames=field_names)
    writer.writeheader()

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
    camera_matrix = np.array([
        [2769.863525390625, 0.,                 943.1942138671875],
        [0.,                2769.16650390625,   585.1966552734375],
        [0.,                0.,                 1.]])
    dist_coeffs = np.array([-0.27518853545188904, 0.4559890329837799, -0.0008437871001660824, -0.0001234150113305077, -0.7441142797470093])

    try:
        with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.is_file() and ".png" in entry.path:
                    color_path = entry.path
                    depth_path = f"{entry.path.split('.png')[0]}.npy"
                    depth_image = np.load(depth_path)
                    color_image = cv2.imread(color_path)
                    print(f"Processing {color_path}")

                    detection_start = time.time()
                    result_poses = sam_seg_estimator.estimate(color_image, depth_image, camera_matrix)
                    detection_time = time.time() - detection_start
                    print("Detected poses:",result_poses)
                    if result_poses == []:
                        continue
                    axes_img = color_image.copy()
                    for pose in result_poses:
                        cv2.drawFrameAxes(axes_img, camera_matrix, dist_coeffs, pose[1], pose[0], 0.05, 1)

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
                        print("Board not detected")
                        continue
                
                    # calculate pose error and cos
                    rvec_true = bottle_rvec.reshape(1,3)
                    R_true, _ = cv2.Rodrigues(rvec_true)
                    z_true = R_true[:, 2]
                    rvec_est = result_poses[0][1]
                    R_est, _ = cv2.Rodrigues(rvec_est)
                    z_est = R_est[:, 2]
                    cos_phi = (np.dot(z_true, z_est)) / (np.linalg.norm(z_true) * np.linalg.norm(z_est))

                    tvec_true = bottle_tvec.reshape(1,3)
                    tvec_est = result_poses[0][0]
                    pose_error = np.linalg.norm(tvec_est - tvec_true)

                    # write results to log file
                    color_img_name = color_path.split('/')[-1]
                    row = {
                        "Case": color_img_name,
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
                        "cos": cos_phi,
                        "pose error": pose_error
                    }
                    writer.writerow(row)

                    cv2.imshow("Pose estimation", axes_img)
                    cv2.waitKey(1)
                    img_path = curr_dir + f"/../testing/zivid/{folder_name}/{color_img_name}"
                    cv2.imwrite(img_path, axes_img)
    finally:
        cv2.destroyAllWindows()
        log_file.close()


if __name__ == '__main__':
    main()
