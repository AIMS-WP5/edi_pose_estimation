import pyrealsense2 as rs
import numpy as np
import cv2
import time
from pose_estimator import estimate_pose, estimate_pose_fast, estimation_setup

# ground truth bottle position relative to aruco board, meters and radians
BOARD_BOTTLE_TVEC = np.array([0.07, 0.00, -0.10])
BOARD_BOTTLE_RVEC = np.array([0.00, 0.00, np.pi/4])

# Load ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
marker_length = 0.02 # meters
marker_separation = 0.01
board = cv2.aruco.GridBoard((5,7), marker_length, marker_separation, aruco_dict)

# setup for faster pose estimation
det_model, cls_idxs, sam, DOWN_SAMPLE_SIZE, ref_pcd = estimation_setup()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# Get intrinsic matrix of camera
intr = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx = float(intr.fx) # Focal length of x
fy = float(intr.fy) # Focal length of y
ppx = float(intr.ppx) # Principle Point Offsey of x (aka. cx)
ppy = float(intr.ppy) # Principle Point Offsey of y (aka. cy)
axs = 0.0 # Axis skew

camera_matrix = np.array([[fx, axs, ppx],
                    [0.0, fy, ppy],
                    [0.0, 0.0, 1.0]])
dist_coeffs = np.asanyarray(intr.coeffs)

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("Depth camera with Color sensor is required")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for both frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # cv2.imwrite("./testing/color.png", color_image)
        # cv2.imwrite("./testing/depth.png", depth_image)

        # result_poses = estimate_pose(color_img=color_image, depth_img=depth_image, Kdepth=camera_matrix)
        result_poses = estimate_pose_fast(color_image, depth_image, camera_matrix, det_model, cls_idxs, sam, DOWN_SAMPLE_SIZE, ref_pcd)
        print("Detected poses:",result_poses)
        # with open("./testing/camera_intrinsics.txt", "w") as file:
        #     file.write(str(camera_matrix))
        #     file.write(f"\n{result_poses}")
        for pose in result_poses:
            cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, pose[1], pose[0], 0.05, 1)

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
            # print(f"Rotation:\n{rvec}")
            # print(f"Translation:\n{tvec}")
            R_board, _ = cv2.Rodrigues(rvec)
            R_bottle, _ = cv2.Rodrigues(BOARD_BOTTLE_RVEC)
            R = R_board.dot(R_bottle)
            bottle_rvec, _ = cv2.Rodrigues(R)
            bottle_tvec = tvec + R_board.dot(BOARD_BOTTLE_TVEC.reshape(3,1))
            cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, bottle_rvec, bottle_tvec, 0.05, 2)

        cv2.imshow("Pose estimation", color_image)
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # time.sleep(0.1)

finally:

    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
