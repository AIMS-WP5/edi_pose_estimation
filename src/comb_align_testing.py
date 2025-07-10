from pypylon import pylon
import numpy as np
import cv2
import pyrealsense2 as rs
import math
import estimator
import csv
import datetime
import os
import yaml
# rvec = np.array([
#     [-0.02942914],
#     [-0.02656359],
#     [0.03366882]
# ])
rvec_col_to_depth = np.array([
    [0.0],
    [0.0],
    [0.0]
])
tvec_col_to_depth = np.array([
    [0.157],
    [0.088],
    [-0.002]
])


def align_depth_to_color(depth_image, color_image, K_depth, K_color, R, T, depth_scale_to_m):
    height = depth_image.shape[0]
    width = depth_image.shape[1]
    # print(height, width)
    fxd = K_depth[0][0]
    fyd = K_depth[1][1]
    cxd = K_depth[0][2]
    cyd = K_depth[1][2]
    fxrgb = K_color[0][0]
    fyrgb = K_color[1][1]
    cxrgb = K_color[0][2]
    cyrgb = K_color[1][2]

    aligned_points = np.zeros_like(color_image)
    aligned_point_colors = np.zeros_like(color_image)

    aligned_depth = np.zeros_like(depth_image)
    
    for v in range(height):
        for u in range(width):
            # first get 3d points from depth image
            z = depth_image[v, u] * depth_scale_to_m # convert ot m
            x = (u - cxd) * z / fxd
            y = (v - cyd) * z / fyd
            point = np.array([x, y, z])
            # transform to color coordinate frame
            point = (R @ point.T).T
            # print("point after rotation:", point)
            # print("T", T)
            point += T
            aligned_points[v, u] = point
            # print(f"Point {v} {u}: {point}")

            # then get corresponding color to each point
            x_col = (point[0] * fxrgb / point[2]) + cxrgb
            y_col = (point[1] * fyrgb / point[2]) + cyrgb

            # some stuff to aviod inf values
            if math.isinf(x_col):
                x_col_idx = width
            else:
                x_col_idx = int(round(x_col))

            if math.isinf(y_col):
                y_col_idx = height
            else:        
                y_col_idx = int(round(y_col))

            # assign the color and put depth info into that pixel coordinate
            if x_col_idx >= 0 and x_col_idx < width and y_col_idx >= 0 and y_col_idx < height:
                if aligned_depth[y_col_idx, x_col_idx] != 0 and aligned_depth[y_col_idx, x_col_idx] <= depth_image[v, u]:
                    continue
                aligned_point_colors[v, u] = color_image[y_col_idx, x_col_idx]
                aligned_depth[y_col_idx, x_col_idx] = depth_image[v, u]
    
    # print(len(aligned_points.reshape((-1,3))))
    # visualize result
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(aligned_points.reshape((-1,3)))
    # # pcd.colors = o3d.utility.Vector3dVector(aligned_point_colors.reshape((-1,3))/255)
    # # pcd.paint_uniform_color([0,1,0])
    # o3d.visualization.draw_geometries([pcd])
    
    return aligned_depth


def main():

    record_log = True
    use_aruco = True
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    if record_log:
        ## setup csv file for data recording/logging
        t = datetime.datetime.now()
        timestamp = f"{t.year}_{t.month}_{t.day}_{t.hour}_{t.minute}_{t.second}"
        # curr_dir = os.path.dirname(os.path.abspath(__file__))
        log_file_path = curr_dir + f"/../tmp/{timestamp}.csv"
        log_file = open(log_file_path, "w", newline="")
        field_names = [
            "True T x", "True T y", "True T z", "True R x", "True R y", "True R z", 
            "T x", "T y", "T z", "R x", "R y", "R z", 
            "Fitness", "Inlier rmse", "Correspondence set size"
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

    ## Blaze Camera setup
    dc = pylon.DeviceInfo()
    dc.SetDeviceClass("BaslerGTC/Basler/GenTL_Producer_for_Basler_blaze_101_cameras")
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(dc))
    camera.Open()
    # Print the model name of the camera.
    print("Using device ", camera.GetDeviceInfo().GetModelName())

    camera.OperatingMode.Value = 'ShortRange'
    # camera.OperatingMode.Value = 'LongRange'
    print("OperatingMode: ", camera.OperatingMode.Value)

    # camera.FastMode.Value = True
    print("FastMode: ", camera.FastMode.Value)

    # camera.FilterSpatial.Value = True
    print("FilterSpatial: ", camera.FilterSpatial.Value)

    # camera.FilterTemporal.Value = True
    print("FilterTemporal: ", camera.FilterTemporal.Value)

    # FilterTemporal must be enabled before setting
    # FilterStrengh is possible.
    if camera.FilterTemporal.Value:
        # camera.FilterStrength.Value = 200
        pass
    print("FilterStrength: ", camera.FilterStrength.Value)

    # camera.OutlierRemoval.Value = True
    print("OutlierRemoval: ", camera.OutlierRemoval.Value)

    # camera.ConfidenceThreshold.Value = 20
    print("ConfidenceThreshold: ", camera.ConfidenceThreshold.Value)

    # camera.GammaCorrection.Value = True
    print("GammaCorrection: ", camera.GammaCorrection.Value)

    # Set the working range to the values displayed for the Max. Depth [mm] and
    # Min. Depth [mm] parameters.
    # The working range depends on the current operating mode, so
    # the OperatingMode parameter must be set before adjusting DepthMax and DepthMin.
    # LongRange: 0 .. 9990mm
    # ShortRange: 0 .. 1498mm
    # camera.DepthMin.Value = 0
    # camera.DepthMax.Value = 9990
    print("Min. Depth [mm]: ", camera.DepthMin.Value)
    print("Max. Depth [mm]: ", camera.DepthMax.Value)

    # Control pixel formats for image components.
    # Range information can be sent either as a 16-bit gray value image or as
    # 3D coordinates (point cloud).
    # For this sample, we want to acquire 3D coordinates.
    # Note: To change the format of an image component, the Component Selector parameter
    # must first be set to the component
    # you want to configure.
    # To use 16-bit integer depth information, choose "Coord3D_C16" instead of "Coord3D_ABC32f".
    camera.ComponentSelector.Value = "Range"
    camera.ComponentEnable.Value = True
    camera.PixelFormat.Value = "Coord3D_C16" #"Coord3D_ABC32f"

    camera.ComponentSelector.Value = "Intensity"
    camera.ComponentEnable.Value = True
    camera.PixelFormat.Value = "Mono16"

    # camera.ComponentSelector.Value = "Confidence"
    # camera.ComponentEnable.Value = True
    # camera.PixelFormat.Value = "Confidence16"

    # Enable GenDC.
    camera.GenDCStreamingMode.Value = "On"

    # get intrinsic matrix
    fx = float(camera.Scan3dFocalLength.GetValue())
    fy = float(camera.Scan3dFocalLengthY.GetValue())
    cx = float(camera.Scan3dPrincipalPointU.GetValue())
    cy = float(camera.Scan3dPrincipalPointV.GetValue())
    blaze_camera_matrix = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]])
    print("Blaze camera matrix:\n", blaze_camera_matrix)
    gray2mm = float(camera.Scan3dCoordinateScale.GetValue())

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
    rs_camera_matrix = np.array([
        [fx, axs, ppx],
        [0.0, fy, ppy],
        [0.0, 0.0, 1.0]])
    print("RealSense camera matrix:\n", rs_camera_matrix)
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

    print('To exit, press Q in one of the image windows')

    # start both cameras
    pipeline.start(config)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)


    while camera.IsGrabbing():
        # Break by pressing 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

        # get blaze cam frames
        grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

        # get realsense frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        rs_depth_image = np.asanyarray(depth_frame.get_data()) # not used
        rs_color_image = np.asanyarray(color_frame.get_data())

        if grabResult.GrabSucceeded():
            # Get the grab result as a PylonDataContainer, e.g., when working with 3D cameras.
            pylonDataContainer = grabResult.GetDataContainer()
            # Access data components if the component type indicates image data
            for componentIndex in range(pylonDataContainer.DataComponentCount):
                pylonDataComponent = pylonDataContainer.GetDataComponent(componentIndex)
                if pylonDataComponent.ComponentType == pylon.ComponentType_Intensity:
                    # Access the component data.
                    intensity = pylonDataComponent.Array
                    _2d_intensity = intensity.reshape(pylonDataComponent.Height, pylonDataComponent.Width)
                    # convert to shape (h, w, 3)
                    # _2d_new = np.ones((pylonDataComponent.Height, pylonDataComponent.Width, 3))
                    _2d_intensity = cv2.cvtColor(_2d_intensity, cv2.COLOR_GRAY2BGR)
                    color_img_blaze = (_2d_intensity * 255.0 / 65536.0).astype(np.uint8)
                    # print(_2d_new[0])
                    # print(_2d_intensity[0])
                elif pylonDataComponent.ComponentType == pylon.ComponentType_Range:
                    pointcloud = pylonDataComponent.Array
                    # _3d = pointcloud.reshape(pylonDataComponent.Height, pylonDataComponent.Width, 3)
                    _3d = pointcloud
                # elif pylonDataComponent.ComponentType == pylon.ComponentType_Confidence:
                #     confidence = pylonDataComponent.Array
                #     _2d_confidence = confidence.reshape(pfter rotation:", point)
            # print("T", T)ylonDataComponent.Height, pylonDataComponent.Width)
                pylonDataComponent.Release()

            # Show the captured images as grayscale.
            # We only show the z-component of the point cloud.
            # If you choose "Coord3D_C16" as pixel format, you have to remove [:,:,2].
            # OpenCV can't show float values. We convert it for visualization to uint8.
            _3d_scaled = _3d * 255.0 / camera.DepthMax.Value
            # cv2.imshow('depth', _3d_scaled[:, :, 2].astype(np.uint8))
            cv2.imshow('Blaze original depth', _3d_scaled.astype(np.uint8))
            # cv2.imshow('_2d_confidence', _2d_confidence)

            # convert depth data to meters
            depth_img_blaze = _3d * gray2mm / 1000

            # cv2.imwrite("/home/arnis/aitools/pose_estimation/tmp/blaze_intensity.png", color_img_blaze)
            # cv2.imwrite("/home/arnis/aitools/pose_estimation/tmp/blaze_depth.png", depth_img_blaze)

            # axes_img_blaze = color_img_blaze.copy()
            # axes_img_rs = rs_color_image.copy()

            
            R, _ = cv2.Rodrigues(rvec_col_to_depth)
            T = np.array([tvec_col_to_depth[0][0], tvec_col_to_depth[1][0], tvec_col_to_depth[2][0]])
            # print(f"Result:\nR: {rvec}\nT: {tvec}")

            cv2.imshow("RealSense color", rs_color_image)

            # print("Aligning frames")
            # print(R,"\n", T)
            aligned_image = align_depth_to_color(
                _3d,
                rs_color_image,
                blaze_camera_matrix,
                rs_camera_matrix,
                R,
                T,
                gray2mm/1000
            )
            aligned_scaled = aligned_image * 255.0 / camera.DepthMax.Value
            cv2.imshow("Blaze aligned depth", aligned_scaled.astype(np.uint8))
            # cv2.imwrite("/home/arnis/aitools/pose_estimation/tmp/comb_aligned.png", aligned_image)
            # cv2.imwrite("/home/arnis/aitools/pose_estimation/tmp/comb_color.png", rs_color_image)
            # cv2.imwrite("/home/arnis/aitools/pose_estimation/tmp/comb_rs_depth.png", rs_depth_image)

            aligned_image = aligned_image * gray2mm / 1000
            result_poses = sam_seg_estimator.estimate(rs_color_image, aligned_image, rs_camera_matrix, False)
            print("Detected poses:",result_poses)
            if result_poses == []:
                continue
            axes_img = rs_color_image.copy()
            for pose in result_poses:
                cv2.drawFrameAxes(axes_img, rs_camera_matrix, dist_coeffs, pose[1], pose[0], 0.05, 1)

            if use_aruco:
                # detect board position
                gray = cv2.cvtColor(rs_color_image, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                corners, ids, rejected, recovered_ids = cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejected, rs_camera_matrix, dist_coeffs)
                rvec = None
                tvec = None
                if ids is not None:
                    retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                        corners, ids, board, rs_camera_matrix, dist_coeffs, rvec, tvec, False
                    )
                    R_board, _ = cv2.Rodrigues(rvec)
                    R_bottle, _ = cv2.Rodrigues(BOARD_BOTTLE_RVEC)
                    R = R_board.dot(R_bottle)
                    bottle_rvec, _ = cv2.Rodrigues(R)
                    bottle_tvec = tvec + R_board.dot(BOARD_BOTTLE_TVEC.reshape(3,1))
                    cv2.drawFrameAxes(axes_img, rs_camera_matrix, dist_coeffs, bottle_rvec, bottle_tvec, 0.05, 2)
                else:
                    print("Board not detected, dropping results")
                    continue
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
                    "Correspondence set size": len(result_poses[0][2].correspondence_set)
                }
                writer.writerow(row)
            
            cv2.imshow("Estimated bottle pose", axes_img)

        grabResult.Release()

    # Releasing the resource
    camera.StopGrabbing()
    cv2.destroyAllWindows()
    pipeline.stop()
    if record_log:
        log_file.close()

if __name__=="__main__":
    main()
