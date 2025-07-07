from pypylon import pylon
import numpy as np
import cv2
import os
import yaml
import pyrealsense2 as rs
import math


def avg_rotation(rvecs):
    """
    Compute the average rotation from a list of Rodrigues rotation vectors.

    Parameters:
    - rvecs: np.ndarray of shape (N, 3), where each row is a Rodrigues rotation vector.

    Returns:
    - avg_rvec: np.ndarray of shape (3,), the average Rodrigues rotation vector.
    """

    # Step 1: Convert all Rodrigues vectors to rotation matrices
    R_matrices = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]

    # Step 2: Average the rotation matrices
    R_stack = np.stack(R_matrices)
    R_avg = np.mean(R_stack, axis=0)

    # Step 3: Re-orthonormalize the averaged matrix using SVD
    U, _, Vt = np.linalg.svd(R_avg)
    R_avg_ortho = U @ Vt

    # Step 4: Convert the averaged rotation matrix back to a Rodrigues vector
    avg_rvec, _ = cv2.Rodrigues(R_avg_ortho)
    return avg_rvec.ravel()


def get_transform_depth2col(depth_rvec, depth_tvec, col_rvec, col_tvec):
    '''
    rvecs and tvecs are transforms **of an aruco board** position in
    the corresponding camera coordinate frame.

    Returns such a transform that a point p in depth coord frame 
    can be transformed into color coordinate frame using:
    p' = R*p + T
    '''
    if depth_rvec is None or depth_tvec is None or col_rvec is None or col_tvec is None:
        return np.array([0,0,0]), np.array([0,0,0])

    R0, _ = cv2.Rodrigues(depth_rvec)
    R1, _ = cv2.Rodrigues(col_rvec) 
    R_diff = R1 @ R0.T # to rotate color camera to depth orientation
    rvec, _ = cv2.Rodrigues(R_diff)
    tvec = col_tvec - depth_tvec # vector from color origin to depth

    return rvec, tvec


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
            print("point after rotation:", point)
            print("T", T)
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

    # idk just something
    blaze_bottle_rvec = None
    blaze_bottle_tvec = None
    rs_bottle_rvec = None
    rs_bottle_tvec = None

    ## ArUco setup
    curr_dir = os.path.dirname(os.path.abspath(__file__))
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
    # print("Camera matrix:", camera_matrix)
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

    num_samples = 0
    res_arr = []

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
                #     _2d_confidence = confidence.reshape(pylonDataComponent.Height, pylonDataComponent.Width)
                pylonDataComponent.Release()

            # Show the captured images as grayscale.
            # We only show the z-component of the point cloud.
            # If you choose "Coord3D_C16" as pixel format, you have to remove [:,:,2].
            # OpenCV can't show float values. We convert it for visualization to uint8.
            _3d_scaled = _3d * 255.0 / camera.DepthMax.Value
            # cv2.imshow('depth', _3d_scaled[:, :, 2].astype(np.uint8))
            cv2.imshow('depth', _3d_scaled.astype(np.uint8))
            # cv2.imshow('_2d_confidence', _2d_confidence)

            # convert depth data to meters
            depth_img_blaze = _3d * gray2mm / 1000

            # cv2.imwrite("/home/arnis/aitools/pose_estimation/tmp/blaze_intensity.png", color_img_blaze)
            # cv2.imwrite("/home/arnis/aitools/pose_estimation/tmp/blaze_depth.png", depth_img_blaze)

            axes_img_blaze = color_img_blaze.copy()
            axes_img_rs = rs_color_image.copy()

            # detect board position (blaze)
            gray_blaze = cv2.cvtColor(color_img_blaze, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray_blaze, aruco_dict, parameters=parameters)
            corners, ids, rejected, recovered_ids = cv2.aruco.refineDetectedMarkers(gray_blaze, board, corners, ids, rejected, blaze_camera_matrix, np.array([]))
            rvec = None
            tvec = None
            if ids is not None:
                retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                    corners, ids, board, blaze_camera_matrix, np.array([]), rvec, tvec, False
                )
                R_board, _ = cv2.Rodrigues(rvec)
                R_bottle, _ = cv2.Rodrigues(BOARD_BOTTLE_RVEC)
                R = R_board.dot(R_bottle)
                blaze_bottle_rvec, _ = cv2.Rodrigues(R)
                blaze_bottle_tvec = tvec + R_board.dot(BOARD_BOTTLE_TVEC.reshape(3,1))
                cv2.drawFrameAxes(axes_img_blaze, blaze_camera_matrix, np.array([]), blaze_bottle_rvec, blaze_bottle_tvec, 0.026, 2)
                # print("Board detected by blaze")
            else:
                print("Board not detected by blaze")
                continue

            # detect board position (realsense)
            gray_rs = cv2.cvtColor(rs_color_image, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray_rs, aruco_dict, parameters=parameters)
            corners, ids, rejected, recovered_ids = cv2.aruco.refineDetectedMarkers(gray_rs, board, corners, ids, rejected, rs_camera_matrix, dist_coeffs)
            rvec = None
            tvec = None
            if ids is not None:
                retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                    corners, ids, board, rs_camera_matrix, dist_coeffs, rvec, tvec, False
                )
                R_board, _ = cv2.Rodrigues(rvec)
                R_bottle, _ = cv2.Rodrigues(BOARD_BOTTLE_RVEC)
                R = R_board.dot(R_bottle)
                rs_bottle_rvec, _ = cv2.Rodrigues(R)
                rs_bottle_tvec = tvec + R_board.dot(BOARD_BOTTLE_TVEC.reshape(3,1))
                cv2.drawFrameAxes(axes_img_rs, rs_camera_matrix, dist_coeffs, rs_bottle_rvec, rs_bottle_tvec, 0.026, 2)
                # print("Board detected by realsense")
            else:
                print("Board not detected by realsense")
                continue

            print(f"Blaze:\nR:{blaze_bottle_rvec}\nT:{blaze_bottle_tvec}")
            print(f"RealSense:\nR:{rs_bottle_rvec}\nT:{rs_bottle_tvec}")
            if cv2.waitKey(5) & 0xFF == ord('s'):
                rvec, tvec = get_transform_depth2col(blaze_bottle_rvec, blaze_bottle_tvec, rs_bottle_rvec, rs_bottle_tvec)
                R, _ = cv2.Rodrigues(rvec)
                T = np.array([tvec[0][0], tvec[1][0], tvec[2][0]])
                print(f"Result:\nR: {rvec}\nT: {tvec}")
                num_samples += 1
                res_arr.append([rvec, tvec])

            cv2.imshow("Blaze pose estimation", axes_img_blaze)
            cv2.imshow("RealSense pose estimation", axes_img_rs)

            # if cv2.waitKey() & 0xFF == ord('y'):
            #     # align images using computed transform
            #     print("Aligning frames")
            #     print(R,"\n", T)
            #     aligned_image = align_depth_to_color(
            #         _3d,
            #         rs_color_image,
            #         blaze_camera_matrix,
            #         rs_camera_matrix,
            #         R,
            #         T,
            #         gray2mm/1000
            #     )
            #     cv2.imwrite("/home/arnis/aitools/pose_estimation/tmp/comb_aligned.png", aligned_image)
            #     cv2.imwrite("/home/arnis/aitools/pose_estimation/tmp/comb_color.png", rs_color_image)
            # else:
            #     continue

        grabResult.Release()

    # Releasing the resource
    camera.StopGrabbing()
    cv2.destroyAllWindows()
    pipeline.stop()
    print("Avg results")
    res_arr = np.array(res_arr)
    print(res_arr[:, 0].shape)
    print(avg_rotation(res_arr[:,0]))
    # avg translation
    print(np.average(res_arr[:,1,0]))
    print(np.average(res_arr[:,1,1]))
    print(np.average(res_arr[:,1,2]))
    print(f"From {num_samples} samples")

if __name__=="__main__":
    main()
