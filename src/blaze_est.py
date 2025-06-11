from pypylon import pylon
import numpy as np
import cv2
import estimator
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import scipy

def main():
    ## ros setup
    node = rclpy.create_node("blaze_pose_estimation")
    pose_pub = Node.create_publisher(node, Pose, "/bottle_pose", 10)

    ## estimator setup
    sam_seg_estimator = estimator.Estimator()


    ## Camera setup
    dc = pylon.DeviceInfo()
    dc.SetDeviceClass("BaslerGTC/Basler/GenTL_Producer_for_Basler_blaze_101_cameras")
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(dc))
    camera.Open()
    # Print the model name of the camera.
    print("Using device ", camera.GetDeviceInfo().GetModelName())

    # camera.OperatingMode.Value = 'ShortRange'
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
    camera_matrix = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]])
    gray2mm = float(camera.Scan3dCoordinateScale.GetValue())


    print('To exit, press ESC in one of the image windows')
    # Grabbing continuously (video) with minimal delay.
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    while camera.IsGrabbing():
        # Break by pressing 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

        grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

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
                    color_img = (_2d_intensity * 255.0 / 65536.0).astype(np.uint8)
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
            depth_img = _3d * gray2mm / 1000

            # cv2.imwrite("/home/arnis/aitools/pose_estimation/tmp/blaze_intensity.png", color_img)
            # cv2.imwrite("/home/arnis/aitools/pose_estimation/tmp/blaze_depth.png", depth_img)

            result_poses = sam_seg_estimator.estimate(color_img, depth_img, camera_matrix, False)
            print("Detected poses:",result_poses)
            if result_poses == []:
                cv2.imshow("Pose estimation", color_img)
                continue
            axes_img = color_img.copy()
            for pose in result_poses:
                cv2.drawFrameAxes(axes_img, camera_matrix, np.array([]), pose[1], pose[0], 0.05, 1)

            est_pose_tvec = np.array(result_poses[0][0])
            # convert rotation vector to quaternion
            est_pose_rvec = np.array(result_poses[0][1])
            est_pose_rmat, _ = cv2.Rodrigues(est_pose_rvec)
            est_pose_quat = scipy.spatial.transform.Rotation.from_matrix(est_pose_rmat).as_quat() # [x y z w]

            # publish ros msg with estimated pose
            pose_msg = Pose()
            pose_msg.position.x = est_pose_tvec[0]
            pose_msg.position.y = est_pose_tvec[1]
            pose_msg.position.z = est_pose_tvec[2]
            pose_msg.orientation.x = est_pose_quat[0]
            pose_msg.orientation.y = est_pose_quat[1]
            pose_msg.orientation.z = est_pose_quat[2]
            pose_msg.orientation.w = est_pose_quat[3]
            pose_pub.publish(pose_msg)

            cv2.imshow("Pose estimation", axes_img)

        grabResult.Release()

    # Releasing the resource
    camera.StopGrabbing()
    cv2.destroyAllWindows()

if __name__=="__main__":
    rclpy.init()
    main()
    rclpy.shutdown()
