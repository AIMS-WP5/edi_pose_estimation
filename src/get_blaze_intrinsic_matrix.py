from pypylon import pylon

# Connect to the first camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Check all available features
# for feature in camera.GetNodeMap().GetNodes():
#     if "Scan3d" in feature.GetNode().GetName():
#         print(feature.GetNode().GetName())

try:
    fx = float(camera.Scan3dFocalLength.GetValue())
    fy = float(camera.Scan3dFocalLengthY.GetValue())
    cx = float(camera.Scan3dPrincipalPointU.GetValue())
    cy = float(camera.Scan3dPrincipalPointV.GetValue())
    gray2mm = float(camera.Scan3dCoordinateScale.GetValue())

    intrinsic_matrix = [
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ]

    print("Intrinsic matrix:")
    for row in intrinsic_matrix:
        print(row)
    print(f"gray2mm: {gray2mm}")

    prev_val = camera.Scan3dDistortionCoefficientSelector.Value
    print("prev val:", prev_val)
    # values = ["k1", "k2", "p1", "p2", "k3"]
    # for idx in range(4):
    #     # error thrown if value not available
    #     camera.Scan3dDistortionCoefficientSelector.Value = values[idx]
    #     print(float(camera.Scan3dDistortionCoefficientValue.GetValue()))
    # camera.Scan3dDistortionCoefficientSelector.Value = prev_val

except Exception as e:
    print("Could not retrieve parameters:", e)

camera.Close()
