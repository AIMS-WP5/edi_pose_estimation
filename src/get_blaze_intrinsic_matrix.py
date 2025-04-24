from pypylon import pylon

# Connect to the first camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Check all available features
# for feature in camera.GetNodeMap().GetNodes():
#     if "Scan3d" in feature.GetNode().GetName():
#         print(feature.GetNode().GetName())

try:
    fx = float(camera.GetNodeMap().GetNode("Scan3dFocalLength").GetValue())
    fy = float(camera.GetNodeMap().GetNode("Scan3dFocalLengthY").GetValue())
    cx = float(camera.GetNodeMap().GetNode("Scan3dPrincipalPointU").GetValue())
    cy = float(camera.GetNodeMap().GetNode("Scan3dPrincipalPointV").GetValue())

    intrinsic_matrix = [
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ]

    print("Intrinsic matrix:")
    for row in intrinsic_matrix:
        print(row)

except Exception as e:
    print("Could not retrieve parameters:", e)

camera.Close()
