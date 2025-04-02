from pypylon import pylon
import cv2
import numpy as np

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# demonstrate some feature access
new_width = camera.Width.Value - camera.Width.Inc
if new_width >= camera.Width.Min:
    camera.Width.Value = new_width

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_Mono8
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        img = grabResult.Array
        cv2.imshow("Blaze view", img)

        conv_img = converter.Convert(grabResult)
        conv_img = conv_img.Array
        cv2.imshow("Converted", conv_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    grabResult.Release()
camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
