import os
import cv2
import yaml

curr_dir = os.path.dirname(os.path.abspath(__file__))
aruco_config_path = curr_dir + "/../config/aruco.yml"
with open(aruco_config_path, "r") as config_file:
    aruco_config = yaml.safe_load(config_file)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
marker_length = aruco_config["marker_length"] # m
marker_separation = aruco_config["marker_separation"] # m
board = cv2.aruco.GridBoard((5,7), marker_length, marker_separation, aruco_dict)


print(board.getMarkerLength(), board.getMarkerSeparation())
marker_length_px = 100
marker_separation_px = 50
imgSize = (5 * (marker_length_px + marker_separation_px) + marker_separation_px, 7 * (marker_length_px + marker_separation_px) + marker_separation_px)
img = cv2.aruco.drawPlanarBoard(board, imgSize, 50, 1)
cv2.imwrite("aruco_board.png", img)
