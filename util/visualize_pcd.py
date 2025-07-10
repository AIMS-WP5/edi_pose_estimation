import cv2
import open3d as o3d
import numpy as np

depth_img = cv2.imread("/home/arnis/aitools/pose_estimation/tmp/comb_aligned.png", cv2.IMREAD_UNCHANGED)
color_img = cv2.imread("/home/arnis/aitools/pose_estimation/tmp/comb_color.png")
# print(depth_img.shape)
# print(color_img.shape)
# K_depth = np.array([[505.2425231933594, 0, 317.013916015625], [0, 505.2425231933594, 225.34295654296875], [0, 0, 1]]) # blaze
# K_depth = np.array([[300., 0., 400.], [  0., 300., 300.], [  0.,   0.,   1.]]) # simulated
K_depth = np.array([[612.82, 0, 320.63], [0, 612.95, 241.23], [0, 0, 1]]) # realsense 435 color
pts = cv2.rgbd.depthTo3d(depth_img, K_depth)
# print(pts.shape)
verts = pts.reshape((-1,3))
# print(verts.shape)
idx = ~np.isnan(verts).any(axis=1)
verts = verts[idx,:]
color = color_img.reshape((-1,3))
color = color[idx,:]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(verts)
pcd.colors = o3d.utility.Vector3dVector(color/255)
o3d.visualization.draw_geometries([pcd])
