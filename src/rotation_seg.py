import copy
import os

import cv2
import numpy as np
import open3d as o3d
from ultralytics import RTDETR, SAM, YOLO


class SAMSegmentEstimator:

    def __init__(self, config):

        # --- models ---
        self.seg_model_type = str(config.get("seg_model_type", "yolo")).lower()
        self.seg_model_path = config.get("seg_model_path", "/home/arnis/aims/python_env/edi_pose_estimation/src/rotation.pt")
        self.seg_conf = float(config.get("seg_conf", 0.01 if self.seg_model_type == "yolo" else 0.25))
        self.seg_iou = float(config.get("seg_iou", 0.7))
        self.seg_imgsz = int(config.get("seg_imgsz", 640))

        if self.seg_model_type == "sam":
            self.det_model = RTDETR(config.get("det_model_path", "rtdetr-x.pt"))
            self.seg_model = SAM(self.seg_model_path)
            self.bottle_cls = [
                idx for idx, name in self.det_model.names.items()
                if name in ['bottle', 'cup']
            ]
        else:
            self.det_model = None
            self.seg_model = YOLO(self.seg_model_path)
            self.bottle_cls = []

        # --- CAD / mesh reference ---
        self.model_path = config.get(
            'model_path',
            config.get('model_pcd', "/home/arnis/aims/python_env/edi_pose_estimation/asset/madara_white.obj")
        )
        if not self.model_path:
            raise ValueError(
                "Set model path in config['model_path'] or config['model_pcd']"
            )

        self.down_sample_size = 4.0  # 4 mm
        # Physical default: scale madara_white.obj to ~112 mm height.
        self.model_scale = float(config.get("model_scale", 112.0 / 135.0))
        self.icp_coarse_dist = float(config.get("icp_coarse_dist", 70.0))
        self.icp_fine_dist = float(config.get("icp_fine_dist", 20.0))
        self.icp_iters = int(config.get("icp_iterations", 120))
        self.min_points_for_pose = int(config.get("min_points_for_pose", 40))
        self.min_fitness = float(config.get("min_fitness", 0.05))
        self.max_rmse = float(config.get("max_rmse", 30.0))
        self.init_yaw_steps = int(config.get("init_yaw_steps", 8))
        self.init_roll_steps = int(config.get("init_roll_steps", 6))
        self.rmse_weight = float(config.get("rmse_weight", 0.01))
        self.iou_weight = float(config.get("iou_weight", 1.0))
        self.min_proj_iou = float(config.get("min_proj_iou", 0.05))
        self.min_mask_area = int(config.get("min_mask_area", 300))
        self.parallel_weight = float(config.get("parallel_weight", 0.25))
        self.neck_weight = float(config.get("neck_weight", 0.8))
        self.neck_sigma_px = float(config.get("neck_sigma_px", 45.0))
        self.axis_weight = float(config.get("axis_weight", 0.6))
        self.neck_dir_weight = float(config.get("neck_dir_weight", 0.9))
        self.table_ransac_dist = float(config.get("table_ransac_dist", 8.0))
        self.refine_yaw_deg = float(config.get("refine_yaw_deg", 60.0))
        self.refine_yaw_step_deg = float(config.get("refine_yaw_step_deg", 10.0))
        self.refine_xy_mm = float(config.get("refine_xy_mm", 15.0))
        self.refine_xy_step_mm = float(config.get("refine_xy_step_mm", 5.0))
        self.refine_z_mm = float(config.get("refine_z_mm", 40.0))
        self.refine_z_step_mm = float(config.get("refine_z_step_mm", 8.0))
        target_xy = config.get("target_point_xy", None)
        self.target_point_xy = tuple(target_xy) if target_xy is not None else None
        self.depth_scale = float(config.get("depth_scale", 1.0))
        self.completion_dist_thresh = float(config.get("completion_dist_thresh", 4.0))
        self.completion_attach_to_mask = bool(config.get("completion_attach_to_mask", True))
        self.completion_attach_dist_thresh = float(config.get("completion_attach_dist_thresh", 8.0))
        self.completion_edge_side = str(config.get("completion_edge_side", "front")).lower()
        self.completion_attach_axis = str(config.get("completion_attach_axis", "xyz")).lower()
        self.completion_attach_y_sign = float(config.get("completion_attach_y_sign", -1.0))
        self.completion_attach_y_gain = float(config.get("completion_attach_y_gain", 1.0))
        self.completion_attach_y_max_mm = float(config.get("completion_attach_y_max_mm", 1000.0))
        self.completion_above_mask_mm = float(config.get("completion_above_mask_mm", 0.0))
        self.completion_low_y_bias_mm = float(config.get("completion_low_y_bias_mm", 1.0))
        self.completion_contact_glue = bool(config.get("completion_contact_glue", True))
        self.completion_contact_percentile = float(config.get("completion_contact_percentile", 5.0))
        self.completion_depth_glue = bool(config.get("completion_depth_glue", True))
        self.completion_depth_glue_knn = int(config.get("completion_depth_glue_knn", 160))
        self.completion_force_lower_edge_glue = bool(config.get("completion_force_lower_edge_glue", True))
        self.completion_force_mode_lowest_point_glue = bool(config.get("completion_force_mode_lowest_point_glue", True))
        self.completion_prune_added_clusters = bool(config.get("completion_prune_added_clusters", False))
        self.completion_remove_sparse_regions = bool(config.get("completion_remove_sparse_regions", True))
        self.completion_sparse_eps = float(config.get("completion_sparse_eps", 12.0))
        self.completion_sparse_min_points = int(config.get("completion_sparse_min_points", 25))
        self.completion_sparse_keep_top_k = int(config.get("completion_sparse_keep_top_k", 1))
        self.completion_sparse_stat_nb = int(config.get("completion_sparse_stat_nb", 20))
        self.completion_sparse_stat_std = float(config.get("completion_sparse_stat_std", 2.0))
        self.completion_align_height_to_mask = bool(config.get("completion_align_height_to_mask", True))
        self.completion_height_percentile = float(config.get("completion_height_percentile", 5.0))
        self.completion_scale_to_mask_2d = bool(config.get("completion_scale_to_mask_2d", False))
        self.completion_scale_min = float(config.get("completion_scale_min", 0.5))
        self.completion_scale_max = float(config.get("completion_scale_max", 1.2))
        self.completion_proj_dilate = int(config.get("completion_proj_dilate", 3))
        self.completion_fit_inside_mask_2d = bool(config.get("completion_fit_inside_mask_2d", True))
        self.completion_fit_outside_tol = float(config.get("completion_fit_outside_tol", 0.02))
        self.completion_fit_shrink = float(config.get("completion_fit_shrink", 0.97))
        self.completion_fit_max_iter = int(config.get("completion_fit_max_iter", 24))
        self.completion_fit_allow_shrink = bool(config.get("completion_fit_allow_shrink", False))
        self.completion_match_size_to_observed = bool(config.get("completion_match_size_to_observed", False))
        self.completion_match_size_axes = str(config.get("completion_match_size_axes", "xy")).lower()
        self.completion_match_size_min = float(config.get("completion_match_size_min", 0.6))
        self.completion_match_size_max = float(config.get("completion_match_size_max", 2.2))
        self.completion_snap_xy_to_observed = bool(config.get("completion_snap_xy_to_observed", True))

        self.ref_pcd = self._load_reference_pcd(self.model_path)
        self.ref_mean, self.ref_axes = self._pca_frame(np.asarray(self.ref_pcd.points))
        self.ref_axis, self.ref_neck_sign = self._estimate_axis_and_neck(
            np.asarray(self.ref_pcd.points)
        )
        self.ref_neck_tip = self._estimate_neck_tip(
            np.asarray(self.ref_pcd.points), self.ref_axis, self.ref_neck_sign
        )

    def _load_reference_pcd(self, model_path):
        ext = os.path.splitext(model_path)[1].lower()

        # Mesh formats (OBJ/STL/etc.) are sampled into a point cloud for ICP.
        if ext in {'.obj', '.stl', '.off', '.gltf', '.glb', '.fbx'}:
            mesh = o3d.io.read_triangle_mesh(model_path)
            if mesh.is_empty() or len(mesh.triangles) == 0:
                raise ValueError(f"Cannot read mesh or mesh is empty: {model_path}")
            pcd = mesh.sample_points_uniformly(number_of_points=20000)
            if self.model_scale != 1.0:
                pcd.scale(self.model_scale, center=(0, 0, 0))
            return pcd.voxel_down_sample(self.down_sample_size)

        # Point-cloud inputs (PLY/PCD/XYZ/etc.)
        pcd = o3d.io.read_point_cloud(model_path)
        if pcd.is_empty():
            raise ValueError(
                f"Cannot read point cloud or point cloud is empty: {model_path}"
            )
        if self.model_scale != 1.0:
            pcd.scale(self.model_scale, center=(0, 0, 0))
        return pcd.voxel_down_sample(self.down_sample_size)

    # ============================================================
    # ICP pose estimation
    # ============================================================

    def _pca_frame(self, points):
        if points.shape[0] < 3:
            return None, None
        mean = np.mean(points, axis=0)
        centered = points - mean
        cov = centered.T @ centered / max(points.shape[0] - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        axes = vecs[:, order]
        # Right-handed frame.
        if np.linalg.det(axes) < 0:
            axes[:, 2] *= -1.0
        return mean, axes

    def _rot_axis_angle(self, axis, angle):
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        kx, ky, kz = axis
        c = np.cos(angle)
        s = np.sin(angle)
        v = 1.0 - c
        return np.array(
            [
                [kx * kx * v + c, kx * ky * v - kz * s, kx * kz * v + ky * s],
                [ky * kx * v + kz * s, ky * ky * v + c, ky * kz * v - kx * s],
                [kz * kx * v - ky * s, kz * ky * v + kx * s, kz * kz * v + c],
            ],
            dtype=np.float64,
        )

    def _rotation_from_two_vectors(self, a, b):
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        v = np.cross(a, b)
        c = float(np.dot(a, b))
        s = np.linalg.norm(v)
        if s < 1e-10:
            if c > 0:
                return np.eye(3, dtype=np.float64)
            # 180 deg: rotate around any axis orthogonal to a.
            tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(np.dot(tmp, a)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = np.cross(a, tmp)
            return self._rot_axis_angle(axis, np.pi)
        vx = np.array(
            [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
            dtype=np.float64,
        )
        r = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
        return r

    def _estimate_axis_and_neck(self, points):
        if points.shape[0] < 30:
            return None, 1.0

        _, axes = self._pca_frame(points)
        if axes is None:
            return None, 1.0

        axis = axes[:, 0]
        mean = np.mean(points, axis=0)
        proj = (points - mean) @ axis
        p_low = np.percentile(proj, 10.0)
        p_high = np.percentile(proj, 90.0)

        low_pts = points[proj <= p_low]
        high_pts = points[proj >= p_high]
        if low_pts.shape[0] < 10 or high_pts.shape[0] < 10:
            return axis, 1.0

        def end_radius(pts_end):
            pe = (pts_end - mean) @ axis
            closest = mean + np.outer(pe, axis)
            rad = np.linalg.norm(pts_end - closest, axis=1)
            return float(np.median(rad))

        r_low = end_radius(low_pts)
        r_high = end_radius(high_pts)
        # Neck side is the narrower end.
        neck_sign = -1.0 if r_low < r_high else 1.0
        return axis, neck_sign

    def _estimate_neck_tip(self, points, axis, neck_sign):
        if points.shape[0] < 20 or axis is None:
            return np.mean(points, axis=0)
        mean = np.mean(points, axis=0)
        proj = (points - mean) @ axis
        if neck_sign > 0:
            q = np.percentile(proj, 92.0)
            tip_pts = points[proj >= q]
        else:
            q = np.percentile(proj, 8.0)
            tip_pts = points[proj <= q]
        if tip_pts.shape[0] == 0:
            return mean
        return np.mean(tip_pts, axis=0)

    def _estimate_mask_neck_point_2d(self, mask_bool):
        ys, xs = np.where(mask_bool)
        if xs.size < 50:
            return None

        pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = centered.T @ centered / max(pts.shape[0] - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, np.argmax(vals)]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        proj = centered @ axis

        p_lo = np.percentile(proj, 10.0)
        p_hi = np.percentile(proj, 90.0)
        lo_pts = pts[proj <= p_lo]
        hi_pts = pts[proj >= p_hi]
        if lo_pts.shape[0] < 10 or hi_pts.shape[0] < 10:
            return tuple(mean.tolist())

        n = np.array([-axis[1], axis[0]], dtype=np.float64)
        lo_w = np.median(np.abs((lo_pts - lo_pts.mean(axis=0)) @ n))
        hi_w = np.median(np.abs((hi_pts - hi_pts.mean(axis=0)) @ n))
        neck_pts = lo_pts if lo_w < hi_w else hi_pts
        neck = neck_pts.mean(axis=0)
        return (float(neck[0]), float(neck[1]))

    def _estimate_mask_axis_2d(self, mask_bool):
        ys, xs = np.where(mask_bool)
        if xs.size < 50:
            return None
        pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = centered.T @ centered / max(pts.shape[0] - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, np.argmax(vals)]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis

    def _estimate_mask_neck_dir_2d(self, mask_bool, mask_neck_px):
        if mask_neck_px is None:
            return None
        ys, xs = np.where(mask_bool)
        if xs.size < 50:
            return None
        cx = float(xs.mean())
        cy = float(ys.mean())
        v = np.array([mask_neck_px[0] - cx, mask_neck_px[1] - cy], dtype=np.float64)
        n = np.linalg.norm(v)
        if n < 1e-6:
            return None
        return v / n

    def _clean_mask(self, mask):
        mask_u8 = mask.astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if n_labels <= 1:
            return mask_u8.astype(bool)

        # Ignore background (label 0), keep largest valid component.
        best_label = -1
        best_area = -1
        for lbl in range(1, n_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area >= self.min_mask_area and area > best_area:
                best_area = area
                best_label = lbl

        if best_label < 0:
            return mask_u8.astype(bool)
        return (labels == best_label)

    def _mask_centroid(self, mask_bool):
        ys, xs = np.where(mask_bool)
        if xs.size == 0:
            return None
        return (float(xs.mean()), float(ys.mean()))

    def _projected_mask_iou(self, pose, K, mask_bool):
        model = copy.deepcopy(self.ref_pcd)
        model.transform(pose)
        pts = np.asarray(model.points)
        if pts.shape[0] == 0:
            return 0.0

        h, w = mask_bool.shape
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        z = pts[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            return 0.0
        pts = pts[valid]
        z = z[valid]

        u = (fx * pts[:, 0] / z + cx).astype(np.int32)
        v = (fy * pts[:, 1] / z + cy).astype(np.int32)
        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(in_img):
            return 0.0
        u = u[in_img]
        v = v[in_img]

        proj = np.zeros((h, w), dtype=np.uint8)
        proj[v, u] = 1
        proj = cv2.dilate(proj, np.ones((3, 3), np.uint8), iterations=1).astype(bool)

        inter = np.logical_and(mask_bool, proj).sum()
        union = np.logical_or(mask_bool, proj).sum()
        if union == 0:
            return 0.0
        return float(inter) / float(union)

    def _project_ref_neck_point(self, pose, K, mask_shape):
        if self.ref_neck_tip is None:
            return None
        p = pose[:3, :3] @ self.ref_neck_tip + pose[:3, 3]
        z = float(p[2])
        if z <= 1e-6:
            return None
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        u = fx * p[0] / z + cx
        v = fy * p[1] / z + cy
        h, w = mask_shape
        if u < 0 or u >= w or v < 0 or v >= h:
            return None
        return (float(u), float(v))

    def _project_ref_axis_2d(self, pose, K, mask_shape):
        if self.ref_axis is None:
            return None
        base = pose[:3, 3]
        v3 = pose[:3, :3] @ (self.ref_axis * self.ref_neck_sign)
        p1 = base + v3 * 40.0
        p2 = base - v3 * 40.0
        pts = np.stack([p1, p2], axis=0)

        h, w = mask_shape
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        z = pts[:, 2]
        if np.any(z <= 1e-6):
            return None
        u = fx * pts[:, 0] / z + cx
        v = fy * pts[:, 1] / z + cy
        if np.any(u < 0) or np.any(u >= w) or np.any(v < 0) or np.any(v >= h):
            return None
        d2 = np.array([u[0] - u[1], v[0] - v[1]], dtype=np.float64)
        n = np.linalg.norm(d2)
        if n < 1e-6:
            return None
        return d2 / n

    def _axis_alignment_score(self, pose, K, mask_shape, mask_axis_2d):
        if mask_axis_2d is None:
            return 0.0
        proj_axis = self._project_ref_axis_2d(pose, K, mask_shape)
        if proj_axis is None:
            return -1.0
        # Direction sign is ambiguous in silhouette; use absolute cosine.
        return float(abs(np.dot(proj_axis, mask_axis_2d)))

    def _project_model_center_2d(self, pose, K, mask_shape):
        p = pose[:3, 3]
        z = float(p[2])
        if z <= 1e-6:
            return None
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        u = fx * p[0] / z + cx
        v = fy * p[1] / z + cy
        h, w = mask_shape
        if u < 0 or u >= w or v < 0 or v >= h:
            return None
        return (float(u), float(v))

    def _neck_dir_alignment_score(self, pose, K, mask_shape, mask_neck_dir_2d):
        if mask_neck_dir_2d is None:
            return 0.0
        neck_px = self._project_ref_neck_point(pose, K, mask_shape)
        ctr_px = self._project_model_center_2d(pose, K, mask_shape)
        if neck_px is None or ctr_px is None:
            return -1.0
        v = np.array([neck_px[0] - ctr_px[0], neck_px[1] - ctr_px[1]], dtype=np.float64)
        n = np.linalg.norm(v)
        if n < 1e-6:
            return -1.0
        v = v / n
        # Signed cosine: neck must point to the same side.
        return float(np.dot(v, mask_neck_dir_2d))

    def _align_pose_neck_direction_2d(self, pose, K, mask_shape, mask_neck_dir_2d, table_normal):
        if mask_neck_dir_2d is None:
            return pose.copy()

        ctr_px = self._project_model_center_2d(pose, K, mask_shape)
        neck_px = self._project_ref_neck_point(pose, K, mask_shape)
        if ctr_px is None or neck_px is None:
            return pose.copy()

        v = np.array([neck_px[0] - ctr_px[0], neck_px[1] - ctr_px[1]], dtype=np.float64)
        n = np.linalg.norm(v)
        if n < 1e-6:
            return pose.copy()
        v = v / n

        # Signed 2D angle from model neck direction to mask neck direction.
        cross_z = v[0] * mask_neck_dir_2d[1] - v[1] * mask_neck_dir_2d[0]
        dot = float(np.clip(np.dot(v, mask_neck_dir_2d), -1.0, 1.0))
        ang = np.arctan2(cross_z, dot)

        p = pose.copy()
        r = self._rot_axis_angle(table_normal, ang)
        p[:3, :3] = r @ p[:3, :3]
        return p

    def _neck_alignment_score(self, pose, K, mask_shape, mask_neck_px):
        if mask_neck_px is None:
            return 0.0
        proj = self._project_ref_neck_point(pose, K, mask_shape)
        if proj is None:
            return -1.0
        dx = proj[0] - mask_neck_px[0]
        dy = proj[1] - mask_neck_px[1]
        dist = np.hypot(dx, dy)
        # 1 near-perfect, 0 far away.
        return float(np.exp(-(dist * dist) / (2.0 * self.neck_sigma_px * self.neck_sigma_px)))

    def _snap_pose_xy_to_neck(self, pose, K, mask_shape, mask_neck_px):
        if mask_neck_px is None:
            return pose.copy()
        proj_neck = self._project_ref_neck_point(pose, K, mask_shape)
        if proj_neck is None:
            return pose.copy()

        neck_3d = pose[:3, :3] @ self.ref_neck_tip + pose[:3, 3]
        z = float(neck_3d[2])
        if z <= 1e-6:
            return pose.copy()

        fx, fy = float(K[0, 0]), float(K[1, 1])
        du = float(mask_neck_px[0] - proj_neck[0])
        dv = float(mask_neck_px[1] - proj_neck[1])
        dx = du * z / fx
        dy = dv * z / fy

        out = pose.copy()
        out[:3, 3] = out[:3, 3] + np.array([dx, dy, 0.0], dtype=np.float64)
        return out

    def _estimate_table_normal(self, scene_pts):
        valid = ~np.isnan(scene_pts).any(axis=1)
        scene = scene_pts[valid]
        if scene.shape[0] < 500:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scene)
        pcd = pcd.voxel_down_sample(max(self.down_sample_size, 4.0))
        if len(pcd.points) < 100:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)

        try:
            plane, _ = pcd.segment_plane(
                distance_threshold=self.table_ransac_dist,
                ransac_n=3,
                num_iterations=400,
            )
            n = np.asarray(plane[:3], dtype=np.float64)
            n = n / (np.linalg.norm(n) + 1e-12)
            # Keep normal roughly facing camera.
            if n[2] < 0:
                n = -n
            return n
        except Exception:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def _pose_from_mask_geometry(self, obj_pts, table_normal):
        obs_axis, obs_neck_sign = self._estimate_axis_and_neck(obj_pts)
        if obs_axis is None or self.ref_axis is None:
            return None

        # Bottle axis should be parallel to table: project observed axis onto table plane.
        obs_axis_plane = obs_axis - np.dot(obs_axis, table_normal) * table_normal
        if np.linalg.norm(obs_axis_plane) < 1e-6:
            obs_axis_plane = obs_axis
        obs_axis_plane = obs_axis_plane / (np.linalg.norm(obs_axis_plane) + 1e-12)

        obs_neck_dir = obs_axis_plane * obs_neck_sign
        ref_neck_dir = self.ref_axis * self.ref_neck_sign
        rot = self._rotation_from_two_vectors(ref_neck_dir, obs_neck_dir)

        obs_neck_tip = self._estimate_neck_tip(obj_pts, obs_axis_plane, obs_neck_sign)
        ref_neck_tip_rot = rot @ self.ref_neck_tip
        t = obs_neck_tip - ref_neck_tip_rot

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rot
        pose[:3, 3] = t
        return pose, obs_neck_dir

    def _align_axis_parallel_to_table(self, pose, table_normal, obs_neck_dir):
        if self.ref_axis is None:
            return pose.copy()

        p = pose.copy()
        neck_ref = self.ref_axis * self.ref_neck_sign
        cur_dir = p[:3, :3] @ neck_ref
        cur_dir = cur_dir / (np.linalg.norm(cur_dir) + 1e-12)

        # Target direction: observed neck direction projected to table plane.
        tgt = obs_neck_dir - np.dot(obs_neck_dir, table_normal) * table_normal
        if np.linalg.norm(tgt) < 1e-6:
            # Fallback: project current direction to plane.
            tgt = cur_dir - np.dot(cur_dir, table_normal) * table_normal
        if np.linalg.norm(tgt) < 1e-6:
            return p
        tgt = tgt / (np.linalg.norm(tgt) + 1e-12)

        r = self._rotation_from_two_vectors(cur_dir, tgt)
        p[:3, :3] = r @ p[:3, :3]
        return p

    def _refine_pose_with_projection(
        self,
        pose,
        K,
        mask_bool,
        table_normal,
        obs_neck_dir,
        mask_neck_px,
        mask_axis_2d,
        mask_neck_dir_2d,
    ):
        h, w = mask_bool.shape
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        m = cv2.moments(mask_u8, binaryImage=True)
        if m["m00"] <= 1.0:
            return pose.copy(), 0.0, 0.0, 0.0, 0.0, 0.0
        mask_cx = float(m["m10"] / m["m00"])
        mask_cy = float(m["m01"] / m["m00"])

        mask_edges = cv2.Canny(mask_u8, 60, 120)
        out_mask = (mask_u8 == 0).astype(np.uint8)
        dist_out = cv2.distanceTransform(out_mask, cv2.DIST_L2, 3)

        fx, fy = float(K[0, 0]), float(K[1, 1])
        table_normal = table_normal / (np.linalg.norm(table_normal) + 1e-12)

        best_pose = pose.copy()
        best_iou = self._projected_mask_iou(best_pose, K, mask_bool)
        best_obj = -1e9
        best_yaw_deg = 0.0
        best_dx = 0.0
        best_dy = 0.0
        best_dz = 0.0

        yaw_vals = np.arange(-180.0, 180.0 + 1e-6, 6.0, dtype=np.float64)
        z_vals = np.arange(
            -max(self.refine_z_mm * 2.0, 80.0),
            max(self.refine_z_mm * 2.0, 80.0) + 1e-6,
            max(self.refine_z_step_mm, 6.0),
            dtype=np.float64,
        )

        base_pose = pose.copy()
        for yaw_deg in yaw_vals:
            rdelta = self._rot_axis_angle(table_normal, np.deg2rad(float(yaw_deg)))
            yaw_pose = base_pose.copy()
            yaw_pose[:3, :3] = rdelta @ base_pose[:3, :3]

            for dz in z_vals:
                cand = yaw_pose.copy()
                cand[:3, 3] = yaw_pose[:3, 3] + np.array([0.0, 0.0, float(dz)], dtype=np.float64)

                # Hard align by neck first; if neck unavailable then use centroid fallback.
                if mask_neck_px is not None:
                    before = cand.copy()
                    cand = self._snap_pose_xy_to_neck(cand, K, (h, w), mask_neck_px)
                    dx = float(cand[0, 3] - before[0, 3])
                    dy = float(cand[1, 3] - before[1, 3])
                else:
                    ctr = self._project_model_center_2d(cand, K, (h, w))
                    zc = float(cand[2, 3])
                    if ctr is None or zc <= 1e-6:
                        continue
                    du = mask_cx - ctr[0]
                    dv = mask_cy - ctr[1]
                    dx = du * zc / fx
                    dy = dv * zc / fy
                    cand[:3, 3] = cand[:3, 3] + np.array([dx, dy, 0.0], dtype=np.float64)

                model = copy.deepcopy(self.ref_pcd)
                model.transform(cand)
                pts = np.asarray(model.points)
                if pts.shape[0] == 0:
                    continue

                z = pts[:, 2]
                valid = z > 1e-6
                if not np.any(valid):
                    continue
                pts = pts[valid]
                z = z[valid]
                u = (fx * pts[:, 0] / z + float(K[0, 2])).astype(np.int32)
                v = (fy * pts[:, 1] / z + float(K[1, 2])).astype(np.int32)
                in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                if not np.any(in_img):
                    continue
                u = u[in_img]
                v = v[in_img]

                proj = np.zeros((h, w), dtype=np.uint8)
                proj[v, u] = 255
                proj = cv2.dilate(proj, np.ones((3, 3), np.uint8), iterations=1)

                inter = np.logical_and(mask_u8 > 0, proj > 0).sum()
                union = np.logical_or(mask_u8 > 0, proj > 0).sum()
                if union == 0:
                    continue
                iou = float(inter) / float(union)

                proj_edges = cv2.Canny(proj, 60, 120)
                ey, ex = np.where(proj_edges > 0)
                if ex.size > 0:
                    contour_penalty = float(np.mean(dist_out[ey, ex]))
                else:
                    contour_penalty = 50.0

                neck_score = self._neck_alignment_score(cand, K, (h, w), mask_neck_px)
                axis_score = self._axis_alignment_score(cand, K, (h, w), mask_axis_2d)
                neck_dir_score = self._neck_dir_alignment_score(cand, K, (h, w), mask_neck_dir_2d)

                obj = (
                    3.0 * iou
                    - 0.03 * contour_penalty
                    + 1.8 * neck_score
                    + 0.6 * axis_score
                    + 1.0 * neck_dir_score
                )

                if obj > best_obj:
                    best_obj = obj
                    best_pose = cand
                    best_iou = iou
                    best_yaw_deg = float(yaw_deg)
                    best_dx = float(dx)
                    best_dy = float(dy)
                    best_dz = float(dz)

        return best_pose, best_iou, best_yaw_deg, best_dx, best_dy, best_dz

    def _build_init_candidates(self, pcd_ds):
        pts = np.asarray(pcd_ds.points)
        obs_mean, obs_axes = self._pca_frame(pts)
        obs_axis, obs_neck_sign = self._estimate_axis_and_neck(pts)
        if obs_mean is None or obs_axis is None or self.ref_axis is None:
            return []

        init_candidates = []

        # Explicitly align neck direction (narrow end -> narrow end).
        ref_neck_vec = self.ref_axis * self.ref_neck_sign
        obs_neck_vec = obs_axis * obs_neck_sign
        rot_neck = self._rotation_from_two_vectors(ref_neck_vec, obs_neck_vec)
        roll_axis = obs_neck_vec / (np.linalg.norm(obs_neck_vec) + 1e-12)
        base_rots = [rot_neck]

        # Fallback orientation from full PCA frame.
        if self.ref_axes is not None and obs_axes is not None:
            rot_pca = obs_axes @ self.ref_axes.T
            if np.linalg.det(rot_pca) < 0:
                rot_pca[:, 2] *= -1.0
            base_rots.append(rot_pca)

        roll_steps = max(1, self.init_roll_steps)
        for rb in base_rots:
            for i in range(roll_steps):
                a = (2.0 * np.pi * i) / float(roll_steps)
                rr = self._rot_axis_angle(roll_axis, a)
                r = rr @ rb
                init = np.eye(4, dtype=np.float64)
                init[:3, :3] = r
                init[:3, 3] = obs_mean
                init_candidates.append(init)

        # Also keep center-only init as fallback.
        init_center = np.eye(4, dtype=np.float64)
        init_center[:3, 3] = obs_mean
        init_candidates.append(init_center)
        return init_candidates

    def _run_icp_two_stage(self, source_pcd, target_pcd, init):
        reg_coarse = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            self.icp_coarse_dist,
            init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.icp_iters)
        )
        reg_fine = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            self.icp_fine_dist,
            reg_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.icp_iters)
        )
        return reg_fine

    def estimate_pose_for_mask(self, pts, color_img, mask, K=None):

        objpts = pts[mask, :]
        verts = objpts.reshape((-1, 3))
        valid = ~np.isnan(verts).any(axis=1)
        verts = verts[valid]

        if len(verts) < self.min_points_for_pose:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)

        pcd_ds = pcd.voxel_down_sample(self.down_sample_size)
        if len(pcd_ds.points) < self.min_points_for_pose:
            return None

        init_candidates = self._build_init_candidates(pcd_ds)
        if not init_candidates:
            return None

        best_reg = None
        best_score = -np.inf
        for init in init_candidates:
            reg = self._run_icp_two_stage(self.ref_pcd, pcd_ds, init)
            score = reg.fitness - self.rmse_weight * reg.inlier_rmse
            if K is not None:
                proj_iou = self._projected_mask_iou(reg.transformation, K, mask)
                score += self.iou_weight * proj_iou
            if score > best_score:
                best_score = score
                best_reg = reg

        return best_reg

    # ============================================================
    # Pointcloud completion
    # ============================================================

    def complete_pointcloud(self, obj_pts, reg_res, dist_thresh=0.004):
        # Keep only valid observed object points (from selected segmentation mask).
        valid = ~np.isnan(obj_pts).any(axis=1)
        obj_pts = obj_pts[valid]

        observed_pcd = o3d.geometry.PointCloud()
        observed_pcd.points = o3d.utility.Vector3dVector(obj_pts)

        # Transform CAD/model into estimated object pose.
        model = copy.deepcopy(self.ref_pcd)
        model.transform(reg_res.transformation)

        observed_tree = o3d.geometry.KDTreeFlann(observed_pcd)

        missing_pts = []

        for pt in np.asarray(model.points):
            [k, idx, dist] = observed_tree.search_knn_vector_3d(pt, 1)

            if k == 0 or np.sqrt(dist[0]) > dist_thresh:
                missing_pts.append(pt)

        missing_pcd = o3d.geometry.PointCloud()
        if len(missing_pts) > 0:
            missing_pcd.points = o3d.utility.Vector3dVector(
                np.array(missing_pts)
            )

        full_object = observed_pcd + missing_pcd
        return full_object

    def _edge_anchor_mode_top(self, pts):
        if pts.shape[0] == 0:
            return None
        z_ri = np.round(pts[:, 2]).astype(np.int32)
        vals, cnt = np.unique(z_ri, return_counts=True)
        z_mode = int(vals[np.argmax(cnt)])
        layer = pts[np.abs(pts[:, 2] - float(z_mode)) <= 0.5]
        if layer.shape[0] == 0:
            return pts[np.argmax(pts[:, 2])]
        # Camera Y axis points down in image projection:
        # - "front" side uses smallest Y
        # - "back" side uses largest Y (opposite side)
        if self.completion_edge_side == "front":
            return layer[np.argmin(layer[:, 1])]
        return layer[np.argmax(layer[:, 1])]

    def _mask_mode_highest_anchor(self, observed_pts):
        if observed_pts.shape[0] == 0:
            return None
        z_ri = np.round(observed_pts[:, 2]).astype(np.int32)
        vals, cnt = np.unique(z_ri, return_counts=True)
        z_mode = int(vals[np.argmax(cnt)])
        layer = observed_pts[np.abs(observed_pts[:, 2] - float(z_mode)) <= 0.5]
        if layer.shape[0] == 0:
            layer = observed_pts
        # Highest point in camera frame => minimal Y.
        return layer[np.argmin(layer[:, 1])]

    def _mask_mode_lowest_anchor(self, observed_pts):
        if observed_pts.shape[0] == 0:
            return None
        z_ri = np.round(observed_pts[:, 2]).astype(np.int32)
        vals, cnt = np.unique(z_ri, return_counts=True)
        z_mode = int(vals[np.argmax(cnt)])
        layer = observed_pts[np.abs(observed_pts[:, 2] - float(z_mode)) <= 0.5]
        if layer.shape[0] == 0:
            layer = observed_pts
        # Lowest point in camera frame => maximal Y.
        return layer[np.argmax(layer[:, 1])]

    def _object_mode_lowest_anchor(self, pts):
        if pts.shape[0] == 0:
            return None
        z_ri = np.round(pts[:, 2]).astype(np.int32)
        vals, cnt = np.unique(z_ri, return_counts=True)
        z_mode = int(vals[np.argmax(cnt)])
        layer = pts[np.abs(pts[:, 2] - float(z_mode)) <= 0.5]
        if layer.shape[0] == 0:
            layer = pts
        return layer[np.argmax(layer[:, 1])]

    def _object_lowest_anchor(self, pts):
        if pts.shape[0] == 0:
            return None
        return pts[np.argmax(pts[:, 1])]

    def _object_min_z_anchor(self, pts):
        if pts.shape[0] == 0:
            return None
        return pts[np.argmin(pts[:, 2])]

    def _lower_edge_anchor(self, pts):
        if pts.shape[0] == 0:
            return None
        y90 = float(np.percentile(pts[:, 1], 90.0))
        y95 = float(np.percentile(pts[:, 1], 95.0))
        strip = pts[pts[:, 1] >= y95]
        if strip.shape[0] < 8:
            strip = pts[pts[:, 1] >= y90]
        if strip.shape[0] == 0:
            strip = pts

        z_ri = np.round(strip[:, 2]).astype(np.int32)
        vals, cnt = np.unique(z_ri, return_counts=True)
        z_mode = int(vals[np.argmax(cnt)])
        layer = strip[np.abs(strip[:, 2] - float(z_mode)) <= 1.0]
        if layer.shape[0] == 0:
            layer = strip
        return np.array(
            [
                float(np.median(layer[:, 0])),
                float(np.max(layer[:, 1])),
                float(np.median(layer[:, 2])),
            ],
            dtype=np.float64,
        )

    def _mode_layer(self, pts):
        if pts.shape[0] == 0:
            return pts
        z_ri = np.round(pts[:, 2]).astype(np.int32)
        vals, cnt = np.unique(z_ri, return_counts=True)
        z_mode = int(vals[np.argmax(cnt)])
        layer = pts[np.abs(pts[:, 2] - float(z_mode)) <= 0.5]
        return layer if layer.shape[0] > 0 else pts

    def _edge_contact_translation_xyz(self, observed_pts, added_pts):
        if observed_pts.shape[0] == 0 or added_pts.shape[0] == 0:
            return np.zeros((3,), dtype=np.float64)

        obs_edge = self._mode_layer(observed_pts)
        add_edge = self._mode_layer(added_pts)

        if self.completion_edge_side == "front":
            oy = np.percentile(obs_edge[:, 1], 20.0)
            ay = np.percentile(add_edge[:, 1], 20.0)
            obs_edge = obs_edge[obs_edge[:, 1] <= oy]
            add_edge = add_edge[add_edge[:, 1] <= ay]
        else:
            oy = np.percentile(obs_edge[:, 1], 80.0)
            ay = np.percentile(add_edge[:, 1], 80.0)
            obs_edge = obs_edge[obs_edge[:, 1] >= oy]
            add_edge = add_edge[add_edge[:, 1] >= ay]

        if obs_edge.shape[0] == 0 or add_edge.shape[0] == 0:
            return np.zeros((3,), dtype=np.float64)

        obs_pcd = o3d.geometry.PointCloud()
        obs_pcd.points = o3d.utility.Vector3dVector(obs_edge)
        tree = o3d.geometry.KDTreeFlann(obs_pcd)

        best_i = -1
        best_j = -1
        best_d = np.inf
        for i, p in enumerate(add_edge):
            k, idx, dist = tree.search_knn_vector_3d(p, 1)
            if k > 0 and dist[0] < best_d:
                best_d = dist[0]
                best_i = i
                best_j = idx[0]
        if best_i < 0 or best_j < 0:
            return np.zeros((3,), dtype=np.float64)

        return obs_edge[best_j] - add_edge[best_i]

    def _constrain_translation_by_axis(self, t):
        t = np.asarray(t, dtype=np.float64).reshape(3)
        if self.completion_attach_axis == "x":
            return np.array([t[0], 0.0, 0.0], dtype=np.float64)
        if self.completion_attach_axis == "y":
            dy = float(t[1]) * self.completion_attach_y_gain * self.completion_attach_y_sign
            dy = float(np.clip(dy, -self.completion_attach_y_max_mm, self.completion_attach_y_max_mm))
            return np.array([0.0, dy, 0.0], dtype=np.float64)
        if self.completion_attach_axis == "z":
            return np.array([0.0, 0.0, t[2]], dtype=np.float64)
        return np.array([t[0], t[1], t[2]], dtype=np.float64)

    def _project_points_to_binary_mask(self, pts, K, shape_hw, dilate=3):
        h, w = int(shape_hw[0]), int(shape_hw[1])
        out = np.zeros((h, w), dtype=np.uint8)
        if pts.shape[0] == 0:
            return out.astype(bool)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        z = pts[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            return out.astype(bool)
        p = pts[valid]
        z = p[:, 2]
        u = np.round(fx * p[:, 0] / z + cx).astype(np.int32)
        v = np.round(fy * p[:, 1] / z + cy).astype(np.int32)
        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if np.any(in_img):
            out[v[in_img], u[in_img]] = 1
        if int(dilate) > 0:
            out = cv2.dilate(out, np.ones((int(dilate), int(dilate)), np.uint8), iterations=1)
        return out.astype(bool)

    def _scale_and_center_completion_to_mask_2d(self, completed_pts, mask_bool, K):
        if completed_pts.shape[0] == 0:
            return completed_pts
        ys_m, xs_m = np.where(mask_bool)
        if xs_m.size == 0:
            return completed_pts

        h, w = mask_bool.shape
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        mw = float(xs_m.max() - xs_m.min() + 1)
        mh = float(ys_m.max() - ys_m.min() + 1)

        proj = self._project_points_to_binary_mask(completed_pts, K, (h, w), dilate=self.completion_proj_dilate)
        ys_p, xs_p = np.where(proj)
        if xs_p.size == 0:
            return completed_pts
        pw = float(xs_p.max() - xs_p.min() + 1)
        ph = float(ys_p.max() - ys_p.min() + 1)
        if pw <= 1e-6 or ph <= 1e-6:
            return completed_pts

        s = 0.5 * (mw / pw + mh / ph)
        s = float(np.clip(s, self.completion_scale_min, self.completion_scale_max))
        center = np.mean(completed_pts, axis=0)
        pts = (completed_pts - center) * s + center

        proj2 = self._project_points_to_binary_mask(pts, K, (h, w), dilate=self.completion_proj_dilate)
        ys2, xs2 = np.where(proj2)
        if xs2.size == 0:
            return pts

        cmx, cmy = float(xs_m.mean()), float(ys_m.mean())
        cpx, cpy = float(xs2.mean()), float(ys2.mean())
        du, dv = (cmx - cpx), (cmy - cpy)
        zref = float(np.percentile(pts[:, 2], 50))
        dx = du * zref / fx
        dy = dv * zref / fy
        pts[:, 0] += dx
        pts[:, 1] += dy
        return pts

    def _project_points_uv(self, pts, K, shape_hw):
        h, w = int(shape_hw[0]), int(shape_hw[1])
        if pts.shape[0] == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        z = pts[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        p = pts[valid]
        z = p[:, 2]
        u = np.round(fx * p[:, 0] / z + cx).astype(np.int32)
        v = np.round(fy * p[:, 1] / z + cy).astype(np.int32)
        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(in_img):
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        return u[in_img], v[in_img]

    def _fit_points_inside_mask_2d(self, pts, mask_bool, K):
        if pts.shape[0] == 0:
            return pts
        ys_m, xs_m = np.where(mask_bool)
        if xs_m.size == 0:
            return pts
        h, w = mask_bool.shape
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cmx, cmy = float(xs_m.mean()), float(ys_m.mean())

        out = pts.copy()
        shrink = float(np.clip(self.completion_fit_shrink, 0.8, 0.995))
        outside_tol = float(np.clip(self.completion_fit_outside_tol, 0.0, 0.5))
        allow_shrink = bool(self.completion_fit_allow_shrink)
        for _ in range(max(int(self.completion_fit_max_iter), 1)):
            u, v = self._project_points_uv(out, K, (h, w))
            if u.size == 0:
                break
            inside = mask_bool[v, u]
            outside_ratio = 1.0 - float(np.mean(inside))
            if outside_ratio <= outside_tol:
                break

            # Keep depth geometry intact: optionally shrink only in X/Y.
            if allow_shrink:
                center = np.mean(out, axis=0)
                out_xy = (out[:, :2] - center[:2]) * shrink + center[:2]
                out[:, :2] = out_xy

            u2, v2 = self._project_points_uv(out, K, (h, w))
            if u2.size == 0:
                continue
            cpx, cpy = float(np.mean(u2)), float(np.mean(v2))
            du, dv = (cmx - cpx), (cmy - cpy)
            zref = float(np.percentile(out[:, 2], 50))
            out[:, 0] += du * zref / fx
            out[:, 1] += dv * zref / fy
        return out

    def _match_added_extent_to_observed(self, added_pts, observed_pts):
        if added_pts.shape[0] < 10 or observed_pts.shape[0] < 10:
            return added_pts
        # Robust extents (trim outliers).
        lo_o = np.percentile(observed_pts, 5.0, axis=0)
        hi_o = np.percentile(observed_pts, 95.0, axis=0)
        lo_a = np.percentile(added_pts, 5.0, axis=0)
        hi_a = np.percentile(added_pts, 95.0, axis=0)
        ext_o = hi_o - lo_o
        ext_a = hi_a - lo_a
        ratios = []
        axes = self.completion_match_size_axes
        if "x" in axes and ext_a[0] > 1e-6:
            ratios.append(ext_o[0] / ext_a[0])
        if "y" in axes and ext_a[1] > 1e-6:
            ratios.append(ext_o[1] / ext_a[1])
        if "z" in axes and ext_a[2] > 1e-6:
            ratios.append(ext_o[2] / ext_a[2])
        if len(ratios) == 0:
            return added_pts
        s = float(np.median(np.asarray(ratios, dtype=np.float64)))
        s = float(np.clip(s, self.completion_match_size_min, self.completion_match_size_max))
        c = np.mean(added_pts, axis=0)
        out = (added_pts - c) * s + c
        return out

    def _snap_added_depth_to_observed(self, added_pts, observed_pts):
        if added_pts.shape[0] == 0 or observed_pts.shape[0] == 0:
            return added_pts
        obs_pcd = o3d.geometry.PointCloud()
        obs_pcd.points = o3d.utility.Vector3dVector(observed_pts)
        tree = o3d.geometry.KDTreeFlann(obs_pcd)
        dz = []
        d = []
        for p in added_pts:
            k, idx, dist = tree.search_knn_vector_3d(p, 1)
            if k > 0:
                d.append(np.sqrt(dist[0]))
                dz.append(observed_pts[idx[0], 2] - p[2])
        if len(dz) == 0:
            return added_pts
        d = np.asarray(d, dtype=np.float64)
        dz = np.asarray(dz, dtype=np.float64)
        keep = d <= np.percentile(d, 20.0)
        if not np.any(keep):
            keep = np.ones_like(d, dtype=bool)
        out = added_pts.copy()
        out[:, 2] += float(np.median(dz[keep]))
        return out

    def _optimize_added_z_by_nn(self, added_pts, observed_pts):
        if added_pts.shape[0] == 0 or observed_pts.shape[0] == 0:
            return added_pts
        obs_pcd = o3d.geometry.PointCloud()
        obs_pcd.points = o3d.utility.Vector3dVector(observed_pts)
        tree = o3d.geometry.KDTreeFlann(obs_pcd)

        def score_for_offset(dz):
            d = []
            for p in added_pts:
                q = np.array([p[0], p[1], p[2] + dz], dtype=np.float64)
                k, idx, dist = tree.search_knn_vector_3d(q, 1)
                if k > 0:
                    d.append(np.sqrt(dist[0]))
            if len(d) == 0:
                return np.inf
            d = np.asarray(d, dtype=np.float64)
            return float(np.percentile(d, 50.0))

        best_dz = 0.0
        best_score = score_for_offset(0.0)
        for dz in np.linspace(-40.0, 40.0, 81):
            s = score_for_offset(float(dz))
            if s < best_score:
                best_score = s
                best_dz = float(dz)
        out = added_pts.copy()
        out[:, 2] += best_dz
        return out

    def _snap_added_xyz_to_observed(self, added_pts, observed_pts):
        if added_pts.shape[0] == 0 or observed_pts.shape[0] == 0:
            return added_pts
        obs_pcd = o3d.geometry.PointCloud()
        obs_pcd.points = o3d.utility.Vector3dVector(observed_pts)
        tree = o3d.geometry.KDTreeFlann(obs_pcd)
        vecs = []
        d = []
        for p in added_pts:
            k, idx, dist = tree.search_knn_vector_3d(p, 1)
            if k > 0:
                q = observed_pts[idx[0]]
                vecs.append(q - p)
                d.append(np.sqrt(dist[0]))
        if len(vecs) == 0:
            return added_pts
        vecs = np.asarray(vecs, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        keep = d <= np.percentile(d, 20.0)
        if not np.any(keep):
            keep = np.ones_like(d, dtype=bool)
        t = np.median(vecs[keep], axis=0)
        out = added_pts.copy()
        out += t
        return out

    def _snap_added_xy_to_observed(self, added_pts, observed_pts):
        if added_pts.shape[0] == 0 or observed_pts.shape[0] == 0:
            return added_pts
        obs_pcd = o3d.geometry.PointCloud()
        obs_pcd.points = o3d.utility.Vector3dVector(observed_pts)
        tree = o3d.geometry.KDTreeFlann(obs_pcd)
        vecs = []
        d = []
        for p in added_pts:
            k, idx, dist = tree.search_knn_vector_3d(p, 1)
            if k > 0:
                q = observed_pts[idx[0]]
                vecs.append(q - p)
                d.append(np.sqrt(dist[0]))
        if len(vecs) == 0:
            return added_pts
        vecs = np.asarray(vecs, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        keep = d <= np.percentile(d, 20.0)
        if not np.any(keep):
            keep = np.ones_like(d, dtype=bool)
        txy = np.median(vecs[keep, :2], axis=0)
        out = added_pts.copy()
        out[:, 0] += float(txy[0])
        out[:, 1] += float(txy[1])
        return out

    def _align_added_height_to_mask(self, added_pts, observed_pts):
        if added_pts.shape[0] == 0 or observed_pts.shape[0] == 0:
            return added_pts
        p = float(np.clip(self.completion_height_percentile, 0.1, 20.0))
        # In camera frame, smaller Y is higher in image.
        y_obs_top = float(np.percentile(observed_pts[:, 1], p))
        y_add_top = float(np.percentile(added_pts[:, 1], p))
        out = added_pts.copy()
        out[:, 1] += (y_obs_top - y_add_top)
        return out

    def _approx_mask_depth_at_xy(self, observed_pts, x, y):
        if observed_pts.shape[0] < 6:
            return None
        xy = observed_pts[:, :2]
        d2 = (xy[:, 0] - float(x)) ** 2 + (xy[:, 1] - float(y)) ** 2
        k = int(np.clip(self.completion_depth_glue_knn, 6, observed_pts.shape[0]))
        idx = np.argpartition(d2, k - 1)[:k]
        nbr = observed_pts[idx]
        if nbr.shape[0] < 6:
            return None

        # Local depth approximation by weighted plane: z = ax + by + c.
        A = np.column_stack((nbr[:, 0], nbr[:, 1], np.ones((nbr.shape[0],), dtype=np.float64)))
        z = nbr[:, 2]
        w = 1.0 / (np.sqrt(d2[idx]) + 1.0)
        Aw = A * w[:, None]
        zw = z * w
        try:
            coef, _, _, _ = np.linalg.lstsq(Aw, zw, rcond=None)
            z_hat = float(coef[0] * float(x) + coef[1] * float(y) + coef[2])
        except Exception:
            z_hat = float(np.median(z))

        z_lo = float(np.percentile(z, 5.0))
        z_hi = float(np.percentile(z, 95.0))
        return float(np.clip(z_hat, z_lo, z_hi))

    def _snap_added_lower_edge_depth_to_mask(self, added_pts, observed_pts):
        if added_pts.shape[0] == 0 or observed_pts.shape[0] == 0:
            return added_pts
        a_add_low = self._object_lowest_anchor(added_pts)
        if a_add_low is None:
            return added_pts
        z_target = self._approx_mask_depth_at_xy(observed_pts, a_add_low[0], a_add_low[1])
        if z_target is None or not np.isfinite(z_target):
            return added_pts
        dz = float(z_target - a_add_low[2])
        out = added_pts.copy()
        out[:, 2] += dz
        return out

    def _remove_sparse_regions(self, pts):
        if pts.shape[0] < max(self.completion_sparse_min_points, 10):
            return pts
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        labels = np.asarray(
            pcd.cluster_dbscan(
                eps=max(float(self.completion_sparse_eps), 1.0),
                min_points=max(int(self.completion_sparse_min_points), 3),
                print_progress=False,
            )
        )
        if labels.size == 0 or not np.any(labels >= 0):
            return pts

        uniq = np.unique(labels[labels >= 0])
        sizes = [(int(lbl), int(np.sum(labels == lbl))) for lbl in uniq]
        sizes.sort(key=lambda x: x[1], reverse=True)
        keep_k = max(int(self.completion_sparse_keep_top_k), 1)
        keep_lbls = {lbl for lbl, _ in sizes[:keep_k]}
        keep_mask = np.array([(lab in keep_lbls) for lab in labels], dtype=bool)
        kept = pts[keep_mask]
        if kept.shape[0] < 10:
            return pts

        p_kept = o3d.geometry.PointCloud()
        p_kept.points = o3d.utility.Vector3dVector(kept)
        p_kept, _ = p_kept.remove_statistical_outlier(
            nb_neighbors=max(int(self.completion_sparse_stat_nb), 5),
            std_ratio=max(float(self.completion_sparse_stat_std), 0.1),
        )
        arr = np.asarray(p_kept.points)
        return arr if arr.shape[0] > 0 else kept

    def _postprocess_completed_cloud_to_mask(self, completed_cloud, observed_pts, mask_bool, K):
        pts = np.asarray(completed_cloud.points)
        if pts.shape[0] == 0:
            return completed_cloud

        out_pts = pts.copy()
        if self.completion_attach_to_mask and observed_pts.shape[0] > 0:
            # Attach only the added/completed part; keep observed-already part fixed.
            obs_pcd = o3d.geometry.PointCloud()
            obs_pcd.points = o3d.utility.Vector3dVector(observed_pts)
            tree = o3d.geometry.KDTreeFlann(obs_pcd)
            near = np.zeros((out_pts.shape[0],), dtype=bool)
            for i, p in enumerate(out_pts):
                k, _, dist = tree.search_knn_vector_3d(p, 1)
                if k > 0 and np.sqrt(dist[0]) <= self.completion_attach_dist_thresh:
                    near[i] = True

            added_pts = out_pts[~near]
            if added_pts.shape[0] > 0:
                # Requested: lowest point of object should match lowest point of mask.
                a_obs = self._mask_mode_lowest_anchor(observed_pts)
                a_add = self._object_lowest_anchor(added_pts)
                if a_obs is not None and a_add is not None:
                    t = self._constrain_translation_by_axis(a_obs - a_add)
                    added_pts = added_pts + t
                # Hard edge contact in full XYZ: one edge point of model must touch mask edge.
                t_contact = self._constrain_translation_by_axis(
                    self._edge_contact_translation_xyz(observed_pts, added_pts)
                )
                added_pts = added_pts + t_contact


                # Keep only added cluster closest to observed edge to avoid detached fragments.
                if self.completion_prune_added_clusters and added_pts.shape[0] >= 30:
                    p_add = o3d.geometry.PointCloud()
                    p_add.points = o3d.utility.Vector3dVector(added_pts)
                    labels = np.asarray(
                        p_add.cluster_dbscan(
                            eps=max(self.completion_attach_dist_thresh * 2.0, 12.0),
                            min_points=30,
                            print_progress=False,
                        )
                    )
                    if labels.size > 0 and np.any(labels >= 0):
                        best_lbl = None
                        best_d = np.inf
                        for lbl in np.unique(labels[labels >= 0]):
                            c = added_pts[labels == lbl]
                            c_cent = np.mean(c, axis=0)
                            d = np.linalg.norm(c_cent - a_obs) if a_obs is not None else np.linalg.norm(c_cent - np.mean(observed_pts, axis=0))
                            if d < best_d:
                                best_d = d
                                best_lbl = lbl
                        if best_lbl is not None:
                            added_pts = added_pts[labels == best_lbl]

                if self.completion_match_size_to_observed and added_pts.shape[0] > 0:
                    added_pts = self._match_added_extent_to_observed(added_pts, observed_pts)
                if self.completion_remove_sparse_regions and added_pts.shape[0] > 0:
                    added_pts = self._remove_sparse_regions(added_pts)

                if self.completion_contact_glue and added_pts.shape[0] > 0:
                    # Final contact glue in 3D: move added part along depth (Z) so it touches observed cloud.
                    nn_dist = np.empty((added_pts.shape[0],), dtype=np.float64)
                    nn_dz = np.empty((added_pts.shape[0],), dtype=np.float64)
                    for i, p in enumerate(added_pts):
                        k, idx, dist = tree.search_knn_vector_3d(p, 1)
                        if k > 0:
                            nn_dist[i] = np.sqrt(dist[0])
                            nn_dz[i] = observed_pts[idx[0], 2] - p[2]
                        else:
                            nn_dist[i] = np.inf
                            nn_dz[i] = 0.0
                    valid = np.isfinite(nn_dist)
                    if np.any(valid):
                        d = nn_dist[valid]
                        dz = nn_dz[valid]
                        pctl = float(np.clip(self.completion_contact_percentile, 0.1, 50.0))
                        d_th = float(np.percentile(d, pctl))
                        sel = d <= d_th
                        if np.any(sel):
                            dz_shift = float(np.median(dz[sel]))
                            added_pts[:, 2] += dz_shift
                # Enforce edge contact as the very last transform.
                if added_pts.shape[0] > 0:
                    t_contact_final = self._constrain_translation_by_axis(
                        self._edge_contact_translation_xyz(observed_pts, added_pts)
                    )
                    added_pts = added_pts + t_contact_final
                    # Hard final snap: lowest object point must coincide with lowest mask point.
                    a_obs_low = self._mask_mode_lowest_anchor(observed_pts)
                    a_add_low = self._object_lowest_anchor(added_pts)
                    if a_obs_low is not None and a_add_low is not None:
                        # Exact point-to-point snap of lowest points in full XYZ.
                        t_low = np.array(
                            [
                                float(a_obs_low[0] - a_add_low[0]),
                                float(a_obs_low[1] - a_add_low[1]),
                                float(a_obs_low[2] - a_add_low[2]),
                            ],
                            dtype=np.float64,
                        )
                        added_pts = added_pts + t_low
                    # Extra hard constraint: match lowest Y extrema exactly.
                    dy_low = float(np.max(observed_pts[:, 1]) - np.max(added_pts[:, 1]))
                    added_pts[:, 1] += dy_low
                    if self.completion_low_y_bias_mm != 0.0:
                        added_pts[:, 1] += float(self.completion_low_y_bias_mm)
                    if self.completion_above_mask_mm != 0.0:
                        added_pts[:, 1] -= abs(float(self.completion_above_mask_mm))
                    if self.completion_fit_inside_mask_2d:
                        added_pts = self._fit_points_inside_mask_2d(added_pts, mask_bool, K)
                        added_pts = self._snap_added_depth_to_observed(added_pts, observed_pts)
                        added_pts = self._optimize_added_z_by_nn(added_pts, observed_pts)
                        added_pts = self._snap_added_xyz_to_observed(added_pts, observed_pts)
                    if self.completion_align_height_to_mask:
                        added_pts = self._align_added_height_to_mask(added_pts, observed_pts)
                    if self.completion_depth_glue:
                        # Final depth-only glue: keep XY placement, align lower edge by mask depth.
                        added_pts = self._snap_added_lower_edge_depth_to_mask(added_pts, observed_pts)
                    if self.completion_force_lower_edge_glue:
                        # Final hard constraint: glue lower edge of model to lower edge of observed mask.
                        obs_edge = self._lower_edge_anchor(observed_pts)
                        add_edge = self._lower_edge_anchor(added_pts)
                        if obs_edge is not None and add_edge is not None:
                            added_pts = added_pts + self._constrain_translation_by_axis(obs_edge - add_edge)
                    if self.completion_snap_xy_to_observed:
                        # Keep depth as-is, tighten lateral placement to segmented cloud.
                        added_pts = self._snap_added_xy_to_observed(added_pts, observed_pts)
                    if self.completion_fit_inside_mask_2d:
                        # Final 2D placement fix: XY-only alignment to segmentation mask, keep depth.
                        added_pts = self._fit_points_inside_mask_2d(added_pts, mask_bool, K)
                    if self.completion_force_mode_lowest_point_glue:
                        # Absolute last step: pin object's min-Z point to mask mode-lowest depth (Z only).
                        a_obs_low = self._mask_mode_lowest_anchor(observed_pts)
                        a_add_zmin = self._object_min_z_anchor(added_pts)
                        if a_obs_low is not None and a_add_zmin is not None:
                            added_pts[:, 2] += float(a_obs_low[2] - a_add_zmin[2])

                out_pts = np.vstack([observed_pts, added_pts])
            else:
                # Fallback to full-cloud alignment if split failed.
                a_obs = self._mask_mode_lowest_anchor(observed_pts)
                a_cmp = self._object_lowest_anchor(out_pts)
                if a_obs is not None and a_cmp is not None:
                    t = self._constrain_translation_by_axis(a_obs - a_cmp)
                    out_pts += t
                if self.completion_fit_inside_mask_2d:
                    out_pts = self._fit_points_inside_mask_2d(out_pts, mask_bool, K)
                    out_pts = self._snap_added_depth_to_observed(out_pts, observed_pts)

        if self.completion_scale_to_mask_2d:
            out_pts = self._scale_and_center_completion_to_mask_2d(out_pts, mask_bool, K)

        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(out_pts)
        if len(completed_cloud.colors):
            out.colors = completed_cloud.colors
        return out

    # ============================================================
    # MAIN
    # ============================================================

    def _depth_to_3d(self, depth_img, K):
        if hasattr(cv2, "rgbd") and hasattr(cv2.rgbd, "depthTo3d"):
            return cv2.rgbd.depthTo3d(depth_img, K)

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        depth = depth_img.astype(np.float32)
        if depth_img.dtype == np.uint16:
            depth = depth * self.depth_scale

        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        pts = np.stack([x, y, z], axis=-1)
        pts[z <= 0] = np.nan
        return pts

    def estimate(self, color_img, depth_img, K):
        # --- segmentation ---
        if self.seg_model_type == "sam":
            det = self.det_model(color_img, classes=self.bottle_cls)[0]
            if len(det.boxes) == 0:
                return None
            seg_result = self.seg_model.predict(
                color_img,
                bboxes=det.boxes.xyxy,
                conf=self.seg_conf,
                iou=self.seg_iou,
                imgsz=self.seg_imgsz,
                verbose=False
            )[0]
        else:
            seg_result = self.seg_model.predict(
                color_img,
                conf=self.seg_conf,
                iou=self.seg_iou,
                imgsz=self.seg_imgsz,
                verbose=False
            )[0]

        if seg_result.masks is None:
            return None

        # --- depth -> 3D ---
        pts = self._depth_to_3d(depth_img, K)
        table_normal = self._estimate_table_normal(pts.reshape(-1, 3))

        candidates = []

        for mask in seg_result.masks.data.cpu().numpy():
            if mask.shape != depth_img.shape:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (depth_img.shape[1], depth_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask_bool = self._clean_mask(mask.astype(bool))
            mask_neck_px = self._estimate_mask_neck_point_2d(mask_bool)
            mask_axis_2d = self._estimate_mask_axis_2d(mask_bool)
            mask_neck_dir_2d = self._estimate_mask_neck_dir_2d(mask_bool, mask_neck_px)

            reg = self.estimate_pose_for_mask(
                pts,
                color_img,
                mask_bool,
                K=K
            )

            if reg is None:
                continue

            obj_pts = pts[mask_bool, :].reshape(-1, 3)
            obj_pts = obj_pts[~np.isnan(obj_pts).any(axis=1)]
            if obj_pts.shape[0] < self.min_points_for_pose:
                continue
            obs_axis, obs_neck_sign = self._estimate_axis_and_neck(obj_pts)
            if obs_axis is None:
                continue
            obs_neck_dir = obs_axis * obs_neck_sign
            obs_neck_dir = obs_neck_dir / (np.linalg.norm(obs_neck_dir) + 1e-12)
            geom = self._pose_from_mask_geometry(obj_pts, table_normal)
            if geom is None:
                continue
            geom_pose, geom_neck_dir = geom

            # Compare geometric pose vs ICP pose and keep the one that matches mask better.
            icp_pose = self._align_axis_parallel_to_table(
                reg.transformation, table_normal, obs_neck_dir
            )
            iou_geom = self._projected_mask_iou(geom_pose, K, mask_bool)
            iou_icp = self._projected_mask_iou(icp_pose, K, mask_bool)
            if iou_geom >= iou_icp:
                aligned_pose = geom_pose
                obs_neck_dir = geom_neck_dir
            else:
                aligned_pose = icp_pose

            proj_iou = self._projected_mask_iou(aligned_pose, K, mask_bool)
            neck_score = self._neck_alignment_score(aligned_pose, K, mask_bool.shape, mask_neck_px)
            axis_score = self._axis_alignment_score(aligned_pose, K, mask_bool.shape, mask_axis_2d)
            neck_dir_score = self._neck_dir_alignment_score(aligned_pose, K, mask_bool.shape, mask_neck_dir_2d)
            score = (
                reg.fitness
                - self.rmse_weight * reg.inlier_rmse
                + self.iou_weight * proj_iou
                + self.neck_weight * neck_score
                + self.axis_weight * axis_score
                + self.neck_dir_weight * neck_dir_score
            )
            centroid = self._mask_centroid(mask_bool)
            contains_target = False
            dist_to_target = np.inf
            if self.target_point_xy is not None:
                tx, ty = int(self.target_point_xy[0]), int(self.target_point_xy[1])
                if 0 <= ty < mask_bool.shape[0] and 0 <= tx < mask_bool.shape[1]:
                    contains_target = bool(mask_bool[ty, tx])
                if centroid is not None:
                    dist_to_target = np.hypot(centroid[0] - tx, centroid[1] - ty)

            candidates.append(
                {
                    "reg": reg,
                    "mask": mask_bool,
                    "score": score,
                    "proj_iou": proj_iou,
                    "mask_neck_px": mask_neck_px,
                    "mask_axis_2d": mask_axis_2d,
                    "mask_neck_dir_2d": mask_neck_dir_2d,
                    "aligned_pose": aligned_pose,
                    "obs_neck_dir": obs_neck_dir,
                    "contains_target": contains_target,
                    "dist_to_target": dist_to_target,
                }
            )

        if not candidates:
            return None

        pool = candidates
        if self.target_point_xy is not None:
            contains = [c for c in candidates if c["contains_target"]]
            if contains:
                pool = contains
            else:
                pool = sorted(candidates, key=lambda c: c["dist_to_target"])[:1]

        best = max(pool, key=lambda c: c["score"])
        best_reg = best["reg"]
        best_mask = best["mask"]
        best_proj_iou = best["proj_iou"]
        best_mask_neck_px = best["mask_neck_px"]
        best_mask_axis_2d = best["mask_axis_2d"]
        best_mask_neck_dir_2d = best["mask_neck_dir_2d"]
        best_reg.transformation = best["aligned_pose"].copy()
        best_obs_neck_dir = best["obs_neck_dir"]

        # Hard neck-direction alignment before local optimization.
        best_reg.transformation = self._align_pose_neck_direction_2d(
            best_reg.transformation,
            K,
            best_mask.shape,
            best_mask_neck_dir_2d,
            table_normal,
        )
        best_proj_iou = self._projected_mask_iou(best_reg.transformation, K, best_mask)

        # Final local 2D refinement to improve model-to-mask alignment in image space.
        pre_refine_pose = best_reg.transformation.copy()
        refined_pose, refined_iou, refine_yaw_deg, refine_dx, refine_dy, refine_dz = self._refine_pose_with_projection(
            best_reg.transformation,
            K,
            best_mask,
            table_normal,
            best_obs_neck_dir,
            best_mask_neck_px,
            best_mask_axis_2d,
            best_mask_neck_dir_2d,
        )
        if refined_iou > best_proj_iou:
            best_reg.transformation = refined_pose
            best_proj_iou = refined_iou
        else:
            refine_yaw_deg = 0.0
            refine_dx = 0.0
            refine_dy = 0.0
            refine_dz = 0.0

        # Final neck snap to preserve correct bottle direction.
        snapped_pose = self._align_pose_neck_direction_2d(
            best_reg.transformation,
            K,
            best_mask.shape,
            best_mask_neck_dir_2d,
            table_normal,
        )
        snapped_pose = self._snap_pose_xy_to_neck(
            snapped_pose,
            K,
            best_mask.shape,
            best_mask_neck_px,
        )
        snapped_iou = self._projected_mask_iou(snapped_pose, K, best_mask)
        if snapped_iou >= best_proj_iou:
            best_reg.transformation = snapped_pose
            best_proj_iou = snapped_iou

        if best_reg is None or best_mask is None:
            return None
        if best_reg.fitness < self.min_fitness or best_reg.inlier_rmse > self.max_rmse:
            return None
        if best_proj_iou < self.min_proj_iou:
            return None

        segmented_obj_pts = pts[best_mask, :].reshape(-1, 3)

        # --- completion ---
        completed_cloud = self.complete_pointcloud(
            segmented_obj_pts,
            best_reg,
            dist_thresh=self.completion_dist_thresh
        )
        completed_cloud = self._postprocess_completed_cloud_to_mask(
            completed_cloud,
            segmented_obj_pts[~np.isnan(segmented_obj_pts).any(axis=1)],
            best_mask,
            K,
        )

        return {
            "pose": best_reg.transformation,
            "fitness": best_reg.fitness,
            "proj_iou": best_proj_iou,
            "mask_neck_px": best_mask_neck_px,
            "proj_neck_px": self._project_ref_neck_point(best_reg.transformation, K, best_mask.shape),
            "pre_refine_pose": pre_refine_pose,
            "refine_yaw_deg": refine_yaw_deg,
            "refine_dx": refine_dx,
            "refine_dy": refine_dy,
            "refine_dz": refine_dz,
            "completed_cloud": completed_cloud,
            "mask": best_mask
        }
