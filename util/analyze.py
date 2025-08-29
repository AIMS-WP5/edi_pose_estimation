import numpy as np
import csv
import cv2
import sys

file_path = sys.argv[1]
field_names = [
    "True T x", "True T y", "True T z", "True R x", "True R y", "True R z", 
    "T x", "T y", "T z", "R x", "R y", "R z", 
    "Fitness", "Inlier rmse", "Correspondence set size", 
    "Detection time", "Total time"
]

with open(file_path, "r", newline="") as file:
    reader = csv.DictReader(file, field_names)
    rows = [row for row in reader]

    new_file_path = file_path.split(".csv")[0] + "_m.csv"
    new_filed_names = field_names + ["cos", "angle error", "pose error"]
    with open(new_file_path, "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, new_filed_names)
        writer.writeheader()
        for row in rows[1:]:
            rvec_true = np.array(
                [row["True R x"],
                row["True R y"],
                row["True R z"]], 
                dtype=float
            )
            R_true, _ = cv2.Rodrigues(rvec_true)
            z_true = R_true[:, 2]
            rvec_est = np.array(
                [row["R x"],
                row["R y"],
                row["R z"]], 
                dtype=float
            )
            R_est, _ = cv2.Rodrigues(rvec_est)
            z_est = R_est[:, 2]
            cos_phi = (np.dot(z_true, z_est)) / (np.linalg.norm(z_true) * np.linalg.norm(z_est))
            row["cos"] = cos_phi
            row["angle error"] = np.rad2deg(np.arccos(cos_phi))
            tvec_true = np.array(
                [row["True T x"],
                row["True T y"],
                row["True T z"]], 
                dtype=float
            )
            tvec_est = np.array(
                [row["T x"],
                row["T y"],
                row["T z"]], 
                dtype=float
            )
            pose_error = np.linalg.norm(tvec_est - tvec_true)
            row["pose error"] = pose_error

            writer.writerow(row)
print(f"Modified CSV file: {new_file_path}")
