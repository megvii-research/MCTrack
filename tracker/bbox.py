# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np

from pyquaternion import Quaternion


class BBox:
    """Bounding Box"""

    def __init__(
        self,
        frame_id,
        bbox,
        **kwargs,
    ):
        self.frame_id = frame_id
        self.is_fake = False
        self.is_interpolation = False
        self.category = bbox["category"]
        self.det_score = bbox["detection_score"]
        self.lwh = bbox["lwh"]
        self.global_xyz = bbox["global_xyz"]
        self.global_orientation = bbox["global_orientation"]
        self.global_yaw = bbox["global_yaw"]
        self.global_velocity = bbox["global_velocity"]
        self.global_acceleration = bbox["global_acceleration"]
        self.global_xyz_last = self.backward_prediction()

        self.global_velocity_fusion = self.global_velocity
        self.global_acceleration_fusion = self.global_acceleration
        self.global_yaw_fusion = self.global_yaw
        self.lwh_fusion = self.lwh
        self.global_velocity_diff = [0, 0]
        self.global_velocity_curve = [0, 0]

        # self.global_rot = self.transform_quaternion2yaw(self.global_orientation)
        self.global_xyz_lwh_yaw = self.global_xyz + self.lwh + [self.global_yaw]
        self.global_xyz_lwh_yaw_last = (
            self.global_xyz_last + self.lwh + [self.global_yaw]
        )
        self.global_xyz_lwh_yaw_predict = self.global_xyz_lwh_yaw
        self.global_xyz_lwh_yaw_fusion = self.global_xyz_lwh_yaw

        self.camera_type = bbox["bbox_image"].get("camera_type", None)
        self.x1y1x2y2 = bbox["bbox_image"].get("x1y1x2y2", [0.0, 0.0, 0.0, 0.0])
        self.x1y1x2y2_fusion = self.x1y1x2y2
        self.x1y1x2y2_predict = self.x1y1x2y2

        self.unmatch_length = 0

    def backward_prediction(self):
        last_xy = np.array(self.global_xyz[:2]) - np.array(self.global_velocity) * 0.5
        return last_xy.tolist() + [self.global_xyz[2]]

    def transform_quaternion2yaw(self, orientation):
        quat = Quaternion(orientation)
        theta_radians = 2 * np.arccos(quat[0])

        return theta_radians

    def transform_bbox_global2lidar(self, global_xyz_lwh_yaw_fusion, global2lidar):
        global_xyz = global_xyz_lwh_yaw_fusion[:3]
        global_xyz = np.array(global_xyz).reshape((1, -1))
        ones = np.ones(shape=(global_xyz.shape[0], 1))
        global_xyz_expend = np.concatenate([global_xyz, ones], -1)
        lidar_xyz = np.matmul(global_xyz_expend, global2lidar.T)
        lidar_xyz = lidar_xyz[0][:3].tolist()

        global_xyz_lwh_yaw_fusion = [item for item in global_xyz_lwh_yaw_fusion[3:]]
        lidar_xyz_lwh_yaw_fusion = np.array(lidar_xyz + global_xyz_lwh_yaw_fusion)

        return lidar_xyz_lwh_yaw_fusion

    def transform_bbox_lidar2camera(self, lidar_xyz_lwh_yaw_fusion, lidar2camera):
        lidar_xyz = lidar_xyz_lwh_yaw_fusion[:3]
        lidar_xyz = np.array(lidar_xyz).reshape((1, -1))
        ones = np.ones(shape=(lidar_xyz.shape[0], 1))
        lidar_xyz_expend = np.concatenate([lidar_xyz, ones], -1)
        lidar_xyz_expend_mat = np.mat(lidar_xyz_expend)
        lidar2camera_mat = np.mat(lidar2camera)
        lidar2camera_mat = lidar2camera_mat[0:3, 0:4]
        transformed_mat = lidar2camera_mat * lidar_xyz_expend_mat.T
        camera_xyz = np.array(transformed_mat.T, dtype=np.float32)[0].tolist()
        camera_xyz_lwh_yaw_fusion = np.array(
            camera_xyz + lidar_xyz_lwh_yaw_fusion[3:].tolist()
        )

        return camera_xyz_lwh_yaw_fusion

    def transform_3dbox2corners(self, global_xyz_lwh_yaw) -> np.ndarray:
        """
        Info: This function computes the 3D bounding box corners based on the box's position, dimensions, and orientation.
        Parameters:
            input:
                global_xyz_lwh_yaw: Tuple[float, float, float, float, float, float, float], a tuple containing the box center (x, y, z), dimensions (length, width, height),
                and yaw angle (rotation around the z-axis in radians).
            output:
                corners: np.ndarray, shape (8, 3), the 3D coordinates of the eight corners of the bounding box.
        Corner numbering order:
            4 -------- 0     z
           /|         /|     ↑   y
          5 -------- 1 .     |  /
          | |        | |     | /
          . 7 -------- 3     O — — — → x
          |/         |/
          6 -------- 2
        """
        x, y, z, l, w, h, rot = global_xyz_lwh_yaw
        orientation = Quaternion(axis=[0, 0, 1], radians=rot)
        dx1, dx2, dy1, dy2, dz1, dz2 = (
            l / 2.0,
            l / 2.0,
            w / 2.0,
            w / 2.0,
            h / 2.0,
            h / 2.0,
        )
        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = np.array([dx1, dx1, dx1, dx1, dx2, dx2, dx2, dx2]) * np.array(
            [1, 1, 1, 1, -1, -1, -1, -1]
        )
        y_corners = np.array([dy1, dy2, dy2, dy1, dy1, dy2, dy2, dy1]) * np.array(
            [1, -1, -1, 1, 1, -1, -1, 1]
        )
        z_corners = np.array([dz1, dz1, dz2, dz2, dz1, dz1, dz2, dz2]) * np.array(
            [1, 1, -1, -1, 1, 1, -1, -1]
        )
        corners = np.vstack((x_corners, y_corners, z_corners))
        # Rotate
        corners = np.dot(orientation.rotation_matrix, corners)
        # Translate
        # x, y, z = center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z
        return corners.T

    def transform_bbox_tlbr2xywh(self, x1y1x2y2=None):
        """
        Info: This function converts bounding box coordinates from the format '(x1, y1, x2, y2)' (top-left and bottom-right corners) to '(center_x, center_y, width, height)' format.
        Parameters:
            input:
                x1y1x2y2: np.ndarray, shape (4,), coordinates of the bounding box in the format (x1, y1, x2, y2). If None, the method will use 'self.x1y1x2y2'.
            output:
                xywh: np.ndarray, shape (4,), coordinates of the bounding box in the format (center_x, center_y, width, height).
        """
        if x1y1x2y2 is None:
            x1y1x2y2 = self.x1y1x2y2

        center_x = (x1y1x2y2[0] + x1y1x2y2[2]) / 2
        center_y = (x1y1x2y2[1] + x1y1x2y2[3]) / 2
        width = x1y1x2y2[2] - x1y1x2y2[0]
        height = x1y1x2y2[3] - x1y1x2y2[1]

        xywh = np.array([center_x, center_y, width, height])

        return xywh

    def transform_bbox_xywh2tlbr(self, xywh):
        """
        Info: This function converts bounding box coordinates from the format '(center_x, center_y, width, height)' to '(x1, y1, x2, y2)' format,
        where '(x1, y1)' represents the top-left corner and '(x2, y2)' represents the bottom-right corner.
        Parameters:
            input:
                xywh: np.ndarray, shape (4,), coordinates of the bounding box in the format (center_x, center_y, width, height).
            output:
                x1y1x2y2: np.ndarray, shape (4,), coordinates of the bounding box in the format (x1, y1, x2, y2).
        """
        x1 = xywh[0] - (xywh[2] / 2)
        y1 = xywh[1] - (xywh[3] / 2)

        x2 = xywh[0] + (xywh[2] / 2)
        y2 = xywh[1] + (xywh[3] / 2)

        x1y1x2y2 = np.array([x1, y1, x2, y2])

        return x1y1x2y2
