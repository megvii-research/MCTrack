# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from nuscenes-devkit (https://github.com/nutonomy/nuscenes-devkit)
# Copyright (c) 2020 Motional. All Rights Reserved.
# ------------------------------------------------------------------------

import os, json, sys, argparse, time, math
import numpy as np

from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from scipy.spatial.transform import Rotation as R
from enum import IntEnum
from typing import Tuple
from PIL import Image
from typing import List, Tuple, Union
from shapely.geometry import MultiPoint, box


def load_file(file_path):
    file_path = os.path.join(file_path)
    print(f"Parsing {file_path}")
    with open(file_path, "r") as f:
        file_json = json.load(f)
    return file_json


def rotation_matrix_to_quaternion(tm):
    R_3x3 = tm[:3, :3]
    rotation = R.from_matrix(R_3x3)
    quaternion = rotation.as_quat()  # [x, y, z, w]
    wxyz = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
    new_quaternion = Quaternion(wxyz)
    return new_quaternion


def transform_quaternion_global2ego(global_orientation, global2ego):
    rotation_matrix = global2ego[:3, :3]
    rotation_ego = R.from_matrix(rotation_matrix)
    rotation_global = R.from_quat(global_orientation)
    rotation_combined = rotation_ego * rotation_global
    ego_orientation = rotation_combined.as_quat()

    return ego_orientation


def transform_quaternion2yaw(quat, type="yaw"):
    """
    Info: This function transforms a quaternion into a specified Euler angle (yaw, pitch, or roll) in radians.
    Parameters:
        input:
            quat: Quaternion, representing the orientation as a quaternion [w, x, y, z].
            type: str, specifies which Euler angle to compute ("yaw", "pitch", or "roll"). Default is "yaw".
        output:
            theta_radians: float, the calculated angle in radians for the specified type (yaw, pitch, or roll).
    """
    if not isinstance(quat, Quaternion):
        quat = Quaternion(quat)

    w, x, y, z = quat.w, quat.x, quat.y, quat.z

    if type == "roll":
        theta_radians = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    elif type == "pitch":
        theta_radians = math.asin(2 * (w * y - z * x))
    elif type == "yaw":
        theta_radians = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return theta_radians


def transform_matrix(
    translation: np.ndarray = np.array([0, 0, 0]),
    rotation: np.ndarray = np.array([1, 0, 0, 0]),
    inverse: bool = False,
) -> np.ndarray:
    tm = np.eye(4)

    rotation = Quaternion(rotation)
    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm


def obtain_camera2liadar_transf(nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat):
    """
    Refer: https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/nuscenes/nuscenes_utils.py
    Info: This function computes the transformation matrix from a specific camera or sensor to the Top LiDAR in the NuScenes dataset.
    The transformation includes both rotation and translation components across multiple frames of reference (sensor -> ego -> global -> Top LiDAR).
    Parameters:
        input:
            nusc: NuScenes object, used to retrieve calibration and pose information.
            sensor_token: str, token identifying the specific sensor data.
            l2e_t: np.array, translation from Lidar to ego vehicle frame.
            l2e_r_mat: np.array, rotation matrix from Lidar to ego vehicle frame.
            e2g_t: np.array, translation from ego vehicle frame to global frame.
            e2g_r_mat: np.array, rotation matrix from ego vehicle frame to global frame.
        output:
            camera2lidar: dict, contains:
                - "translation": np.array, translation vector from the camera to the LiDAR frame.
                - "rotation": np.array, rotation matrix from the camera to the LiDAR frame.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])

    l2e_r_s = cs_record["rotation"]
    l2e_t_s = cs_record["translation"]
    e2g_r_s = pose_record["rotation"]
    e2g_t_s = pose_record["translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    ).squeeze(0)

    camera2lidar = {
        "translation": T,
        "rotation": R.T,
    }
    return camera2lidar


def obtain_sensor2top(
    nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """
    Info: Obtain the info with RT matric from general sensor to Top LiDAR.

    Parameters:
        input:
            nusc (class): Dataset class in the nuScenes dataset.
            sensor_token (str): Sample data token corresponding to the
                specific sensor type.
            l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
            l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
                in shape (3, 3).
            e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
            e2g_r_mat (np.ndarray): Rotation matrix from ego to global
                in shape (3, 3).
            sensor_type (str): Sensor to calibrate. Default: 'lidar'.
        output:
            sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    ).squeeze(0)
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep


def transform_bbox_camera2image(
    points: np.ndarray, view: np.ndarray, normalize: bool
) -> np.ndarray:
    """
    Info: This function transforms 3D points from the camera coordinate system to the 2D image coordinate system using the camera's intrinsic matrix.
    Parameters:
        input:
            points: np.ndarray, shape (3, n), 3D points in the camera coordinate system.
            view: np.ndarray, shape (n, n), intrinsic matrix of the camera (usually 3x3 or 4x4).
            normalize: bool, whether to normalize the points by dividing by their depth (z-coordinate).
        output:
            points: np.ndarray, shape (3, n), transformed points in the image coordinate system. If `normalize=False`, the third coordinate represents the depth.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def check_box_in_image(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """
    Info: This function checks if the convex hull of the reprojected bounding box corners intersects with the image canvas.
    Parameters:
        input:
            corner_coords: List, list of corner coordinates of the reprojected 2D bounding box.
            imsize: Tuple[int, int], size of the image canvas, default is (1600, 900).
        output:
            intersection_coords: Tuple[float, float, float, float], the bounding box coordinates (min_x, min_y, max_x, max_y) of the intersection between the convex hull and the image canvas.
            Returns None if no intersection.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords]
        )

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def get_corners(xyz, lwh, orientation) -> np.ndarray:
    """
    Info: This function calculates the 3D bounding box corners based on the object's dimensions, position, and orientation.
    Parameters:
        input:
            xyz: Tuple[float, float, float], the (x, y, z) position of the bounding box center.
            lwh: Tuple[float, float, float], the dimensions of the bounding box (length, width, height).
            orientation: Quaternion, the orientation of the bounding box.

        output:
            corners: np.ndarray, shape (3, 8), the 3D coordinates of the eight corners of the bounding box. The first four corners face forward, and the last four face backward.
    """
    l, w, h = lwh
    x, y, z = xyz

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    corners = np.dot(orientation.rotation_matrix, corners)

    # Translate
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners


class BoxVisibility(IntEnum):
    """Enumerates the various level of box visibility in an image"""

    ALL = 0  # Requires all corners are inside the image.
    ANY = 1  # Requires at least one corner visible in the image.
    NONE = (
        2  # Requires no corners to be inside, i.e. box can be fully outside the image.
    )


def check_box_in_imagev2(
    corners_3d,
    intrinsic: np.ndarray,
    imsize: Tuple[int, int],
    vis_level: int = BoxVisibility.ANY,
) -> bool:
    """
    Info: This function checks if a 3D bounding box is visible within an image using the camera's intrinsic matrix.
    Parameters:
        input:
            corners_3d: np.ndarray, shape (3, 8), 3D coordinates of the bounding box corners in the camera coordinate system.
            intrinsic: np.ndarray, shape (3, 3), intrinsic camera matrix.
            imsize: Tuple[int, int], the size of the image as (width, height).
            vis_level: int, visibility level defined by the BoxVisibility enum, determining how many corners need to be visible (BoxVisibility.ALL, BoxVisibility.ANY, BoxVisibility.NONE).
        output:
            bool: True if the visibility condition is satisfied based on the 'vis_level', otherwise False.
    """
    corners_img = transform_bbox_camera2image(corners_3d, intrinsic, normalize=True)[
        :2, :
    ]

    visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < imsize[0])
    visible = np.logical_and(visible, corners_img[1, :] < imsize[1])
    visible = np.logical_and(visible, corners_img[1, :] > 0)
    visible = np.logical_and(visible, corners_3d[2, :] > 1)

    in_front = (
        corners_3d[2, :] > 0.1
    )  # True if a corner is at least 0.1 meter in front of the camera.

    if vis_level == BoxVisibility.ALL:
        return all(visible) and all(in_front)
    elif vis_level == BoxVisibility.ANY:
        return any(visible) and all(in_front)
    elif vis_level == BoxVisibility.NONE:
        return True
    else:
        raise ValueError("vis_level: {} not valid".format(vis_level))


def transform_box_global2image(
    global_xyz, global_orientation, lwh, global2ego, cameras_transform_matrix
):
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    for cam in camera_types:
        camera2image = np.array(cameras_transform_matrix[cam]["camera2image"])
        ego2camera = np.array(cameras_transform_matrix[cam]["ego2camera"])

        global_xyz_one = np.ones((4,))
        global_xyz_one[:3,] = global_xyz
        ego_xyz_one = global2ego.dot(global_xyz_one)
        ego_orientation = Quaternion(
            rotation_matrix_to_quaternion(global2ego)
        ) * Quaternion(global_orientation)

        camera_xyz_one = ego2camera.dot(ego_xyz_one)
        camera_xyz = camera_xyz_one[:3]
        camera_orientation = (
            Quaternion(rotation_matrix_to_quaternion(ego2camera)) * ego_orientation
        )
        camera_corners = get_corners(camera_xyz, lwh, camera_orientation)
        corner_coords = (
            transform_bbox_camera2image(camera_corners, camera2image, True)
            .T[:, :2]
            .tolist()
        )
        final_coords = check_box_in_image(corner_coords)
        # flag = check_box_in_imagev2(camera_corners, camera2image, (1600, 900))
        if final_coords is not None:
            return cam, final_coords

    return None, None


def load_camera_info(nusc, sample_info):
    ref_sd_token = sample_info["data"]["LIDAR_TOP"]
    ref_sd_rec = nusc.get("sample_data", ref_sd_token)
    ref_cs_rec = nusc.get("calibrated_sensor", ref_sd_rec["calibrated_sensor_token"])
    ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])

    l2e_r = ref_cs_rec["rotation"]
    l2e_t = (ref_cs_rec["translation"],)
    e2g_r = ref_pose_rec["rotation"]
    e2g_t = ref_pose_rec["translation"]
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    all_cam_info = {}
    for cam in camera_types:
        cam_token = sample_info["data"][cam]
        cam_path, _, camera_intrinsics = nusc.get_sample_data(cam_token)
        cam_info = obtain_sensor2top(
            nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
        )
        cam_info.update(camera_intrinsics=camera_intrinsics)

        # get image size.
        image_shape = (1600, 900)

        # camera intrinsics
        camera_intrinsics = np.eye(4).astype(np.float32)
        camera_intrinsics[:3, :3] = cam_info["camera_intrinsics"]

        # lidar to image transform
        # lidar2image = camera_intrinsics @ lidar2camera_rt.T

        # camera to ego transform
        camera2ego = np.eye(4).astype(np.float32)
        camera2ego[:3, :3] = Quaternion(cam_info["sensor2ego_rotation"]).rotation_matrix
        camera2ego[:3, 3] = cam_info["sensor2ego_translation"]
        ego2camera = np.linalg.inv(camera2ego)

        # camera to lidar transform
        camera2lidar = np.eye(4).astype(np.float32)
        camera2lidar[:3, :3] = cam_info["sensor2lidar_rotation"]
        camera2lidar[:3, 3] = cam_info["sensor2lidar_translation"]
        lidar2camera = np.linalg.inv(camera2lidar)

        cam_dict = {
            "image_shape": image_shape,
            "ego2camera": ego2camera.tolist(),
            "camera2image": camera_intrinsics.tolist(),
            "lidar2camera": lidar2camera.tolist(),
            "camera_token": cam_token,
            "camera_path": cam_path,
        }
        all_cam_info.update({cam: cam_dict})

    return all_cam_info


def convert_baseversion_bboxes(bboxes, global2ego, cameras_transform_matrix):
    new_bboxes = []
    for bbox in bboxes:
        global_xyz = np.array(bbox["translation"])
        global_orientation = bbox["rotation"]
        global_yaw = transform_quaternion2yaw(global_orientation)
        lwh = [bbox["size"][1], bbox["size"][0], bbox["size"][2]]
        cam, x1y1x2y2 = transform_box_global2image(
            global_xyz, global_orientation, lwh, global2ego, cameras_transform_matrix
        )
        new_bbox = {
            "detection_score": bbox.get("detection_score", 0.0),
            "category": bbox.get("detection_name", None),
            "global_xyz": global_xyz.tolist(),
            "global_orientation": global_orientation,
            "global_yaw": global_yaw,
            "lwh": lwh,
            "global_velocity": bbox.get("velocity", [0.0, 0.0]),
            "global_acceleration": bbox.get("acceleration", [0.0, 0.0]),
            "bbox_image": {
                "camera_type": cam,
                "x1y1x2y2": x1y1x2y2,
            },
        }
        new_bboxes.append(new_bbox)
    return new_bboxes


def nuscenes_main(raw_data_path, dets_path, detector, save_path, split):
    start_time = time.time()
    dets_path = os.path.join(dets_path, detector, split + ".json")
    dets_json = load_file(dets_path)
    if split == "test":
        scene_names = splits.create_splits_scenes()["test"]
        nusc = NuScenes(version="v1.0-test", dataroot=raw_data_path, verbose=True)
    else:
        scene_names = splits.create_splits_scenes()["val"]
        nusc = NuScenes(
            version="v1.0-trainval", dataroot=raw_data_path, verbose=True
        )  # v1.0-trainval
    all_datas = {}
    for scene_index in tqdm(
        range(len(nusc.scene)), desc="Convert nuScense dataset to base-version!"
    ):
        scene_info = nusc.scene[scene_index]
        scene_name = scene_info["name"]
        if scene_name not in scene_names:
            continue
        first_sample_token = scene_info["first_sample_token"]
        cur_sample_token = deepcopy(first_sample_token)

        frame_id = 0
        scene_datas = []
        while True:
            sample_info = nusc.get("sample", cur_sample_token)
            cameras_transform_matrix = load_camera_info(nusc, sample_info)

            lidar_token = sample_info["data"]["LIDAR_TOP"]
            timestamp = sample_info["timestamp"]
            lidar_info = nusc.get("sample_data", lidar_token)

            ego_token = lidar_info["ego_pose_token"]
            ego_pose = nusc.get("ego_pose", ego_token)

            global2ego = transform_matrix(
                ego_pose["translation"], ego_pose["rotation"], inverse=True
            )
            ego2global = transform_matrix(
                ego_pose["translation"], ego_pose["rotation"], inverse=False
            )

            calibrated_sensor = nusc.get(
                "calibrated_sensor", lidar_info["calibrated_sensor_token"]
            )
            lidar2ego = transform_matrix(
                calibrated_sensor["translation"],
                calibrated_sensor["rotation"],
                inverse=False,
            )
            ego2lidar = transform_matrix(
                calibrated_sensor["translation"],
                calibrated_sensor["rotation"],
                inverse=True,
            )

            lidar2global = ego2global.dot(lidar2ego)
            global2lidar = ego2lidar.dot(global2ego)

            raw_dets = dets_json["results"][cur_sample_token]
            bboxes = convert_baseversion_bboxes(
                raw_dets, global2ego, cameras_transform_matrix
            )

            scene_data = {
                "frame_id": frame_id,
                "cur_sample_token": cur_sample_token,
                "timestamp": timestamp,
                "bboxes": bboxes,
                "transform_matrix": {
                    "global2ego": global2ego.tolist(),
                    "ego2lidar": ego2lidar.tolist(),
                    "global2lidar": global2lidar.tolist(),
                    "cameras_transform_matrix": cameras_transform_matrix,
                },
            }
            scene_datas.append(scene_data)

            cur_sample_token = sample_info["next"]
            if cur_sample_token == "":
                break
            frame_id += 1

        all_datas[scene_name] = scene_datas

    save_path = os.path.join(save_path, detector)
    os.makedirs(save_path, exist_ok=True)
    save_path_temp = os.path.join(save_path, split + ".json")
    with open(save_path_temp, "w") as f:
        json.dump(all_datas, f)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Done conveting in {elapsed_time} seconds")
    print("Data format conversion is complete, please enjoy the tracking!!!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="/data/projects/datasets/nuScenes/nuscenes/raw_data/",
    )
    parser.add_argument("--dets_path", type=str, default="data/nuscenes/detectors/")
    parser.add_argument("--save_path", type=str, default="data/base_version/nuscenes")
    parser.add_argument("--detector", type=str, default="centerpoint")
    parser.add_argument("--split", type=str, default="val", help="test/val")
    args = parser.parse_args()

    all_datas = nuscenes_main(
        args.raw_data_path, args.dets_path, args.detector, args.save_path, args.split
    )
