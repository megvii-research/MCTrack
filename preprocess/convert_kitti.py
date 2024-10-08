# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from kitti-devkit (https://github.com/utiasSTARS/pykitti)
# Copyright (c) 2020 KITTI. All Rights Reserved.
# ------------------------------------------------------------------------

import os, json, sys, argparse, math, re
import numpy as np

from copy import deepcopy
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from scipy.spatial.transform import Rotation as R


def load_file(file_path):
    """
    Info: This function loads a JSON file from the specified file path and returns its contents.
    Parameters:
        input:
            file_path: str, path to the JSON file.
        output:
            file_json: dict or list, parsed JSON content from the file.
    """
    file_path = os.path.join(file_path)
    print(f"Parsing {file_path}")
    with open(file_path, "r") as f:
        file_json = json.load(f)
    return file_json


def read_calib(calib_path):
    """
    Info: This function reads the calibration file and extracts the transformation matrices for projecting from 3D camera coordinates to 2D image pixels,
    and from 3D Velodyne Lidar coordinates to 3D camera coordinates.
    Parameters:
        input:
            calib_path: str, path to the calibration text file.
        output:
            vti_mat: np.array, shape (4, 4), projection matrix from 3D camera coordinates to 2D image pixels (P2).
            vtc_mat: np.array, shape (4, 4), transformation matrix from Velodyne Lidar coordinates to camera coordinates.
    """
    with open(calib_path) as f:
        for line in f.readlines():
            if line[:2] == "P2":
                P2 = re.split(" ", line.strip())
                P2 = np.array(P2[-12:], np.float32)
                P2 = P2.reshape((3, 4))
            if line[:14] == "Tr_velo_to_cam" or line[:11] == "Tr_velo_cam":
                vtc_mat = re.split(" ", line.strip())
                vtc_mat = np.array(vtc_mat[-12:], np.float32)
                vtc_mat = vtc_mat.reshape((3, 4))
                vtc_mat = np.concatenate([vtc_mat, [[0, 0, 0, 1]]])
            if line[:7] == "R0_rect" or line[:6] == "R_rect":
                R0 = re.split(" ", line.strip())
                R0 = np.array(R0[-9:], np.float32)
                R0 = R0.reshape((3, 3))
                R0 = np.concatenate([R0, [[0], [0], [0]]], -1)
                R0 = np.concatenate([R0, [[0, 0, 0, 1]]])
    vtc_mat = np.matmul(R0, vtc_mat)
    vti_mat = np.ones((4, 4))
    vti_mat[:3, :4] = P2

    return (vti_mat, vtc_mat)


def read_pose(path):
    pose_per_seq = {}
    pose_path = path
    with open(pose_path) as f:
        PoseList = f.readlines()
        for id, PoseStr in enumerate(PoseList):
            pose = PoseStr.split(" ")
            pose = np.array(pose, dtype=np.float32).reshape((-1, 4))
            pose = np.concatenate([pose, [[0, 0, 0, 1]]])
            pose_per_seq[id] = pose
    return pose_per_seq


def get_registration_angle(mat):
    cos_theta = mat[0, 0]
    sin_theta = mat[1, 0]

    if cos_theta < -1:
        cos_theta = -1
    if cos_theta > 1:
        cos_theta = 1

    theta_cos = np.arccos(cos_theta)

    if sin_theta >= 0:
        return theta_cos
    else:
        return 2 * np.pi - theta_cos


def get_global_yaw(yaw, pose):
    new_yaw = (np.pi - yaw) + np.pi / 2
    ang = get_registration_angle(pose)
    global_yaw = new_yaw + ang
    return global_yaw


def transform_quaternion_lidar2global(global_orientation, global2ego):
    """
    Info: This function transforms a quaternion representing orientation from the global coordinate system to the ego (vehicle) coordinate system using the provided transformation matrix.
    Parameters:
        input:
            global_orientation: np.array, shape (4,), quaternion representing orientation in the global coordinate system.
            global2ego: np.array, shape (4, 4), transformation matrix from the global frame to the ego (vehicle) frame.
        output:
            ego_orientation: np.array, shape (4,), quaternion representing the orientation in the ego coordinate system.
    """
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


def transform_yaw2quaternion(yaw):
    orientation = Quaternion(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    return orientation


def transform_camera2lidar(camera_xyz, camera_orientation, lidar2camera):
    """
    Info: This function converts 3D coordinates and orientation from the camera coordinate system to the Lidar coordinate system.
    Parameters:
        input:
            camera_xyz: np.array, shape (PointsNum, 3), 3D points in the camera coordinate system.
            camera_orientation: np.array, shape (4,), quaternion representing the orientation in the camera coordinate system [w, x, y, z].
            lidar2camera: np.array, shape (4, 4), transformation matrix from the Lidar coordinate system to the camera coordinate system.
        output:
            lidar_xyz: np.array, shape (PointsNum, 3), 3D points in the Lidar coordinate system.
            lidar_orientation: Quaternion, orientation in the Lidar coordinate system as a quaternion.
    """
    camera_xyz = camera_xyz.reshape((-1, 3))
    mat = np.ones(shape=(camera_xyz.shape[0], 4), dtype=np.float32)
    mat[:, 0:3] = camera_xyz[:, 0:3]
    mat = np.mat(mat)
    normal = np.mat(lidar2camera)
    normal = normal[0:3, 0:4]
    transformed_mat = normal * mat.T
    lidar_xyz = np.array(transformed_mat.T, dtype=np.float32)

    camera_orientation = [
        camera_orientation[0],
        camera_orientation[1],
        camera_orientation[2],
        camera_orientation[3],
    ]
    camera_orientation_r = R.from_quat(camera_orientation).as_matrix()
    normal_r = normal[:, :3]
    lidar_orientation_r = np.dot(normal_r, camera_orientation_r)
    lidar_orientation = R.from_matrix(lidar_orientation_r).as_quat()
    lidar_orientation = Quaternion(lidar_orientation)

    return lidar_xyz, lidar_orientation


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


def load_camera_info(lidar2camera=None, camera2image=None, image_shape=None):
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
        if cam == "CAM_FRONT":
            image_shape = (1242, 375)
            lidar2camera = lidar2camera.tolist()
            camera2image = camera2image.tolist()
        cam_dict = {
            "image_shape": image_shape,
            "ego2camera": None,
            "camera2image": camera2image,
            "lidar2camera": lidar2camera,
            "camera_token": None,
            "camera_image": None,
        }
        all_cam_info.update({cam: cam_dict})
    return all_cam_info


def convert_baseversion_bboxes(det_path, lidar2global, lidar2camera):
    new_bboxes = []
    with open(det_path) as f:
        for bbox_info in f.readlines():
            infos = re.split(" ", bbox_info)
            if infos[0] == "\n":
                return new_bboxes
            x1y1x2y2 = [
                float(infos[4]),
                float(infos[5]),
                float(infos[6]),
                float(infos[7]),
            ]
            lwh = [float(infos[10]), float(infos[9]), float(infos[8])]
            detection_score = 1 / (1 + np.exp(-float(infos[15].replace("\n", ""))))
            category = infos[0].lower()
            camera_xyz = np.array([float(i) for i in infos[11:14]])
            camera_rot = float(infos[14].replace("\n", ""))
            camera_orientation = Quaternion(
                np.cos(camera_rot / 2), 0, 0, np.sin(camera_rot / 2)
            )
            camera2lidar = np.linalg.inv(np.array(lidar2camera))
            lidar_xyz, lidar_orientation = transform_camera2lidar(
                camera_xyz, camera_orientation, camera2lidar
            )
            lidar_xyz[:, 2] += lwh[2] / 2

            lidar_xyz = np.array(lidar_xyz).reshape((1, -1))

            ones = np.ones(shape=(lidar_xyz.shape[0], 1))
            lidar_xyz_expend = np.concatenate([lidar_xyz, ones], -1)
            global_xyz = np.matmul(lidar2global, lidar_xyz_expend.T)
            global_xyz = global_xyz[:3].reshape(3).tolist()

            global_yaw = get_global_yaw(camera_rot, lidar2global)
            global_orientation = transform_yaw2quaternion(global_yaw)
            global_orientation_list = [
                global_orientation.w,
                global_orientation.x,
                global_orientation.y,
                global_orientation.z,
            ]

            new_bbox = {
                "detection_score": detection_score,
                "category": category,
                "global_xyz": global_xyz,
                "global_orientation": global_orientation_list,
                "global_yaw": global_yaw,
                "lwh": lwh,
                "global_velocity": [0.0, 0.0],
                "global_acceleration": [0.0, 0.0],
                "bbox_image": {
                    "camera_type": "CAM_FRONT",
                    "x1y1x2y2": x1y1x2y2,
                },
            }
            new_bboxes.append(new_bbox)
    return new_bboxes


def kitti_main(dataset_root, detections_root, detector, save_path, split):
    datset_split = "training" if split == "val" else "testing"
    all_seqs = 21 if datset_split == "training" else 29

    all_datas = {}
    dataset_path = os.path.join(dataset_root, datset_split)
    detections_path = os.path.join(detections_root, detector, datset_split)
    for seq_id in tqdm(range(all_seqs)):
        scene_name = str(seq_id).zfill(4)
        dets_path = os.path.join(detections_path, scene_name)
        calib_path = os.path.join(dataset_path, "calib", scene_name + ".txt")
        pose_path = os.path.join(dataset_path, "pose", scene_name + ".txt")
        dets_list = sorted(os.listdir(dets_path))
        dets_list = [int(re.findall(r"\d+", det)[0]) for det in dets_list]

        ego_poses = read_pose(pose_path)
        camera2image, lidar2camera = read_calib(calib_path)

        scene_datas = []
        for frame_id in tqdm(dets_list):
            frame_name = f"{frame_id:06d}.txt"
            lidar2global = ego_poses[frame_id]

            cameras_transform_matrix = load_camera_info(lidar2camera, camera2image)
            det_path = os.path.join(dets_path, frame_name)
            bboxes = convert_baseversion_bboxes(det_path, lidar2global, lidar2camera)

            global2lidar = np.mat(lidar2global).I
            scene_data = {
                "frame_id": frame_id,
                "cur_sample_token": None,
                "timestamp": None,
                "bboxes": bboxes,
                "transform_matrix": {
                    "global2ego": None,
                    "ego2lidar": None,
                    "global2lidar": global2lidar.tolist(),
                    "cameras_transform_matrix": cameras_transform_matrix,
                },
            }
            scene_datas.append(scene_data)
        all_datas[scene_name] = scene_datas

    save_path = os.path.join(save_path, detector)
    os.makedirs(save_path, exist_ok=True)
    save_path_temp = os.path.join(save_path, split + ".json")
    with open(save_path_temp, "w") as f:
        json.dump(all_datas, f)
    print("Data format conversion is complete, please enjoy the tracking!!!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, default="data/kitti/datasets/")
    parser.add_argument("--dets_path", type=str, default="data/kitti/detectors/")

    parser.add_argument("--save_path", type=str, default="./data/base_version/kitti/")
    parser.add_argument("--detector", type=str, default="virconv")

    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()

    all_datas = kitti_main(
        args.raw_data_path, args.dets_path, args.detector, args.save_path, args.split
    )
