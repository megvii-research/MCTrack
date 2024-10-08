# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from waymo-open-dataset (https://github.com/waymo-research/waymo-open-dataset)
# Copyright (c) 2020 waymo. All Rights Reserved.
# ------------------------------------------------------------------------

import os, numpy as np, argparse, json, sys
import tensorflow as tf

from copy import deepcopy
from pyquaternion import Quaternion
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
from waymo_open_dataset.utils import box_utils


class BBox:
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, o=None):
        self.x = x  # center x
        self.y = y  # center y
        self.z = z  # center z
        self.h = h  # height
        self.w = w  # width
        self.l = l  # length
        self.o = o  # orientation
        self.s = None  # detection score

    def __str__(self):
        return "x: {}, y: {}, z: {}, heading: {}, length: {}, width: {}, height: {}, score: {}".format(
            self.x, self.y, self.z, self.o, self.l, self.w, self.h, self.s
        )

    @classmethod
    def bbox2dict(cls, bbox):
        return {
            "center_x": bbox.x,
            "center_y": bbox.y,
            "center_z": bbox.z,
            "height": bbox.h,
            "width": bbox.w,
            "length": bbox.l,
            "heading": bbox.o,
        }

    @classmethod
    def bbox2array(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h])
        else:
            return np.array(
                [bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h, bbox.s]
            )

    @classmethod
    def array2bbox(cls, data):
        bbox = BBox()
        bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox

    @classmethod
    def dict2bbox(cls, data):
        bbox = BBox()
        bbox.x = data["center_x"]
        bbox.y = data["center_y"]
        bbox.z = data["center_z"]
        bbox.h = data["height"]
        bbox.w = data["width"]
        bbox.l = data["length"]
        bbox.o = data["heading"]
        if "score" in data.keys():
            bbox.s = data["score"]
        return bbox

    @classmethod
    def copy_bbox(cls, bboxa, bboxb):
        bboxa.x = bboxb.x
        bboxa.y = bboxb.y
        bboxa.z = bboxb.z
        bboxa.l = bboxb.l
        bboxa.w = bboxb.w
        bboxa.h = bboxb.h
        bboxa.o = bboxb.o
        bboxa.s = bboxb.s
        return

    @classmethod
    def box2corners2d(cls, bbox):
        """the coordinates for bottom corners"""
        bottom_center = np.array([bbox.x, bbox.y, bbox.z - bbox.h / 2])
        cos, sin = np.cos(bbox.o), np.sin(bbox.o)
        pc0 = np.array(
            [
                bbox.x + cos * bbox.l / 2 + sin * bbox.w / 2,
                bbox.y + sin * bbox.l / 2 - cos * bbox.w / 2,
                bbox.z - bbox.h / 2,
            ]
        )
        pc1 = np.array(
            [
                bbox.x + cos * bbox.l / 2 - sin * bbox.w / 2,
                bbox.y + sin * bbox.l / 2 + cos * bbox.w / 2,
                bbox.z - bbox.h / 2,
            ]
        )
        pc2 = 2 * bottom_center - pc0
        pc3 = 2 * bottom_center - pc1

        return [pc0.tolist(), pc1.tolist(), pc2.tolist(), pc3.tolist()]

    @classmethod
    def box2corners3d(cls, bbox):
        """the coordinates for bottom corners"""
        center = np.array([bbox.x, bbox.y, bbox.z])
        bottom_corners = np.array(BBox.box2corners2d(bbox))
        up_corners = 2 * center - bottom_corners
        corners = np.concatenate([up_corners, bottom_corners], axis=0)
        return corners.tolist()

    @classmethod
    def motion2bbox(cls, bbox, motion):
        result = deepcopy(bbox)
        result.x += motion[0]
        result.y += motion[1]
        result.z += motion[2]
        result.o += motion[3]
        return result

    @classmethod
    def set_bbox_size(cls, bbox, size_array):
        result = deepcopy(bbox)
        result.l, result.w, result.h = size_array
        return result

    @classmethod
    def set_bbox_with_states(cls, prev_bbox, state_array):
        prev_array = BBox.bbox2array(prev_bbox)
        prev_array[:4] += state_array[:4]
        prev_array[4:] = state_array[4:]
        bbox = BBox.array2bbox(prev_array)
        return bbox

    @classmethod
    def box_pts2world(cls, ego_matrix, pcs):
        new_pcs = np.concatenate((pcs, np.ones(pcs.shape[0])[:, np.newaxis]), axis=1)
        new_pcs = ego_matrix @ new_pcs.T
        new_pcs = new_pcs.T[:, :3]
        return new_pcs

    @classmethod
    def edge2yaw(cls, center, edge):
        vec = edge - center
        yaw = np.arccos(vec[0] / np.linalg.norm(vec))
        if vec[1] < 0:
            yaw = -yaw
        return yaw

    @classmethod
    def bbox2world(cls, ego_matrix, box):
        # center and corners
        corners = np.array(BBox.box2corners2d(box))
        center = BBox.bbox2array(box)[:3][np.newaxis, :]
        center = BBox.box_pts2world(ego_matrix, center)[0]
        corners = BBox.box_pts2world(ego_matrix, corners)
        # heading
        edge_mid_point = (corners[0] + corners[1]) / 2
        yaw = BBox.edge2yaw(center[:2], edge_mid_point[:2])

        result = deepcopy(box)
        result.x, result.y, result.z = center
        result.o = yaw
        return result


if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()


def transform_yaw2quaternion(yaw):
    orientation = Quaternion(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    return orientation


def project_vehicle_to_image(ego2global, extrinsic, intrinsic, width, height, points):
    """
    Info: This function projects 3D points from the vehicle's coordinate system to the image plane using global shutter camera projection.
    Parameters:
        input:
            ego2global: np.ndarray, shape (4, 4), transformation matrix from vehicle to world (global) coordinates.
            extrinsic: np.ndarray, shape (4, 4), camera extrinsic matrix that defines the transformation from the world to the camera frame.
            intrinsic: np.ndarray, shape (3, 3), camera intrinsic matrix that defines the projection from the camera frame to the image plane.
            width: int, the width of the image.
            height: int, the height of the image.
            points: np.ndarray, shape (N, 3), 3D points in the vehicle's coordinate system to be projected.
        output:
            np.ndarray, shape (N, 3), projected 2D image coordinates with shape [N, 3] where (u, v) are image coordinates and `ok` indicates if the point is visible in the image.
    """
    # Transform points from vehicle to world coordinate system (can be
    # vectorized).
    pose_matrix = ego2global
    world_points = np.zeros_like(points)
    for i, point in enumerate(points):
        cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
        world_points[i] = (cx, cy, cz)

    # Populate camera image metadata. Velocity and latency stats are filled with
    # zeroes.
    extrinsic = tf.reshape(tf.constant(list(extrinsic), dtype=tf.float32), [4, 4])
    intrinsic = tf.constant(list(intrinsic), dtype=tf.float32)
    metadata = tf.constant(
        [
            width,
            height,
            5,
        ],
        dtype=tf.int32,
    )
    ego2global = ego2global.flatten()
    camera_image_metadata = list(ego2global) + [0.0] * 10
    # Perform projection and return projected image coordinates (u, v, ok).
    return py_camera_model_ops.world_to_image(
        extrinsic, intrinsic, metadata, camera_image_metadata, world_points
    ).numpy()


def transform_ego2image(ego_bbox, cam_info, ego2global):
    camera_type = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]

    for cam in camera_type:
        intrinsic = np.array(cam_info[cam]["intrinsic"])
        extrinsic = np.array(cam_info[cam]["extrinsic"])

        box_coords = np.array(
            [
                [
                    ego_bbox[0],
                    ego_bbox[1],
                    ego_bbox[2],
                    ego_bbox[4],
                    ego_bbox[5],
                    ego_bbox[6],
                    ego_bbox[3],
                ]
            ]
        )

        corners = box_utils.get_upright_3d_box_corners(box_coords)[0].numpy()

        width = 1920
        if cam == "SIDE_LEFT" or cam == "SIDE_RIGHT":
            height = 886
        else:
            height = 1280

        projected_corners = project_vehicle_to_image(
            ego2global, extrinsic, intrinsic, width, height, corners
        )
        u, v, ok = projected_corners.transpose()
        ok = ok.astype(bool)
        # Skip object if any corner projection failed. Note that this is very
        # strict and can lead to exclusion of some partially visible objects.
        if not all(ok):
            continue
        u = u[ok]
        v = v[ok]
        # Clip box to image bounds.
        u = np.clip(u, 0, width)
        v = np.clip(v, 0, height)

        if u.max() - u.min() == 0 or v.max() - v.min() == 0:
            continue
        return cam, (int(u.min()), int(v.min()), int(u.max()), int(v.max()))

    return None, None


def load_bboxes(dets_new, inst_types, cam_info, ego_bboxes, ego2global):
    new_bboxes = []
    for _, j in enumerate(dets_new):
        ego_bbox = ego_bboxes[_]
        lwh = [j.l.item(), j.w.item(), j.h.item()]
        detection_score = j.s.item()
        global_yaw = j.o.item()
        global_xyz = [j.x.item(), j.y.item(), j.z.item()]
        global_orientation = transform_yaw2quaternion(global_yaw)
        global_orientation_list = [
            global_orientation.w,
            global_orientation.x,
            global_orientation.y,
            global_orientation.z,
        ]
        type_name_str = inst_types[_]
        if type_name_str == 1:
            type_name_str = "car"
        elif type_name_str == 2:
            type_name_str = "pedestrian"
        elif type_name_str == 4:
            type_name_str = "bicycle"

        cam, x1y1x2y2 = transform_ego2image(ego_bbox, cam_info, ego2global)

        new_bbox = {
            "detection_score": detection_score,
            "category": type_name_str,
            "global_xyz": global_xyz,
            "global_orientation": global_orientation_list,
            "global_yaw": global_yaw,
            "lwh": lwh,
            "global_velocity": [0.0, 0.0],
            "global_acceleration": [0.0, 0.0],
            "bbox_image": {
                "camera_type": cam,
                "x1y1x2y2": x1y1x2y2,
            },
        }
        new_bboxes.append(new_bbox)
    return new_bboxes


def load_camera_info(camera_info_file):
    camera_type = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
    all_cam_info = {}
    for i in range(1, 6):
        ex_idx = i
        in_idx = 5 + i
        camera2ego = camera_info_file[ex_idx]
        ego2camera = np.linalg.inv(camera2ego)
        intrinsic = camera_info_file[in_idx]
        camera2image = np.eye(4)
        camera2image[0, 0] = intrinsic[0]
        camera2image[0, 2] = intrinsic[2]
        camera2image[1, 1] = intrinsic[1]
        camera2image[1, 2] = intrinsic[3]

        if i == 3 or i == 4:
            image_shape = (1920, 886)
        else:
            image_shape = (1920, 1280)

        cam_dict = {
            "image_shape": image_shape,
            "ego2camera": ego2camera.tolist(),
            "camera2image": camera2image.tolist(),
            "intrinsic": intrinsic.tolist(),
            "extrinsic": camera2ego.tolist(),
            "lidar2camera": None,
        }

        all_cam_info.update({camera_type[i - 1]: cam_dict})

    return all_cam_info


def waymo_main(raw_data_path, dets_path, det_name, save_path, split):
    raw_data_path = os.path.join(raw_data_path, split)
    dets_path = os.path.join(dets_path, det_name, split)
    all_datas = {}
    file_names = sorted(os.listdir(os.path.join(raw_data_path, "ego_info")))
    for file_index, file_name in enumerate(file_names[:]):
        print("SEQ {:} / {:}".format(file_index + 1, len(file_names)))
        segment_name = file_name.split(".")[0]
        print(segment_name)
        ts_info = json.load(
            open(
                os.path.join(raw_data_path, "ts_info", "{:}.json".format(segment_name)),
                "r",
            )
        )
        ego_info = np.load(
            os.path.join(raw_data_path, "ego_info", "{:}.npz".format(segment_name)),
            allow_pickle=True,
        )
        dets = np.load(
            os.path.join(dets_path, "{:}.npz".format(segment_name)),
            allow_pickle=True,
        )

        max_frame = len(dets["bboxes"])
        scene_datas = []

        for i in range(max_frame):
            new_bboxes = []
            bboxes = dets["bboxes"][i]
            inst_types = dets["types"][i]
            time_stamp = ts_info[i]
            ego_info_frame = ego_info["ego_infos"][i][0]
            ego2global = ego_info_frame[0]
            global2ego = np.linalg.inv(ego2global)
            cam_info = load_camera_info(ego_info_frame)
            dets_new = [BBox.bbox2world(ego2global, BBox.array2bbox(b)) for b in bboxes]
            new_bboxes = load_bboxes(dets_new, inst_types, cam_info, bboxes, ego2global)
            ego2global = ego2global.tolist()
            scene_data = {
                "frame_id": i,
                "cur_sample_token": None,
                "timestamp": time_stamp,
                "bboxes": new_bboxes,
                "transform_matrix": {
                    "global2ego": global2ego.tolist(),
                    "ego2lidar": None,
                    "global2lidar": None,
                    "cameras_transform_matrix": cam_info,
                },
            }
            scene_datas.append(scene_data)
        all_datas[segment_name] = scene_datas
    save_path = os.path.join(save_path, det_name)
    os.makedirs(save_path, exist_ok=True)
    save_path_temp = os.path.join(save_path, split + ".json")
    with open(save_path_temp, "w") as f:
        json.dump(all_datas, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # running configurations
    parser.add_argument("--det_name", type=str, default="ctrl")
    # paths
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="data/waymo/datasets/",
    )
    parser.add_argument(
        "--dets_path",
        type=str,
        default="data/waymo/detectors/",
    )
    parser.add_argument("--save_path", type=str, default="./data/base_version/waymo/")

    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()

    waymo_main(
        args.raw_data_path, args.dets_path, args.det_name, args.save_path, args.split
    )
