# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import json
import numpy as np

from typing import Tuple
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from utils.utils import transform_yaw2quaternion, blend_nms


CLASS_SEG_TO_STR_CLASS = {
    "bicycle": 0,
    "bus": 1,
    "car": 2,
    "motorcycle": 3,
    "pedestrian": 4,
    "trailer": 5,
    "truck": 6,
}
CLASS_STR_TO_SEG_CLASS = {
    0: "bicycle",
    1: "bus",
    2: "car",
    3: "motorcycle",
    4: "pedestrian",
    5: "trailer",
    6: "truck",
}


def save_results_nuscenes(tracking_results, result_path):
    """
    Info: This function saves tracking results in the required NuScenes format as a JSON file.
    Parameters:
        input:
            tracking_results: dict, tracking data for multiple scenes, where each scene contains frame-level trajectory information.
            result_path: str, the directory path where the JSON file will be saved.
        output:
            A JSON file named 'results.json' is saved in the specified directory with the formatted tracking results.
    """
    result = {
        "results": {},
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
    }

    for scene_id, scene_trajs in tracking_results.items():
        for frame_id, frame_data in tqdm(scene_trajs.items(), desc="Converting"):
            sample_token = frame_data["cur_sample_token"]
            trajs = frame_data["trajs"]

            sample_results = []
            for track_id, bbox in trajs.items():
                global_orientation = transform_yaw2quaternion(bbox.global_yaw)
                box_result = {
                    "sample_token": sample_token,
                    "translation": [
                        float(bbox.global_xyz_lwh_yaw_fusion[0]),
                        float(bbox.global_xyz_lwh_yaw_fusion[1]),
                        float(bbox.global_xyz_lwh_yaw_fusion[2]),
                    ],
                    "size": [
                        float(bbox.lwh_fusion[1]),
                        float(bbox.lwh_fusion[0]),
                        float(bbox.lwh_fusion[2]),
                    ],
                    "rotation": [
                        float(global_orientation[0]),
                        float(global_orientation[1]),
                        float(global_orientation[2]),
                        float(global_orientation[3]),
                    ],
                    "velocity": [
                        float(bbox.global_velocity[0]),
                        float(bbox.global_velocity[1]),
                    ],
                    "tracking_id": str(track_id),
                    "tracking_name": bbox.category,
                    "tracking_score": bbox.det_score,
                }
                sample_results.append(box_result)

            if sample_token in result["results"]:
                result["results"][sample_token] = (
                    result["results"][sample_token] + sample_results
                )
            else:
                result["results"][sample_token] = sample_results

    for sample_token in result["results"].keys():
        confs = sorted(
            [
                (-d["tracking_score"], ind)
                for ind, d in enumerate(result["results"][sample_token])
            ]
        )
        result["results"][sample_token] = [
            result["results"][sample_token][ind]
            for _, ind in confs[: min(500, len(confs))]
        ]

    with open(result_path + "/results.json", "w") as f:
        json.dump(result, f)


def save_results_nuscenes_for_motion(tracking_results, result_path):
    """
    Info: This function saves tracking results in the required motion metric format as a JSON file.
    Parameters:
        input:
            tracking_results: dict, tracking data for multiple scenes, where each scene contains frame-level trajectory information.
            result_path: str, the directory path where the JSON file will be saved.
        output:
            A JSON file named 'results_for_motion.json' is saved in the specified directory with the formatted tracking results.
    """
    result = {
        "results": {},
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
    }

    for scene_id, scene_trajs in tracking_results.items():
        for frame_id, frame_data in tqdm(scene_trajs.items(), desc="Converting"):
            sample_token = frame_data["cur_sample_token"]
            trajs = frame_data["trajs"]

            sample_results = []
            for track_id, bbox in trajs.items():
                global_orientation_det = transform_yaw2quaternion(bbox.global_yaw)
                global_orientation_kalman_cv = transform_yaw2quaternion(
                    bbox.global_yaw_fusion
                )
                box_result = {
                    "sample_token": sample_token,
                    "translation": [
                        float(bbox.global_xyz_lwh_yaw_fusion[0]),
                        float(bbox.global_xyz_lwh_yaw_fusion[1]),
                        float(bbox.global_xyz_lwh_yaw_fusion[2]),
                    ],
                    "size": [
                        float(bbox.lwh_fusion[1]),
                        float(bbox.lwh_fusion[0]),
                        float(bbox.lwh_fusion[2]),
                    ],
                    "rotation_det": [
                        float(global_orientation_det[0]),
                        float(global_orientation_det[1]),
                        float(global_orientation_det[2]),
                        float(global_orientation_det[3]),
                    ],
                    "rotation_kalman_cv": [
                        float(global_orientation_kalman_cv[0]),
                        float(global_orientation_kalman_cv[1]),
                        float(global_orientation_kalman_cv[2]),
                        float(global_orientation_kalman_cv[3]),
                    ],
                    "velocity_det": [
                        float(bbox.global_velocity[0]),
                        float(bbox.global_velocity[1]),
                    ],
                    "velocity_kalman_cv": [
                        float(bbox.global_velocity_fusion[0]),
                        float(bbox.global_velocity_fusion[1]),
                    ],
                    "velocity_diff": [
                        float(bbox.global_velocity_diff[0]),
                        float(bbox.global_velocity_diff[1]),
                    ],
                    "velocity_curve": [
                        float(bbox.global_velocity_curve[0]),
                        float(bbox.global_velocity_curve[1]),
                    ],
                    "tracking_id": str(track_id),
                    "tracking_name": bbox.category,
                    "tracking_score": bbox.det_score,
                }
                sample_results.append(box_result)

            if sample_token in result["results"]:
                result["results"][sample_token] = (
                    result["results"][sample_token] + sample_results
                )
            else:
                result["results"][sample_token] = sample_results

    for sample_token in result["results"].keys():
        confs = sorted(
            [
                (-d["tracking_score"], ind)
                for ind, d in enumerate(result["results"][sample_token])
            ]
        )
        result["results"][sample_token] = [
            result["results"][sample_token][ind]
            for _, ind in confs[: min(500, len(confs))]
        ]

    with open(result_path + "/results_for_motion.json", "w") as f:
        json.dump(result, f)


def obtain_box_bottom_corners(dets, ids=None):
    """
    Info: This function calculates the bottom corners of bounding boxes based on their 3D dimensions, orientation, and other properties.
    Parameters:
        input:
            dets: [x, y, z, w, l, h, vx, vy, qw, qx, qy, qz, det_score, class_label]
            ids: (Optional) NumPy array of tracking IDs corresponding to each bounding box.
        output:
            boxes_bottom_corners: (n, 4, 2).
    """

    dets[:, [3, 4]] = dets[:, [4, 3]]

    if dets.ndim == 1:
        dets = dets[None, :]
    assert dets.shape[1] == 14, "The number of observed states must satisfy 14"

    def abs_orientation_axisZ(orientation: Quaternion) -> Quaternion:
        return -orientation if orientation.axis[-1] < 0 else orientation

    def box_volum(wlh: Tuple[float, float, float]) -> float:
        return wlh[0] * wlh[1] * wlh[2]

    def box_bottom_area(wlh: Tuple[float, float, float]) -> float:
        return wlh[0] * wlh[1]

    NuscBoxes, boxes_bottom_corners = [], []
    for idx, det in enumerate(dets):
        center, size, rotation, velocity = (
            det[0:3],
            det[3:6],
            det[8:12],
            tuple(det[6:8].tolist() + [0.0]),
        )
        score, label = det[12], int(det[13])
        name = CLASS_STR_TO_SEG_CLASS[label]

        orientation = abs_orientation_axisZ(Quaternion(rotation))

        class NuscBox(Box):
            def __init__(self):
                super().__init__(
                    center, size, orientation, label, score, velocity, name, token=None
                )
                assert self.orientation.axis[-1] >= 0

                self.tracking_id = int(ids[idx]) if ids is not None else None
                self.yaw = self.orientation.radians
                self.name_label = CLASS_SEG_TO_STR_CLASS[name]
                self.bottom_corners_ = self.bottom_corners()[:2].T  # [4, 2]
                self.volume = box_volum(self.wlh)
                self.area = box_bottom_area(self.wlh)

            def __repr__(self):
                repr_str = super().__repr__() + ", tracking id: {}"
                return repr_str.format(self.tracking_id)

        curr_box = NuscBox()
        NuscBoxes.append(curr_box)
        boxes_bottom_corners.append(curr_box.bottom_corners_)

    return np.array(boxes_bottom_corners)


def filter_bboxes_with_nms(filtered_bboxes, cfg):
    """
    Info: Filters bounding boxes using non-maximum suppression (NMS).
    Parameters:
        input:
            filtered_bboxes: List of bounding box dictionaries, where each dictionary contains:
                - global_xyz: [x, y, z] coordinates of the box's center.
                - lwh: [length, width, height] dimensions of the box.
                - global_velocity: [vx, vy] velocity of the object in global coordinates.
                - global_orientation: [qw, qx, qy, qz] quaternion representing the box orientation.
                - detection_score: A confidence score of the detection.
                - category: Category label of the detected object (mapped to a number via cfg).
            cfg: Configuration dictionary containing category mappings and threshold values for NMS.
        output:
            filtered_bboxes: List of bounding boxes after applying NMS.
    """
    dets = np.array(
        [
            [
                bbox["global_xyz"][0],
                bbox["global_xyz"][1],
                bbox["global_xyz"][2],
                bbox["lwh"][0],
                bbox["lwh"][1],
                bbox["lwh"][2],
                bbox["global_velocity"][0],
                bbox["global_velocity"][1],
                bbox["global_orientation"][0],
                bbox["global_orientation"][1],
                bbox["global_orientation"][2],
                bbox["global_orientation"][3],
                bbox["detection_score"],
                cfg["CATEGORY_MAP_TO_NUMBER"][bbox["category"]],
            ]
            for bbox in filtered_bboxes
        ]
    )

    np_dets_bottom_corners = obtain_box_bottom_corners(dets)
    tmp_infos = {"np_dets": dets, "np_dets_bottom_corners": np_dets_bottom_corners}
    keep = blend_nms(
        box_infos=tmp_infos, metrics="iou_bev", thre=cfg["THRESHOLD"]["NMS_THRE"]
    )
    filtered_bboxes = [filtered_bboxes[i] for i in keep]
    return filtered_bboxes
