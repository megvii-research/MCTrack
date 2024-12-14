# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from waymo-open-dataset (https://github.com/waymo-research/waymo-open-dataset)
# Copyright (c) 2020 waymo. All Rights Reserved.
# ------------------------------------------------------------------------

import os
import numpy as np
from tqdm import tqdm
from utils.waymo_utils.bbox_waymo import BBox


def save_results_waymo(tracking_results, save_path):
    """
    Info: This function saves tracking results in the required Waymo format, organizing predictions by category (vehicle, pedestrian, cyclist) and saving them as compressed NumPy files.
    Parameters:
        input:
            tracking_results: dict, tracking data for multiple scenes.
            save_path: str, the directory path where the .npz file will be saved.
        output:
            Compressed NumPy files (.npz) saved in the appropriate category directories for vehicle, pedestrian, and cyclist.
    """
    for scene_id, scene_trajs in tracking_results.items():
        (
            IDs_vehicle,
            bboxes_vehicle,
            IDs_pedestrian,
            bboxes_pedestrian,
            IDs_cyclist,
            bboxes_cyclist,
        ) = (
            list(),
            list(),
            list(),
            list(),
            list(),
            list(),
        )
        for frame_id, frame_data in tqdm(scene_trajs.items(), desc="Converting"):
            traj = frame_data["trajs"]

            result_pred_bboxes_vehicle = []
            result_pred_ids_vehicle = []
            result_pred_bboxes_pedestrian = []
            result_pred_ids_pedestrian = []
            result_pred_bboxes_cyclist = []
            result_pred_ids_cyclist = []

            for track_id, track in traj.items():
                bbox = track
                box_result = BBox(
                    x=float(bbox.global_xyz_lwh_yaw_fusion[0]),
                    y=float(bbox.global_xyz_lwh_yaw_fusion[1]),
                    z=float(bbox.global_xyz_lwh_yaw_fusion[2]),
                    l=float(bbox.lwh_fusion[0]),
                    w=float(bbox.lwh_fusion[1]),
                    h=float(bbox.lwh_fusion[2]),
                    o=float(bbox.global_yaw),
                )
                box_result.s = float(bbox.det_score)
                id = track_id

                if bbox.category == "car":
                    result_pred_bboxes_vehicle.append(box_result)
                    result_pred_ids_vehicle.append(id)
                elif bbox.category == "pedestrian":
                    result_pred_bboxes_pedestrian.append(box_result)
                    result_pred_ids_pedestrian.append(id)
                else:
                    result_pred_bboxes_cyclist.append(box_result)
                    result_pred_ids_cyclist.append(id)

            IDs_vehicle.append(result_pred_ids_vehicle)
            result_pred_bboxes_vehicle = [
                BBox.bbox2array(bbox) for bbox in result_pred_bboxes_vehicle
            ]
            bboxes_vehicle.append(result_pred_bboxes_vehicle)
            IDs_pedestrian.append(result_pred_ids_pedestrian)
            result_pred_bboxes_pedestrian = [
                BBox.bbox2array(bbox) for bbox in result_pred_bboxes_pedestrian
            ]
            bboxes_pedestrian.append(result_pred_bboxes_pedestrian)
            IDs_cyclist.append(result_pred_ids_cyclist)
            result_pred_bboxes_cyclist = [
                BBox.bbox2array(bbox) for bbox in result_pred_bboxes_cyclist
            ]
            bboxes_cyclist.append(result_pred_bboxes_cyclist)

        summary_folder_tmp = os.path.join(
            save_path, "vehicle"
        )  # cyclist vehicle pedestrian
        if not os.path.exists(summary_folder_tmp):
            os.makedirs(summary_folder_tmp)
        np.savez_compressed(
            os.path.join(summary_folder_tmp, "{}.npz".format(scene_id)),
            ids=IDs_vehicle,
            bboxes=bboxes_vehicle,
            states=None,
        )

        summary_folder_tmp = os.path.join(save_path, "pedestrian")
        if not os.path.exists(summary_folder_tmp):
            os.makedirs(summary_folder_tmp)
        np.savez_compressed(
            os.path.join(summary_folder_tmp, "{}.npz".format(scene_id)),
            ids=IDs_pedestrian,
            bboxes=bboxes_pedestrian,
            states=None,
        )

        summary_folder_tmp = os.path.join(save_path, "cyclist")
        if not os.path.exists(summary_folder_tmp):
            os.makedirs(summary_folder_tmp)
        np.savez_compressed(
            os.path.join(summary_folder_tmp, "{}.npz".format(scene_id)),
            ids=IDs_cyclist,
            bboxes=bboxes_cyclist,
            states=None,
        )
