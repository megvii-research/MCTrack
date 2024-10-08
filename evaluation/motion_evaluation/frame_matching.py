# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import cv2
import os
import numpy as np
from pyquaternion import Quaternion
from motion_evaluation.Posebox import PoseBox
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from easydict import EasyDict


def visualize_bboxes(bbox, gt_bbox, track_id):
    # Define the x-coordinate threshold
    x_threshold = 1500

    # Filter bounding boxes based on the x-coordinate threshold
    # filtered_bbox = [box for box in bbox if box[0] >= x_threshold]
    # filtered_gt_bbox = [box for box in gt_bbox if box[0] >= x_threshold]
    filtered_bbox = bbox
    filtered_gt_bbox = gt_bbox

    # Combine all filtered bounding boxes to find the min and max values
    all_bboxes = filtered_bbox + filtered_gt_bbox

    if not all_bboxes:
        print("No bounding boxes meet the x-coordinate threshold.")
        return

    x_min = min(box[0] - box[3] / 2 for box in all_bboxes)
    x_max = max(box[0] + box[3] / 2 for box in all_bboxes)
    y_min = min(box[1] - box[4] / 2 for box in all_bboxes)
    y_max = max(box[1] + box[4] / 2 for box in all_bboxes)

    # Calculate image size based on the bounding box range
    img_width = int(x_max - x_min) + 500
    img_height = int(y_max - y_min) + 500

    # Create a blank image with calculated size
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # Function to adjust bounding boxes to fit the new image
    def adjust_box(box):
        x_center, y_center, _, length, width, _, _, _, _ = box
        length = 10 * length
        width = 10 * width
        x_center = x_center - x_min + 250
        y_center = y_center - y_min + 250
        top_left = (int(x_center - length / 2), int(y_center - width / 2))
        bottom_right = (int(x_center + length / 2), int(y_center + width / 2))
        return top_left, bottom_right

    # Plot detected bboxes in red
    for box in filtered_bbox:
        top_left, bottom_right = adjust_box(box)
        cv2.rectangle(
            img, top_left, bottom_right, color=(0, 0, 255), thickness=2
        )  # Red for detected bboxes

    # Plot ground truth bboxes in green
    for box in filtered_gt_bbox:
        top_left, bottom_right = adjust_box(box)
        cv2.rectangle(
            img, top_left, bottom_right, color=(0, 255, 0), thickness=2
        )  # Green for ground truth bboxes

    # Create or clear the directory
    output_dir = "./visual_result/"

    # Save the image with the track_id in the filename
    output_path = os.path.join(output_dir, f"{track_id}_vis_bbox.jpg")
    cv2.imwrite(output_path, img)


GT_ID_IN_DT = EasyDict(
    TP="real_gt_id",
    FP=-1,
    TRUE_KILL=-2,
    WRONG_KILL=-3,
    INVALID=-99999,
)
# 每个gt 框对应的track_id取值
TRACK_ID_IN_GT = EasyDict(
    TP="real_track_id",
    FN=-1,
    WRONG_KILL=-3,
    INVALID=-99999,
)


def cal_bev_iou(gt_bev, dt_bev):
    """
    Calculate the BEV IoU (Intersection over Union) between two sets of bounding boxes.

    :param gt_bev: Ground truth bounding boxes in BEV, numpy array of shape (N, 3).
    :param dt_bev: Detected bounding boxes in BEV, numpy array of shape (M, 3).
    :return: BEV IoU value.
    """

    def create_polygon(bev_array):
        """Create a Polygon object from the BEV array. Assumes the array is in order."""
        return Polygon(bev_array[:, :2])  # We only need the x and y coordinates.

    # Create polygons from BEV arrays
    gt_polygon = create_polygon(gt_bev)
    dt_polygon = create_polygon(dt_bev)

    # Calculate intersection and union
    intersection = gt_polygon.intersection(dt_polygon).area
    union = gt_polygon.union(dt_polygon).area

    # Calculate IoU
    bev_iou = intersection / union if union != 0 else 0.0

    return bev_iou


def cal_3d_iou(gt_pose, dt_pose, iou_type="bev_iou"):
    gt_pose_box = PoseBox(
        xyz=gt_pose[0:3],
        lwh=gt_pose[3:6],
        orientation=Quaternion(axis=[0, 0, 1], radians=gt_pose[6]),
    )

    dt_pose_box = PoseBox(
        xyz=dt_pose[0:3],
        lwh=dt_pose[3:6],
        orientation=Quaternion(axis=[0, 0, 1], radians=dt_pose[6]),
    )

    if iou_type == "bev_iou":
        gt_bev = gt_pose_box.corners().T[[3, 2, 6, 7], :]
        dt_bev = dt_pose_box.corners().T[[3, 2, 6, 7], :]
        iou3d = cal_bev_iou(gt_bev, dt_bev)
    return iou3d


def cal_similarity(gt_box, dt_box, metric="bev_iou"):
    gt_pose = gt_box
    dt_pose = dt_box
    if metric == "bev_iou" or metric == "box3d_iou":
        sim = cal_3d_iou(gt_pose, dt_pose, iou_type=metric)
    else:
        raise NotImplementedError(f"{metric} is not supported for cal_similarity")

    return sim


def matching_v5(
    dt,
    gt,
    mask,
    threshold=0.5,
    match_min_max=True,
    metric="bev_iou",
    substitute_sim_threshold=1.0,
):
    """
    dt = {
      frame_id1:{
          "bbox": [],
          "track_id":[],
          "confidence": [],
          "gt_quality_level": [],
      },
      frame_id2: ...
    }

    gt = {
      frame_id1:{
          "bbox": [],
          "track_id":[],
          "confidence": [],
          "gt_quality_level": [],
      },
      frame_id2: ...
    }

    return: traj_track格式的匹配结果, gt_track格式的匹配结果
    """

    if metric == "box3d_iou" or metric == "bev_iou":
        sim_threshold = threshold
    elif metric == "center_dis":
        sim_threshold = -threshold
        substitute_sim_threshold = -substitute_sim_threshold
    else:
        raise NotImplementedError(f"{metric} is not supported for cal_similarity")

    all_frame_ids = sorted(list(set(list(dt.keys()) + list(gt.keys()))))
    traj_track, gt_track = {}, {}
    for frame_id in all_frame_ids:
        dt_info = dt.get(frame_id, dict(bbox=[], track_id=[], confidence=[]))
        gt_info = gt.get(frame_id, dict(bbox=[], track_id=[]))
        # dt按照confidence排序，这很重要
        dt_det_bboxes = zip(dt_info["bbox"], dt_info["track_id"], dt_info["confidence"])
        dt_det_bboxes = sorted(dt_det_bboxes, key=lambda x: x[2], reverse=True)

        dt_num, gt_num = len(dt_info["bbox"]), len(gt_info["bbox"])
        gt_bboxes_val = (
            list(zip(gt_info["bbox"], gt_info["track_id"], list(range(gt_num))))
            if gt_info
            else []
        )
        gt_matched = [False] * gt_num
        dt_ids = np.array([item[1] for item in dt_det_bboxes])
        gt_ids = np.array([item[1] for item in gt_bboxes_val])

        similarity = -1e9 * np.ones((dt_num, gt_num))
        for dt_idx, dt_bbox_tuple in enumerate(dt_det_bboxes):
            for gt_idx, gt_bbox_tuple in enumerate(gt_bboxes_val):
                similarity[dt_idx][gt_idx] = cal_similarity(
                    gt_bbox_tuple[0][:7], dt_bbox_tuple[0][:7], metric=metric
                )

        for dt_idx, dt_bbox_tuple in enumerate(dt_det_bboxes):
            dt_bbox, dt_track_id, dt_det_conf = dt_bbox_tuple
            # 如果当前检测框的跟踪ID在轨迹跟踪字典中不存在，则初始化该ID的记录
            if dt_track_id not in traj_track:
                traj_track[dt_track_id] = {
                    "frame_id": [],
                    "bbox": [],
                    "substitute_gt_ids": [],
                    "gt_id": [],
                    "gt_bbox": [],
                }
            # det匹配gt
            index_max_val = 0
            sim_score_max_val = -1e9
            match_gt_id = None
            for gt_bbox_tuple in gt_bboxes_val:
                gt_bbox, gt_bbox_track_id, gt_idx = gt_bbox_tuple
                if gt_matched[gt_idx]:
                    continue
                sim_score = similarity[dt_idx][gt_idx]
                if sim_score > sim_score_max_val:
                    sim_score_max_val = sim_score
                    index_max_val = gt_idx
                    match_gt_id = gt_bbox_track_id

            # 找替补ID：获取当前检测框可能替代的真实目标ID
            if gt_num > 0:
                match_gt_mask = similarity[dt_idx, :] > substitute_sim_threshold
                match_gt_idx = np.arange(gt_num)[match_gt_mask]
                substitute_gt_ids = gt_ids[match_gt_idx].tolist()
            else:
                substitute_gt_ids = []
            if match_gt_id:
                gt_idx = index_max_val
                match_dt_mask = similarity[:, gt_idx] > substitute_sim_threshold
                match_dt_idx = np.arange(dt_num)[match_dt_mask]
                substitute_dt_ids = dt_ids[match_dt_idx].tolist()
            else:
                substitute_dt_ids = []

            # det匹配上gt,tp
            if sim_score_max_val > sim_threshold:
                gt_bbox_tuple = gt_bboxes_val[index_max_val]
                gt_bbox, gt_bbox_track_id, gt_idx = gt_bbox_tuple
                gt_matched[gt_idx] = True
                assert gt_bbox_track_id == match_gt_id, "gt_bbox_track_id==match_gt_id"

                # 匹配上的dt框组织成轨迹
                dt_track_id_int = int(dt_track_id)
                if dt_track_id_int >= 0:
                    traj_track[dt_track_id]["frame_id"].append(frame_id)
                    traj_track[dt_track_id]["bbox"].append(dt_bbox)
                    traj_track[dt_track_id]["gt_id"].append(match_gt_id)
                    traj_track[dt_track_id]["substitute_gt_ids"].append(
                        substitute_gt_ids
                    )
                    traj_track[dt_track_id]["gt_bbox"].append(gt_bbox)
                else:
                    # 匹配上,但被杀掉的dt 框加入dt轨迹，表示错杀的TP
                    traj_track[dt_track_id]["frame_id"].append(frame_id)
                    traj_track[dt_track_id]["bbox"].append(dt_bbox)
                    traj_track[dt_track_id]["gt_id"].append(GT_ID_IN_DT.WRONG_KILL)
                    traj_track[dt_track_id]["substitute_gt_ids"].append(
                        substitute_gt_ids
                    )
                    traj_track[dt_track_id]["gt_bbox"].append(gt_bbox)

                # 匹配上的gt框组织成轨迹
                if match_gt_id not in gt_track:
                    gt_track[match_gt_id] = {
                        "frame_id": [frame_id],
                        "bbox": [gt_bbox],
                        "substitute_dt_ids": [substitute_dt_ids],
                        "track_id": [
                            (
                                dt_track_id
                                if dt_track_id != -1
                                else TRACK_ID_IN_GT.WRONG_KILL
                            )
                        ],
                    }
                else:
                    gt_track[match_gt_id]["frame_id"].append(frame_id)
                    gt_track[match_gt_id]["bbox"].append(gt_bbox)
                    gt_track[match_gt_id]["track_id"].append(
                        dt_track_id if dt_track_id != -1 else TRACK_ID_IN_GT.WRONG_KILL
                    )
                    gt_track[match_gt_id]["substitute_dt_ids"].append(substitute_dt_ids)

    # # for debug
    # output_dir = "./visual_result/"
    # if os.path.exists(output_dir):
    #     # Remove all files and subdirectories in the directory
    #     for root, dirs, files in os.walk(output_dir, topdown=False):
    #         for name in files:
    #             os.remove(os.path.join(root, name))
    #         for name in dirs:
    #             os.rmdir(os.path.join(root, name))
    #     os.rmdir(output_dir)
    # os.makedirs(output_dir)
    # for track_id, track_info in traj_track.items():
    #     visualize_bboxes(traj_track[track_id]['bbox'], traj_track[track_id]['gt_bbox'], int(track_id))

    return {"traj_track": traj_track, "gt_track": gt_track}
