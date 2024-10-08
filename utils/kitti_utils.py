# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import copy, os

from utils.file_process import *


def get_global_yaw(yaw, pose):
    new_yaw = (np.pi - yaw) + np.pi / 2
    ang = get_registration_angle(pose)
    global_yaw = new_yaw + ang
    return global_yaw


def get_lidar_yaw(yaw, pose):
    ang = get_registration_angle(pose)
    new_yaw = yaw + ang
    lidar_yaw = -new_yaw - np.pi / 2
    return lidar_yaw


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


def corners3d_to_img_boxes(P2, corners3d):
    """
    Info: This function projects 3D bounding box corners from the rectified camera coordinate system into 2D image coordinates, producing bounding boxes and corners in image space.
    Parameters:
        input:
            P2: (3, 4) projection matrix from 3D to 2D image coordinates.
            corners3d: (N, 8, 3), 3D coordinates of the 8 corners of the bounding boxes in rectified coordinates.
        output:
            img_boxes: (N, 4), projected 2D bounding boxes in [x1, y1, x2, y2] format.
            boxes_corner: (N, 8, 2), 2D image coordinates of all 8 corners of the 3D bounding boxes.
            is_invalid: Boolean mask indicating invalid boxes.
    """
    sample_num = corners3d.shape[0]
    corners3d_hom = np.concatenate(
        (corners3d, np.ones((sample_num, 8, 1))), axis=2
    )  # (N, 8, 4)

    img_pts = np.matmul(corners3d_hom, P2.T)  # (N, 8, 3)

    x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
    x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
    x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

    img_boxes = np.concatenate(
        (x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)),
        axis=1,
    )
    boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)
    is_invalid = bbox_is_invalid(img_boxes)
    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, 1242 - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, 375 - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, 1242 - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, 375 - 1)

    return img_boxes, boxes_corner, is_invalid


def bb3d_2_bb2d(bb3d, P2):
    x, y, z, l, w, h, yaw = (
        bb3d[0],
        bb3d[1],
        bb3d[2],
        bb3d[3],
        bb3d[4],
        bb3d[5],
        bb3d[6],
    )

    pt1 = [l / 2, 0, w / 2, 1]
    pt2 = [l / 2, 0, -w / 2, 1]
    pt3 = [-l / 2, 0, w / 2, 1]
    pt4 = [-l / 2, 0, -w / 2, 1]
    pt5 = [l / 2, -h, w / 2, 1]
    pt6 = [l / 2, -h, -w / 2, 1]
    pt7 = [-l / 2, -h, w / 2, 1]
    pt8 = [-l / 2, -h, -w / 2, 1]
    pts = np.array([[pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8]])
    transpose = np.array(
        [
            [np.cos(np.pi - yaw), 0, -np.sin(np.pi - yaw), x],
            [0, 1, 0, y],
            [np.sin(np.pi - yaw), 0, np.cos(np.pi - yaw), z],
            [0, 0, 0, 1],
        ]
    )
    pts = np.matmul(pts, transpose.T)
    box, _, is_invalid = corners3d_to_img_boxes(P2, pts[:, :, 0:3])

    return box, is_invalid


def bbox_is_invalid(img_boxes):
    intervals = np.array([[0, 1241], [0, 374], [0, 1241], [0, 374]])

    def check_point_in_interval(point, interval):
        return interval[0] <= point <= interval[1]

    left_top = img_boxes[0, :2]
    right_bottom = img_boxes[0, 2:]

    left_top_flag = False
    right_bottom_flag = False

    left_top_x_in_interval = check_point_in_interval(left_top[0], intervals[0])
    left_top_y_in_interval = check_point_in_interval(left_top[1], intervals[1])
    left_top_flag = (not left_top_x_in_interval and left_top_y_in_interval) or (
        not left_top_y_in_interval and left_top_x_in_interval
    )

    right_bottom_x_in_interval = check_point_in_interval(right_bottom[0], intervals[2])
    right_bottom_y_in_interval = check_point_in_interval(right_bottom[1], intervals[3])
    right_bottom_flag = (
        not right_bottom_x_in_interval and right_bottom_y_in_interval
    ) or (not right_bottom_y_in_interval and right_bottom_x_in_interval)

    is_invalid = False
    if left_top_flag and right_bottom_flag:
        is_invalid = True
    elif (
        (not left_top_x_in_interval and not left_top_y_in_interval)
        and (right_bottom_x_in_interval and right_bottom_y_in_interval)
    ) or (
        (not right_bottom_x_in_interval and not right_bottom_y_in_interval)
        and (left_top_x_in_interval and left_top_y_in_interval)
    ):
        is_invalid = False

    in_intervals = (img_boxes >= intervals[:, 0]) & (img_boxes <= intervals[:, 1])
    is_invalid = not np.any(in_intervals)

    in_intervals_lower = img_boxes >= intervals[:, 1]
    in_intervals_upper = img_boxes <= intervals[:, 0]
    is_invalid = np.any(in_intervals_lower) or np.any(in_intervals_upper)

    return is_invalid


def transform_bbox_to_lidar(bbox, global2lidar):
    global_bbox = bbox.global_xyz_lwh_yaw_fusion
    lidar_bbox = bbox.transform_bbox_global2lidar(global_bbox, global2lidar)
    lidar_bbox[-1] = get_lidar_yaw(global_bbox[-1], global2lidar)
    lidar_bbox[2] -= lidar_bbox[5] / 2

    return lidar_bbox


def transform_bbox_to_kitti(bbox, transform_matrix):
    global2lidar = np.array(transform_matrix["global2lidar"])
    lidar2camera = np.array(
        transform_matrix["cameras_transform_matrix"]["CAM_FRONT"]["lidar2camera"]
    )
    camera2image = np.array(
        transform_matrix["cameras_transform_matrix"]["CAM_FRONT"]["camera2image"]
    )
    lidar_bbox = transform_bbox_to_lidar(bbox, global2lidar)
    camera_bbox = bbox.transform_bbox_lidar2camera(lidar_bbox, lidar2camera)
    kitti_bbox_3d = copy.deepcopy(camera_bbox)
    kitti_bbox_2d, is_invalid = bb3d_2_bb2d(kitti_bbox_3d, camera2image)

    return kitti_bbox_2d, kitti_bbox_3d, is_invalid


def save_scene_results(scene_id, scene_result, cfg):
    save_path = os.path.join(cfg["SAVE_PATH"], cfg["DETECTOR"], cfg["SPLIT"], "data")
    os.makedirs(save_path, exist_ok=True)

    save_name = os.path.join(save_path, scene_id + ".txt")

    save_image_dir = os.path.join(
        cfg["SAVE_PATH"], cfg["DETECTOR"], cfg["SPLIT"], "image", scene_id
    )
    os.makedirs(save_image_dir, exist_ok=True)

    with open(save_name, "a") as file:
        for frame_id, frame_data in scene_result.items():
            trajs = frame_data["trajs"]
            transform_matrix = frame_data["transform_matrix"]
            if trajs:
                for track_id, bbox in trajs.items():
                    kitti_bbox_2d, kitti_bbox_3d, _ = transform_bbox_to_kitti(
                        bbox, transform_matrix
                    )
                    str_to_write = (
                        f"{frame_id} {track_id} {bbox.category.capitalize()} -1 -1 -10 "
                        f"{kitti_bbox_2d[0][0]:.4f} {kitti_bbox_2d[0][1]:.4f} {kitti_bbox_2d[0][2]:.4f} {kitti_bbox_2d[0][3]:.4f} "
                        f"{kitti_bbox_3d[5]:.4f} {kitti_bbox_3d[4]:.4f} {kitti_bbox_3d[3]:.4f} "
                        f"{kitti_bbox_3d[0]:.4f} {kitti_bbox_3d[1]:.4f} {kitti_bbox_3d[2]:.4f} "
                        f"{kitti_bbox_3d[6]:.4f} {bbox.det_score:.4f}\n"
                    )
                    file.write(str_to_write)


def save_results_kitti(tracking_results, cfg):
    for scene_id in sorted(tracking_results.keys()):
        save_scene_results(scene_id, tracking_results[scene_id], cfg)
