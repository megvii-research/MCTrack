# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import cv2
import numpy as np

from utils.utils import orientation_similarity


def cal_rotation_iou_inbev(pose1, pose2):
    # pose : xyz_lwh_yaw
    box1 = np.array([pose1[0], pose1[1], pose1[3], pose1[4], pose1[6] * 180 / np.pi])
    box2 = np.array([pose2[0], pose2[1], pose2[3], pose2[4], pose2[6] * 180 / np.pi])
    base_xy = np.array(box1[:2])
    box1[:2] -= base_xy
    box2[:2] -= base_xy
    area1 = pose1[3] * pose1[4]
    area2 = pose2[3] * pose2[4]
    box1_inp = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
    box2_inp = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])
    int_pts = cv2.rotatedRectangleIntersection(tuple(box1_inp), tuple(box2_inp))[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        union = cv2.contourArea(order_pts)
        iou = union * 1.0 / (area1 + area2 - union)
    else:
        iou = 0
        union = 0
    return iou, union


def cal_rotation_gdiou_inbev(box_trk, box_det, cfg, cal_flag=None):
    """
    Info: This function calculates the rotation-based Generalized IoU (GDIoU) for two bounding boxes in bird's-eye view (BEV).
    Parameters:
        input:
            box_trk: Trajectory, the tracked object's trajectory.
            box_det: BBox, the detected bounding box.
            cfg: dict, configuration containing category mapping and threshold weights.
            cal_flag: str, specifies whether to use the predicted or fused pose for the tracked object. Options are "Predict" or "BackPredict".
        output:
            ro_gdiou: float, the rotation-based Generalized IoU (GDIoU) score between the tracked and detected bounding boxes in BEV.
    """
    # pose : xyz_lwh_yaw  pose1, pose2
    # corners: 8*3
    if cal_flag == "Predict":
        pose1 = box_trk.bboxes[-1].global_xyz_lwh_yaw_predict
        pose2 = box_det.global_xyz_lwh_yaw
    elif cal_flag == "BackPredict":
        pose1 = box_trk.bboxes[-1].global_xyz_lwh_yaw_fusion
        pose2 = box_det.global_xyz_lwh_yaw_last
    else:
        raise ValueError(f"Unexpected cal_flag value: {cal_flag}")

    corners1 = box_trk.bboxes[-1].transform_3dbox2corners(pose1)
    corners2 = box_det.transform_3dbox2corners(pose2)

    bev_idxes = [2, 3, 7, 6]
    bev_corners1 = corners1[bev_idxes, 0:2]
    bev_corners2 = corners2[bev_idxes, 0:2]
    pose1 = np.copy(pose1)
    pose2 = np.copy(pose2)
    iou, inter_area = cal_rotation_iou_inbev(pose1, pose2)
    union_points = np.concatenate([bev_corners1, bev_corners2], axis=0).astype(
        np.float64
    )
    union_points -= pose1[:2].reshape(1, 2)
    rect = cv2.minAreaRect(union_points.astype(np.float32))  # (x,y),(w,h),(angle)
    universe_area = rect[1][0] * rect[1][1]
    a_area = pose1[3] * pose1[4]
    b_area = pose2[3] * pose2[4]
    extra_area = universe_area - (a_area + b_area - inter_area)
    box_center_distance = (pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2
    union_distance = np.linalg.norm(np.array(rect[1])) ** 2

    box_trk_volume = pose1[3] * pose1[4] * pose1[5]
    box_det_volume = pose2[3] * pose2[4] * pose2[5]
    volume_ratio = (
        box_trk_volume / box_det_volume
        if box_trk_volume >= box_det_volume
        else box_det_volume / box_trk_volume
    )
    angle_ratio = orientation_similarity(pose1[6], pose2[6])
    category_num = cfg["CATEGORY_MAP_TO_NUMBER"][box_trk.bboxes[-1].category]

    w1 = cfg["THRESHOLD"]["BEV"]["WEIGHT_RO_GDIOU"][category_num]["w1"]
    w2 = cfg["THRESHOLD"]["BEV"]["WEIGHT_RO_GDIOU"][category_num]["w2"]

    ro_gdiou = (
        iou
        - w1 * extra_area / universe_area
        - w2 * box_center_distance / union_distance
    )

    return ro_gdiou


def cal_iou_inrv(box_trk, box_det, cfg=None, cal_flag=None):
    if (
        box_trk.bboxes[-1].camera_type != "CAM_FRONT"
        or box_det.camera_type != "CAM_FRONT"
    ):
        return float("-inf")
    boxA = box_trk.bboxes[-1].x1y1x2y2
    boxB = box_det.x1y1x2y2

    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def cal_giou_inrv(box_trk, box_det, cfg, cal_flag=None):
    if (
        box_trk.bboxes[-1].camera_type != "CAM_FRONT"
        or box_det.camera_type != "CAM_FRONT"
    ):
        return float("-inf")
    boxA = box_trk.bboxes[-1].x1y1x2y2
    boxB = box_det.x1y1x2y2
    iou = cal_iou_inrv(box_trk, box_det)

    x1, x2, y1, y2 = (
        boxA[0],
        boxA[1],
        boxA[2],
        boxA[3],
    )
    x3, x4, y3, y4 = boxB[0], boxB[1], boxB[2], boxB[3]

    area_C = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * (
        max(y1, y2, y3, y4) - min(y1, y2, y3, y4)
    )
    area_1 = (x2 - x1) * (y1 - y2)
    area_2 = (x4 - x3) * (y3 - y4)
    sum_area = area_1 + area_2
    w1 = x2 - x1
    w2 = x4 - x3
    h1 = y1 - y2
    h2 = y3 - y4
    W = min(x1, x2, x3, x4) + w1 + w2 - max(x1, x2, x3, x4)
    H = min(y1, y2, y3, y4) + h1 + h2 - max(y1, y2, y3, y4)
    Area = W * H
    add_area = sum_area - Area
    end_area = (area_C - add_area) / (area_C + 0.000001)
    giou = iou - end_area
    return giou


def cal_diou_inrv(box_trk, box_det, cfg, cal_flag=None):
    if (
        box_trk.bboxes[-1].camera_type != "CAM_FRONT"
        or box_det.camera_type != "CAM_FRONT"
    ):
        return float("-inf")
    boxA = np.array(box_trk.bboxes[-1].x1y1x2y2)
    boxB = np.array(box_det.x1y1x2y2)

    # cal the box's area of boxA and boxess
    boxes1Area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxes2Area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # cal Intersection
    left_up = np.maximum(boxA[:2], boxB[:2])
    right_down = np.minimum(boxA[2:], boxB[2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[0] * inter_section[1]
    union_area = boxes1Area + boxes2Area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    # cal outer boxes
    outer_left_up = np.minimum(boxA[:2], boxB[:2])
    outer_right_down = np.maximum(boxA[2:], boxB[2:])
    outer = np.maximum(outer_right_down - outer_left_up, 0.0)
    outer_diagonal_line = np.square(outer[0]) + np.square(outer[1])

    # cal center distance
    boxes1_center = (boxA[:2] + boxA[2:]) * 0.5
    boxes2_center = (boxB[:2] + boxB[2:]) * 0.5
    center_dis = np.square(boxes1_center[0] - boxes2_center[0]) + np.square(
        boxes1_center[1] - boxes2_center[1]
    )

    # cal diou
    dious = ious - center_dis / outer_diagonal_line

    return dious


def cal_sdiou_inrv(box_trk, box_det, cfg, cal_flag=None):
    if (
        box_trk.bboxes[-1].camera_type != "CAM_FRONT"
        or box_det.camera_type != "CAM_FRONT"
    ):
        return float("-inf")
    boxA = np.array(box_trk.bboxes[-1].x1y1x2y2)
    boxB = np.array(box_det.x1y1x2y2)

    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    outXmin, outYmin = min(boxA[0], boxB[0]), min(boxA[1], boxB[1])
    outXmax, outYmax = max(boxA[2], boxB[2]), max(boxA[3], boxB[3])

    inCenterxAx, inCenterxAy = (boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2
    inCenterxBx, inCenterxBy = (boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2

    trk_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    det_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    area_ratio = trk_area / det_area if det_area > trk_area else det_area / trk_area

    if (
        (trk_area + det_area - inter_area == 0)
        or ((outXmax - outXmin) ** 2 + (outYmax - outYmin) ** 2 == 0)
        or (area_ratio == 0)
    ):
        return 0

    distanceRatio = math.sqrt(
        (inCenterxBx - inCenterxAx) ** 2 + (inCenterxBy - inCenterxAy) ** 2
    ) / math.sqrt((outXmax - outXmin) ** 2 + (outYmax - outYmin) ** 2)
    distanceRatio = 1 - distanceRatio

    aspect_ratioA = (boxA[2] - boxA[0]) / (boxA[3] - boxA[1])
    aspect_ratioB = (boxB[2] - boxB[0]) / (boxB[3] - boxB[1])
    aspect_ratio = (
        aspect_ratioA / aspect_ratioB
        if aspect_ratioB > aspect_ratioA
        else aspect_ratioB / aspect_ratioA
    )

    sdiou_inbev = (
        inter_area / float(trk_area + det_area - inter_area)
    ) + area_ratio * distanceRatio * aspect_ratio

    return sdiou_inbev
