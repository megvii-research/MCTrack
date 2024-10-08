# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np

from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf
from scipy.signal import savgol_filter, find_peaks
from rdp import rdp


def norm_radian(radians):
    """
    Info: This function normalizes input radian values to the range [-pi, pi].
    Parameters:
        input:
            radians: Array-like or scalar radian values.
        output:
            radian_norm: Normalized radian values in the range [-pi, pi], either as a scalar or array.
    """
    radians = np.array(radians).reshape(-1)
    radians_norm = []
    for radian in radians:
        n = np.floor(radian / (2 * np.pi))
        radian = radian - n * 2 * np.pi
        radians_norm.append(radian)
    radians_norm = np.array(radians_norm)
    if len(radians_norm) == 1:
        radian_norm = radians_norm[0]
    else:
        radian_norm = radians_norm
    if radian_norm > np.pi:
        return radian_norm - np.pi
    else:
        return radian_norm


def norm_realative_radian(radians_diff):
    """
    Info: This function normalizes relative radian differences to the range [-pi, pi].
    Parameters:
        input:
            radians_diff: Array-like or scalar relative radian differences.
        output:
            radians_diff_norm: Normalized relative radian values in the range [-pi, pi], either as a scalar or array.
    """
    radians_diff = np.array(radians_diff).reshape(-1)
    radians_diff_norm = []
    for rad in radians_diff:
        if rad < -np.pi:
            rad += 2 * np.pi
        elif rad > np.pi:
            rad -= 2 * np.pi
        radians_diff_norm.append(rad)
    radians_diff_norm = np.array(radians_diff_norm)
    if len(radians_diff_norm) == 1:
        return radians_diff_norm[0]
    else:
        return radians_diff_norm


def sg_filter_velocity(v_xyz, window=15, order=3):
    if len(v_xyz) < 2 * window:
        window = len(v_xyz) // 4 * 2 + 1
        order = window // 2
    v_xyz = np.array(v_xyz)
    bf_v_xyz = []
    for i in range(v_xyz.shape[1]):
        bf_v_xyz.append(savgol_filter(v_xyz[:, i], window, order, mode="mirror"))
    bf_v_xyz = np.vstack(bf_v_xyz).transpose((1, 0))

    return bf_v_xyz


def cal_moving_interval(peak, half_win, offset, gt_len, dt_len):
    dt_peak = peak + offset
    if dt_peak > dt_len or peak > gt_len or dt_peak < 0 or peak < 0:
        return [-1, -1, -1, -1, -1]
    gt_left_width, gt_right_width = (
        peak - max(0, peak - half_win),
        min(peak + half_win, gt_len) - peak,
    )
    dt_left_width, dt_right_width = (
        dt_peak - max(0, dt_peak - half_win),
        min(dt_peak + half_win, dt_len) - dt_peak,
    )
    dt_left_width, dt_right_width = max(0, dt_left_width), max(0, dt_right_width)
    left_width, right_width = min(dt_left_width, gt_left_width), min(
        dt_right_width, gt_right_width
    )

    gt_left, gt_right = peak - left_width, peak + right_width
    dt_left, dt_right = dt_peak - left_width, dt_peak + right_width
    win = left_width + right_width
    assert gt_left >= 0, f"gt_left:{gt_left} >=0 ?"
    assert gt_right <= gt_len, f"gt_right:{gt_right}<=gt_len:{gt_len} ?"
    assert dt_left >= 0, f"dt_left:{dt_left} >=0 ?"
    assert dt_right <= dt_len >= 0, f"dt_right:{dt_right}<=dt_len:{dt_len}?"
    return gt_left, gt_right, dt_left, dt_right, win


def find_line_middle_points(gt, interval_width_thresh=30, velocity_thresh=1.0):
    points = [[idx, p] for idx, p in enumerate(gt)]
    key_points = rdp(points, epsilon=1.0)
    key_points = np.array(key_points).reshape(-1, 2)

    middle_idx = (key_points[1:, 0] + key_points[0:-1, 0]) // 2
    interval_width = key_points[1:, 0] - key_points[0:-1, 0]
    velocity_diff = np.abs(key_points[1:, 1] - key_points[0:-1, 1])

    selected_pair = []
    for idx, width, v_diff in zip(middle_idx, interval_width, velocity_diff):
        if width > interval_width_thresh and v_diff > velocity_thresh:
            selected_pair.append((int(idx), width))
    selected_pair = sorted(selected_pair, key=lambda x: x[1], reverse=True)
    selected_idx = [sp[0] for sp in selected_pair]
    return selected_idx


def judge_static_traj(
    gt_boxes, velocity_type, static_velocity_thresh=1.0, static_movement_thresh=1.5
):
    gt_v_xy = np.array([box[-2:] for box in gt_boxes])
    gt_v = np.linalg.norm(gt_v_xy, axis=-1)
    gt_xyz = np.array([box[:3] for box in gt_boxes])
    movement = np.linalg.norm(gt_xyz - np.mean(gt_xyz, axis=0), axis=-1)
    if max(gt_v) < static_velocity_thresh or max(movement) < static_movement_thresh:
        return True
    else:
        return False


def cal_delay(gt_v_xyz, dt_v_xyz, window=20, at_least_win=1, max_peaks=3):
    delay_times = []
    gt = np.linalg.norm(gt_v_xyz, axis=-1)
    if len(gt) < window:
        return delay_times
    max_value = np.max(gt)
    min_value = np.min(gt)
    difference = abs(max_value - min_value)
    if difference < 0.5:
        return delay_times
    # if np.std(gt) < 1.0:
    #     return delay_times

    peaks1, _ = find_peaks(gt, width=1, distance=10)
    peaks2 = find_line_middle_points(gt, 2, 1)
    points = list(peaks1[:max_peaks]) + list(peaks2[:max_peaks])

    if len(points) == 0:
        return delay_times

    half_win = window // 2
    dt_v_xyz_filter = sg_filter_velocity(dt_v_xyz)
    dt = np.linalg.norm(dt_v_xyz_filter, axis=-1)
    offsets = range(0, window)
    for point in points:
        plt_offset, plt_std = [], []
        min_std_offset = None
        min_std = 1e9
        for offset in offsets:
            gt_left, gt_right, dt_left, dt_right, win = cal_moving_interval(
                point, half_win, offset, len(gt), len(dt)
            )
            if win > at_least_win:
                gt_snap, dt_snap = gt[gt_left:gt_right], dt[dt_left:dt_right]
                snap_std = np.std(np.array(gt_snap - dt_snap)) + np.average(
                    np.abs(np.array(gt_snap - dt_snap))
                )
                plt_offset.append(offset)
                plt_std.append(snap_std)
                if snap_std < min_std:
                    min_std = snap_std
                    min_std_offset = offset
        if not min_std_offset:
            continue
        delay_times.append(min_std_offset * 0.5)

    return delay_times


def cal_velocity_delay(gt_trajs, dt_trajs):
    dt_track_id_track_id_box = {}
    for track_id, dt_traj in dt_trajs.items():
        frame_id2box = {}
        for dt_box, det_frame_id in zip(dt_traj["bbox"], dt_traj["frame_id"]):
            frame_id2box[det_frame_id] = dt_box
        dt_track_id_track_id_box[track_id] = frame_id2box

    time_delays = []
    for gt_id, gt_traj in gt_trajs.items():
        single_gt_boxes, single_dt_boxes = [], []
        for track_id, gt_box, gt_frame_id in zip(
            gt_traj["track_id"], gt_traj["bbox"], gt_traj["frame_id"]
        ):
            track_id_int = int(track_id)
            if track_id_int < 0:
                continue
            dt_box = dt_track_id_track_id_box[track_id][gt_frame_id]
            single_gt_boxes.append(gt_box)
            single_dt_boxes.append(dt_box)

        if len(single_dt_boxes) > 0:
            gt_v_xy = np.array([[box[-2], box[-1]] for box in single_gt_boxes])
            dt_v_xy = np.array([[box[-2], box[-1]] for box in single_dt_boxes])
            time_delay = cal_delay(gt_v_xy, dt_v_xy)
            time_delays.extend(time_delay)
    return time_delays


def cal_velocity_smoothness(single_dt_traj, velocity_type="velocity_global"):
    v_xyz = np.array([[box[-2], box[-1], 0] for box in single_dt_traj["bbox"]])
    if len(v_xyz) != 0:
        v_xyz_filter = sg_filter_velocity(v_xyz, window=15, order=3)
        velocity_smooth_error = np.linalg.norm(np.abs(v_xyz - v_xyz_filter), axis=-1)
        return velocity_smooth_error
    return None


def cal_velocity_angle_norm(gt_box, dt_box, velocity_type="velocity_global"):
    gt_v = dict(x=gt_box[-2], y=gt_box[-1])
    dt_v = dict(x=dt_box[-2], y=dt_box[-1])
    gt_v["norm"] = (gt_v["y"] ** 2 + gt_v["x"] ** 2) ** 0.5
    dt_v["norm"] = (dt_v["y"] ** 2 + dt_v["x"] ** 2) ** 0.5
    gt_v["degree"] = np.arctan2(gt_v["y"], gt_v["x"]) * 180 / np.pi
    dt_v["degree"] = np.arctan2(dt_v["y"], dt_v["x"]) * 180 / np.pi
    velocity_error = {k: np.abs(gt_v[k] - dt_v[k]) for k in gt_v.keys()}
    velocity_error["degree"] = min(
        velocity_error["degree"], np.abs(gt_v["degree"] + 360 - dt_v["degree"])
    )
    velocity_error["degree"] = min(
        velocity_error["degree"], np.abs(gt_v["degree"] - (dt_v["degree"] + 360))
    )

    return velocity_error


def cal_orientation_error(gt_box, dt_box):
    gt_yaw = gt_box[-3]
    dt_yaw = dt_box[-3]
    gt_yaw = norm_radian(gt_yaw)
    dt_yaw = norm_radian(dt_yaw)
    yaw_error = np.abs(norm_realative_radian(gt_yaw - dt_yaw))
    return yaw_error


def cal_size_error(gt_box, dt_box):
    gt_lwh = gt_box[3:6]
    dt_lwh = dt_box[3:6]
    size_error = np.abs(np.array(gt_lwh) - np.array(dt_lwh))
    return size_error


def cal_translation_error(gt_box, dt_box):
    gt_xyz = gt_box[0:3]
    dt_xyz = dt_box[0:3]
    trans_error = np.abs(np.array(gt_xyz) - np.array(dt_xyz))
    return trans_error


def compute_velocity_metrics_mean(velocity_metrics):
    # Initialize a new dictionary to hold the mean values
    velocity_metrics_mean = {}

    # Iterate over each key-value pair in the original dictionary
    for key, values in velocity_metrics.items():
        if len(values) > 0:  # Ensure the list is not empty
            # Compute the mean value of the list
            mean_value = np.mean(values)
            # Add the mean value to the new dictionary
            velocity_metrics_mean[key] = mean_value
        else:
            # If the list is empty, you might choose to set a default value or skip it
            velocity_metrics_mean[key] = None  # or np.nan if you prefer

    return velocity_metrics_mean


def filter_invalid_values(velocity_metrics):
    for key in velocity_metrics:
        velocity_metrics[key] = [
            val
            for val in velocity_metrics[key]
            if val is not None and not np.isnan(val)
        ]
    return velocity_metrics


def metric_for_velocity(gt_trajs, dt_trajs, velocity_cfg):
    """
    return velocity_metric:
    --vae: velocity angle error:
    --vne: velocity magnitude norm error
    --vse: velocity smooth error
    --vde: velociyt delay error
    --vaie: velocity angle of inverse error:(angle_inverse is box that angle error >90 )
    --vir: velocity reverse ratio
    --vae_s: vae of static object
    --vne_s: vne of static object
    --aoe: average orientation error
    --ale: average length error
    --awe: average width error
    --ahe: average height error
    --ade: average dimension error
    --axe: average distance error in  axis's X of sensor coordinate
    --aye: average distance error in  axis's Y of sensor coordinate
    --aze: average distance error in  axis's Z of sensor coordinate
    --ate: average translation error in   sensor coordinate
    """
    gt_frame_id_track_id_box = {}
    gt_static = {}
    velocity_type = velocity_cfg["velocity_type"]
    for gt_id, gt_traj in gt_trajs.items():
        frame_id2box = {}
        for gt_box, gt_frame_id in zip(gt_traj["bbox"], gt_traj["frame_id"]):
            # import ipdb; ipdb.set_trace()
            frame_id2box[gt_frame_id] = gt_box
        gt_frame_id_track_id_box[gt_id] = frame_id2box
        gt_static[gt_id] = judge_static_traj(
            gt_traj["bbox"],
            velocity_type,
            velocity_cfg["static_velocity_thresh"],
            velocity_cfg["static_movement_thresh"],
        )

    metric_names = ["vae", "vne", "vde", "vse", "vaie", "vir", "aoe"]
    metric_names += ["ale", "awe", "ahe", "ade"]
    velocity_metrics = {k: [] for k in metric_names}

    for dt_id, dt_traj in dt_trajs.items():
        for match_gt_id, dt_box, det_frame_id in zip(
            dt_traj["gt_id"], dt_traj["bbox"], dt_traj["frame_id"]
        ):
            if gt_static[match_gt_id] is False:
                gt_box = gt_frame_id_track_id_box[match_gt_id][det_frame_id]
                velocity_error = cal_velocity_angle_norm(gt_box, dt_box, velocity_type)
                if velocity_error["norm"] is None:
                    import ipdb

                    ipdb.set_trace()
                velocity_metrics["vne"].append(velocity_error["norm"])
                if velocity_error["degree"] > velocity_cfg["angle_thresh"]:
                    velocity_metrics["vir"].append(1)
                    velocity_metrics["vaie"].append(velocity_error["degree"])
                else:
                    velocity_metrics["vir"].append(0)
                    velocity_metrics["vae"].append(velocity_error["degree"])
            velocity_metrics["aoe"].append(cal_orientation_error(gt_box, dt_box))
            error_lwh = cal_size_error(gt_box, dt_box)
            velocity_metrics["ale"].append(error_lwh[0])  # length
            velocity_metrics["awe"].append(error_lwh[1])  # width
            velocity_metrics["ahe"].append(error_lwh[2])  # height
            velocity_metrics["ade"].append(np.linalg.norm(error_lwh))  # dimension

        velocity_smoothness = cal_velocity_smoothness(dt_traj, velocity_type)
        if velocity_smoothness is not None:
            velocity_smoothness = velocity_smoothness.tolist()
            velocity_metrics["vse"].extend(velocity_smoothness)

    velocity_delay = cal_velocity_delay(gt_trajs, dt_trajs)
    velocity_metrics["vde"].extend(velocity_delay)
    velocity_metrics = filter_invalid_values(velocity_metrics)
    velocity_metrics_mean = compute_velocity_metrics_mean(velocity_metrics)
    return velocity_metrics_mean
