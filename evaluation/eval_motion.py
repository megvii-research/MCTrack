# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import argparse
import pandas as pd
import yaml
import multiprocessing

from motion_evaluation.compare_base import CompareBase


def load_yaml_config(file_path):
    with open(file_path, "r") as file:
        # Load the YAML file using pyyaml
        config = yaml.safe_load(file)
    return config


def eval_velocity_metric(eval_cfg, gt_frame_info, dt_frame_info, velocity_type):
    eval_cls = eval_cfg["velocity_cfg"]["eval_cls"]
    eval_res = {}
    for cls in eval_cls:
        gt_frame_list = gt_frame_info[cls]
        dt_frame_list = dt_frame_info[cls]
        cmp = CompareBase(
            gt=gt_frame_list,
            dt=dt_frame_list,
            gt_mask={},
            eval_type=cls,
            dt_interval=eval_cfg["match_cfg"]["dt_interval"],
            video_name=eval_cfg["match_cfg"]["video_name"],
            metric=eval_cfg["match_cfg"]["metric"],
            threshold=eval_cfg["threshold"]["match_threshold"],
            sensor=eval_cfg["match_cfg"]["sensor"],
            eval_cfg=eval_cfg,
        )

        mot_eval = cmp.get_mot_eval_results(eval_type=cls)
        eval_res[cls] = mot_eval

    df = pd.DataFrame(eval_res).T
    desired_order = [
        "tp_all",
        "vae",
        "vne",
        "vde",
        "vse",
        "vaie",
        "vir",
        "aoe",
        "awe",
        "ale",
        "ahe",
        "ade",
    ]
    df = df[desired_order]

    total_tp_all = df["tp_all"].sum()
    weights = df["tp_all"] / total_tp_all
    weighted_avg_metrics = (df.drop(columns=["tp_all"]).T * weights).T.sum()
    total_metrics = pd.Series(
        [total_tp_all] + weighted_avg_metrics.tolist(), index=df.columns
    )
    df.loc["Total"] = total_metrics

    print(f"Evaluating velocity type: {velocity_type}")
    print(df)
    print("\n")


def eval_velocity_metric_wrapper(args):
    eval_cfg, gt_frame_info_path, dt_frame_info_path, velocity_type = args

    # Load data inside the process to avoid pickling large data structures
    gt_frame_info = pd.read_pickle(gt_frame_info_path)
    dt_frame_info = pd.read_pickle(dt_frame_info_path)

    eval_velocity_metric(eval_cfg, gt_frame_info, dt_frame_info, velocity_type)


def main(yaml_file_path):
    eval_cfg = load_yaml_config(yaml_file_path)

    gt_frame_info_path = eval_cfg["Date_Path"]["gt_frame_info_path"]
    det_frame_info_path = eval_cfg["Date_Path"]["det_frame_info_path"]
    kalman_cv_frame_info_path = eval_cfg["Date_Path"]["kalman_cv_frame_info_path"]
    diff_frame_info_path = eval_cfg["Date_Path"]["diff_frame_info_path"]
    curve_frame_info_path = eval_cfg["Date_Path"]["curve_frame_info_path"]

    # Prepare arguments for each process
    tasks = [
        (eval_cfg, gt_frame_info_path, det_frame_info_path, "velocity_det"),
        (eval_cfg, gt_frame_info_path, kalman_cv_frame_info_path, "velocity_kalman_cv"),
        (eval_cfg, gt_frame_info_path, diff_frame_info_path, "velocity_diff"),
        (eval_cfg, gt_frame_info_path, curve_frame_info_path, "velocity_curve"),
    ]

    processes = []
    for args in tasks:
        p = multiprocessing.Process(target=eval_velocity_metric_wrapper, args=(args,))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process motion evaluation configuration file."
    )
    parser.add_argument(
        "-y",
        "--yaml_file_path",
        type=str,
        default="./config/nuscenes_motion_eval.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.yaml_file_path)
