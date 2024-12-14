import os
from datetime import datetime


def eval_waymo(cfg, save_path):

    cmd_merge_category = f"python evaluation/static_evaluation/waymo/pred_bin.py --obj_types vehicle,pedestrian,cyclist --result_folder {save_path}"
    cmd_eval = f"/data/pkgs/waymo-open-dataset-master/src/bazel-bin/waymo_open_dataset/metrics/tools/compute_tracking_metrics_main {save_path}/bin/pred.bin data/waymo/datasets/val/validation_ground_truth_objects_gt.bin"

    if cfg["SPLIT"] == "test":
        cmd_merge_category = f"{cmd_merge_category} --test"

    os.system(f"{cmd_merge_category}")
    os.system(f"{cmd_eval}")
