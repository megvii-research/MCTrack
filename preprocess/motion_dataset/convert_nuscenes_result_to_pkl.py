# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from nuscenes-devkit (https://github.com/nutonomy/nuscenes-devkit)
# Copyright (c) 2020 Motional. All Rights Reserved.
# ------------------------------------------------------------------------

import copy
import json
import os
import pickle
import numpy as np

from typing import Dict, List, Tuple
from nuscenes import NuScenes
from nuscenes.eval.common.loaders import (
    # load_prediction,
    load_gt,
    add_center_dist,
    filter_eval_boxes,
)
from nuscenes.eval.tracking.data_classes import TrackingConfig, TrackingBox
from nuscenes.eval.tracking.loaders import create_tracks
from nuscenes.eval.common.config import config_factory as track_configs
from nuscenes.eval.common.data_classes import EvalBoxes
from tqdm import tqdm


def quaternion_to_yaw(q):
    qw, qx, qy, qz = q

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return yaw

def process_data(data_variant, rotation_field, velocity_field):
    for sample_token in tqdm(data_variant['results'].keys(), desc="Processing data"):
        detections = data_variant['results'][sample_token]
        for detection in detections:
            # Rename the specified rotation and velocity fields
            detection['rotation'] = detection[rotation_field]
            detection['velocity'] = detection[velocity_field]
            
            # Remove all rotation and velocity fields except the renamed ones
            keys_to_remove = [
                'rotation_det', 'rotation_kalman_cv',
                'velocity_det', 'velocity_kalman_cv',
                'velocity_diff', 'velocity_curve'
            ]
            for key in keys_to_remove:
                if key in detection:
                    del detection[key]

def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False) \
        -> Tuple[EvalBoxes, Dict]:

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
        
    data_det = copy.deepcopy(data)
    data_kalman_cv = copy.deepcopy(data)
    data_diff = copy.deepcopy(data)
    data_curve = copy.deepcopy(data)
    process_data(data_det, 'rotation_det', 'velocity_det')
    process_data(data_kalman_cv, 'rotation_kalman_cv', 'velocity_kalman_cv')
    process_data(data_diff, 'rotation_det', 'velocity_diff')
    process_data(data_curve, 'rotation_det', 'velocity_curve')

    # Deserialize results and get meta data.
    all_results_det = EvalBoxes.deserialize(data_det['results'], box_cls)
    all_results_kalman_cv = EvalBoxes.deserialize(data_kalman_cv['results'], box_cls)
    all_results_diff = EvalBoxes.deserialize(data_diff['results'], box_cls)
    all_results_curve = EvalBoxes.deserialize(data_curve['results'], box_cls)
    meta = data['meta']

    return all_results_det, all_results_kalman_cv, all_results_diff, all_results_curve, meta

def convert_gt_to_pkl(tracks_gt, pkl_file):
    """
    Info: This function converts ground truth tracking results into a PKL format, organizing the data by categories and frames.
    Parameters:
        input:
            tracks_gt: dict, ground truth tracking data structured by scene tokens and frame information.
            pkl_file: str, path where the PKL file will be saved.
    """
    result = {}

    for scene_token, frame_data in tracks_gt.items():
        for frame_id, boxes in frame_data.items():
            for box in boxes:
                frame_id = box.sample_token

                tracking_id = box.tracking_id
                category = box.tracking_name
                confidence = box.tracking_score
                global_xyz = [
                    box.translation[0],
                    box.translation[1],
                    box.translation[2],
                ]
                lwh = [box.size[1], box.size[0], box.size[2]]
                yaw = [quaternion_to_yaw(box.rotation)]
                global_velocity = [box.velocity[0], box.velocity[1]]

                bbox = global_xyz + lwh + yaw + global_velocity
                gt_quality_level = None

                if category not in result:
                    result[category] = {}

                if frame_id not in result[category]:
                    result[category][frame_id] = {
                        "bbox": [],
                        "track_id": [],
                        "confidence": [],
                        "gt_quality_level": [],
                    }

                result[category][frame_id]["bbox"].append(bbox)
                result[category][frame_id]["track_id"].append(tracking_id)
                result[category][frame_id]["confidence"].append(confidence)
                result[category][frame_id]["gt_quality_level"].append(gt_quality_level)

    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)

    with open(pkl_file, "wb") as f:
        pickle.dump(result, f)


class TrackingEval:
    def __init__(
        self,
        config: TrackingConfig,
        result_path: str,
        eval_set: str,
        nusc_version: str,
        nusc_dataroot: str,
        verbose: bool = True,
        render_classes: List[str] = None,
    ):
        """
        Initialize a TrackingEval object.
        :param config: A TrackingConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param nusc_version: The version of the NuScenes dataset.
        :param nusc_dataroot: Path of the nuScenes dataset on disk.
        :param verbose: Whether to print to stdout.
        :param render_classes: Classes to render to disk or None.
        """
        self.cfg = config
        self.result_path = result_path
        self.eval_set = eval_set
        self.verbose = verbose
        self.render_classes = render_classes

        assert os.path.exists(result_path), "Error: The result file does not exist!"

        nusc = NuScenes(version=nusc_version, verbose=verbose, dataroot=nusc_dataroot)

        if verbose:
            print("Initializing nuScenes tracking evaluation")
        det_boxes, kalman_cv_boxes, diff_boxes, curve_boxes, self.meta = load_prediction(
            self.result_path,
            self.cfg.max_boxes_per_sample,
            TrackingBox,
            verbose=verbose,
        )
        gt_boxes = load_gt(nusc, self.eval_set, TrackingBox, verbose=verbose)

        det_boxes = add_center_dist(nusc, det_boxes)
        kalman_cv_boxes = add_center_dist(nusc, kalman_cv_boxes)
        diff_boxes = add_center_dist(nusc, diff_boxes)
        curve_boxes = add_center_dist(nusc, curve_boxes)
        
        gt_boxes = add_center_dist(nusc, gt_boxes)

        det_boxes = filter_eval_boxes(
            nusc, det_boxes, self.cfg.class_range, verbose=verbose
        )
        kalman_cv_boxes = filter_eval_boxes(
            nusc, kalman_cv_boxes, self.cfg.class_range, verbose=verbose
        )
        diff_boxes = filter_eval_boxes(
            nusc, diff_boxes, self.cfg.class_range, verbose=verbose
        )
        curve_boxes = filter_eval_boxes(
            nusc, curve_boxes, self.cfg.class_range, verbose=verbose
        )

        gt_boxes = filter_eval_boxes(
            nusc, gt_boxes, self.cfg.class_range, verbose=verbose
        )

        self.sample_tokens = gt_boxes.sample_tokens

        self.tracks_gt = create_tracks(gt_boxes, nusc, self.eval_set, gt=True)
        
        self.tracks_det = create_tracks(det_boxes, nusc, self.eval_set, gt=False)
        self.tracks_kalman_cv = create_tracks(kalman_cv_boxes, nusc, self.eval_set, gt=False)
        self.tracks_diff = create_tracks(diff_boxes, nusc, self.eval_set, gt=False)
        self.tracks_curve = create_tracks(curve_boxes, nusc, self.eval_set, gt=False)

    def get_tracks_gt(self):
        return self.tracks_gt, self.tracks_det, self.tracks_kalman_cv, self.tracks_diff, self.tracks_curve
    


if __name__ == "__main__":
    result_path = "/data/projects/MOT_baseline/base3dmot/results/nuscenes/20241006_180139/results_for_motion.json"
    nusc_path = "s3://wangxiyang/open_datasets/nuscenes/raw_data/"
    gt_pkl_path = "./data/nuscenes/eval_velocity/gt_results.pkl"
    
    det_pkl_path = "./data/nuscenes/eval_velocity/det_results.pkl"
    kalman_cv_pkl_path = "./data/nuscenes/eval_velocity/kalman_cv_results.pkl"
    diff_pkl_path = "./data/nuscenes/eval_velocity/diff_results.pkl"
    curve_pkl_path = "./data/nuscenes/eval_velocity/curve_results.pkl"
    
    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=result_path,
        eval_set="val",
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=nusc_path,
    )
    tracks_gt, tracks_det, tracks_kalman_cv, tracks_diff, tracks_curve = nusc_eval.get_tracks_gt()
    
    convert_gt_to_pkl(tracks_gt, gt_pkl_path)
    
    convert_gt_to_pkl(tracks_det, det_pkl_path)
    convert_gt_to_pkl(tracks_kalman_cv, kalman_cv_pkl_path)
    convert_gt_to_pkl(tracks_diff, diff_pkl_path)
    convert_gt_to_pkl(tracks_curve, curve_pkl_path)
