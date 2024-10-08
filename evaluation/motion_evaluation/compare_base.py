# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

from motion_evaluation.frame_matching import matching_v5
from motion_evaluation.metric_for_track import metric_for_mot
from motion_evaluation.metric_for_velocity import metric_for_velocity


class CompareBase:
    def __init__(
        self,
        gt,
        dt,
        gt_mask,
        eval_type,
        dt_interval,
        video_name,
        metric="bev_iou",
        threshold=0.5,
        sensor="camera",
        eval_cfg=None,
    ):
        self.gt = gt
        self.dt = dt
        self.gt_mask = gt_mask
        self.eval_type = eval_type
        self.dt_interval = dt_interval
        self.video_name = video_name
        self.metric = metric
        self.threshold = threshold
        self.sensor = sensor
        self.eval_cfg = eval_cfg
        self.dt_track = None
        self.gt_track = None
        self.gt_results = None
        self.dt_results = None

    def get_mot_eval_results(self, eval_type=""):
        VRU_types = ["motorcycle", "bicycle", "tricycle", "cyclist", "pedestrian"]
        if eval_type not in VRU_types:
            track_dict = matching_v5(
                dt=self.dt,
                gt=self.gt,
                mask=self.gt_mask,
                threshold=self.threshold,
                match_min_max=False,
                metric=self.metric,
                substitute_sim_threshold=0,  ## 无替补id
            )
        else:
            # matching_v5 has  track_id or gt_id
            track_dict = matching_v5(
                dt=self.dt,
                gt=self.gt,
                mask=self.gt_mask,
                threshold=self.threshold,
                match_min_max=False,
                metric=self.metric,
                substitute_sim_threshold=self.threshold / 2,
            )

        self.dt_track, self.gt_track = track_dict["traj_track"], track_dict["gt_track"]

        mot_eval = metric_for_mot(self.dt_track)

        velocity_eval = metric_for_velocity(
            self.gt_track, self.dt_track, self.eval_cfg["velocity_cfg"]
        )
        mot_eval.update(velocity_eval)

        return mot_eval
