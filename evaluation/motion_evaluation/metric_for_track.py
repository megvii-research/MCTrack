# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import copy
import json
import numpy as np
from collections import Counter, defaultdict
from motion_evaluation.frame_matching import GT_ID_IN_DT, TRACK_ID_IN_GT


def filter_out_index(func, iter_list):
    res = []
    for idx, item in enumerate(iter_list):
        if func(item):
            res.append(item)
    return res


def cal_id_switch_by_dt_list(List, substitute_gt_ids=None):
    tp = 0
    List = copy.deepcopy(List)
    List = filter_out_index(lambda i: True, List)
    tp = len(List)
    return tp


def metric_for_mot(dt_trajs):
    tp_all = 0

    for dt_id in dt_trajs.keys():
        tp = cal_id_switch_by_dt_list(
            copy.deepcopy(dt_trajs[dt_id]["gt_id"]),
            copy.deepcopy(dt_trajs[dt_id].get("substitute_gt_ids", None)),
        )
        tp_all += tp

    return {
        "tp_all": tp_all,
    }
