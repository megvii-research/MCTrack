# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

from .bbox import BBox
from typing import List


class Frame:
    def __init__(
        self,
        frame_id,
        cur_sample_token=None,
        timestamp=None,
        transform_matrix=None,
    ):

        self.frame_id = frame_id
        self.cur_sample_token = cur_sample_token
        self.timestamp = timestamp
        self.transform_matrix = transform_matrix

        self.bboxes: List[BBox] = []
