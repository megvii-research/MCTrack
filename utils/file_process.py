# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import yaml, os

from easydict import EasyDict as edict


def load_config(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)
        config_dict_str_keys = convert_keys_to_str(config_dict)
        cfg = edict(config_dict_str_keys)

    return cfg


def convert_keys_to_str(d):
    if isinstance(d, dict):
        return {str(k): convert_keys_to_str(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_keys_to_str(i) for i in d]
    else:
        return d


def mkdir_if_inexistence(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)


def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
    color = [int((p * (label**2 - label + 1)) % 255) for p in palette]
    return tuple(color)
