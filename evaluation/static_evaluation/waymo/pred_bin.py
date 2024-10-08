"""
    cited from Immortaltracker
    https://github.com/esdolo/ImmortalTracker
    author: Qitai Wang
"""

import os, numpy as np
import argparse, json

from copy import deepcopy
from tqdm import tqdm
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2


class BBox:
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, o=None):
        self.x = x  # center x
        self.y = y  # center y
        self.z = z  # center z
        self.h = h  # height
        self.w = w  # width
        self.l = l  # length
        self.o = o  # orientation
        self.s = None  # detection score

    def __str__(self):
        return "x: {}, y: {}, z: {}, heading: {}, length: {}, width: {}, height: {}, score: {}".format(
            self.x, self.y, self.z, self.o, self.l, self.w, self.h, self.s
        )

    @classmethod
    def bbox2dict(cls, bbox):
        return {
            "center_x": bbox.x,
            "center_y": bbox.y,
            "center_z": bbox.z,
            "height": bbox.h,
            "width": bbox.w,
            "length": bbox.l,
            "heading": bbox.o,
        }

    @classmethod
    def bbox2array(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h])
        else:
            return np.array(
                [bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h, bbox.s]
            )

    @classmethod
    def array2bbox(cls, data):
        bbox = BBox()
        bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox

    @classmethod
    def dict2bbox(cls, data):
        bbox = BBox()
        bbox.x = data["center_x"]
        bbox.y = data["center_y"]
        bbox.z = data["center_z"]
        bbox.h = data["height"]
        bbox.w = data["width"]
        bbox.l = data["length"]
        bbox.o = data["heading"]
        if "score" in data.keys():
            bbox.s = data["score"]
        return bbox

    @classmethod
    def copy_bbox(cls, bboxa, bboxb):
        bboxa.x = bboxb.x
        bboxa.y = bboxb.y
        bboxa.z = bboxb.z
        bboxa.l = bboxb.l
        bboxa.w = bboxb.w
        bboxa.h = bboxb.h
        bboxa.o = bboxb.o
        bboxa.s = bboxb.s
        return

    @classmethod
    def box2corners2d(cls, bbox):
        """the coordinates for bottom corners"""
        bottom_center = np.array([bbox.x, bbox.y, bbox.z - bbox.h / 2])
        cos, sin = np.cos(bbox.o), np.sin(bbox.o)
        pc0 = np.array(
            [
                bbox.x + cos * bbox.l / 2 + sin * bbox.w / 2,
                bbox.y + sin * bbox.l / 2 - cos * bbox.w / 2,
                bbox.z - bbox.h / 2,
            ]
        )
        pc1 = np.array(
            [
                bbox.x + cos * bbox.l / 2 - sin * bbox.w / 2,
                bbox.y + sin * bbox.l / 2 + cos * bbox.w / 2,
                bbox.z - bbox.h / 2,
            ]
        )
        pc2 = 2 * bottom_center - pc0
        pc3 = 2 * bottom_center - pc1

        return [pc0.tolist(), pc1.tolist(), pc2.tolist(), pc3.tolist()]

    @classmethod
    def box2corners3d(cls, bbox):
        """the coordinates for bottom corners"""
        center = np.array([bbox.x, bbox.y, bbox.z])
        bottom_corners = np.array(BBox.box2corners2d(bbox))
        up_corners = 2 * center - bottom_corners
        corners = np.concatenate([up_corners, bottom_corners], axis=0)
        return corners.tolist()

    @classmethod
    def motion2bbox(cls, bbox, motion):
        result = deepcopy(bbox)
        result.x += motion[0]
        result.y += motion[1]
        result.z += motion[2]
        result.o += motion[3]
        return result

    @classmethod
    def set_bbox_size(cls, bbox, size_array):
        result = deepcopy(bbox)
        result.l, result.w, result.h = size_array
        return result

    @classmethod
    def set_bbox_with_states(cls, prev_bbox, state_array):
        prev_array = BBox.bbox2array(prev_bbox)
        prev_array[:4] += state_array[:4]
        prev_array[4:] = state_array[4:]
        bbox = BBox.array2bbox(prev_array)
        return bbox

    @classmethod
    def box_pts2world(cls, ego_matrix, pcs):
        new_pcs = np.concatenate((pcs, np.ones(pcs.shape[0])[:, np.newaxis]), axis=1)
        new_pcs = ego_matrix @ new_pcs.T
        new_pcs = new_pcs.T[:, :3]
        return new_pcs

    @classmethod
    def edge2yaw(cls, center, edge):
        vec = edge - center
        yaw = np.arccos(vec[0] / np.linalg.norm(vec))
        if vec[1] < 0:
            yaw = -yaw
        return yaw

    @classmethod
    def bbox2world(cls, ego_matrix, box):
        # center and corners
        corners = np.array(BBox.box2corners2d(box))
        center = BBox.bbox2array(box)[:3][np.newaxis, :]
        center = BBox.box_pts2world(ego_matrix, center)[0]
        corners = BBox.box_pts2world(ego_matrix, corners)
        # heading
        edge_mid_point = (corners[0] + corners[1]) / 2
        yaw = BBox.edge2yaw(center[:2], edge_mid_point[:2])

        result = deepcopy(box)
        result.x, result.y, result.z = center
        result.o = yaw
        return result


def get_context_name(file_name: str):
    context = file_name.split(".")[0]  # file name#
    context = context.split("-")[1]  # after segment
    context = context.split("w")[0]  # before with
    context = context[:-1]
    return context


def main(
    name, obj_type, result_folder, raw_data_folder, output_folder, output_file_name
):
    summary_folder = os.path.join(result_folder, obj_type)
    file_names = sorted(os.listdir(summary_folder))[:]

    if obj_type == "vehicle":
        type_token = 1
    elif obj_type == "pedestrian":
        type_token = 2
    elif obj_type == "cyclist":
        type_token = 4

    ts_info_folder = os.path.join(raw_data_folder, "ts_info")
    ego_info_folder = os.path.join(raw_data_folder, "ego_info")
    obj_list = list()
    if args.mode == "val":
        file_names = file_names[:74]

    print("Converting TYPE {:} into WAYMO Format".format(obj_type))
    pbar = tqdm(total=len(file_names))
    for file_index, file_name in enumerate(file_names[:]):
        file_name_prefix = file_name.split(".")[0]
        context_name = get_context_name(file_name)

        ts_path = os.path.join(ts_info_folder, "{}.json".format(file_name_prefix))
        ts_data = json.load(open(ts_path, "r"))  # list of time stamps by order of frame

        # load ego motions
        ego_motions = np.load(
            os.path.join(ego_info_folder, "{:}.npz".format(file_name_prefix)),
            allow_pickle=True,
        )

        pred_result = np.load(
            os.path.join(summary_folder, file_name), allow_pickle=True
        )
        pred_ids, pred_bboxes, pred_states = (
            pred_result["ids"],
            pred_result["bboxes"],
            pred_result["states"],
        )
        pred_velos, pred_accels = None, None
        obj_list += create_sequence(
            pred_ids,
            pred_bboxes,
            type_token,
            context_name,
            ts_data,
            ego_motions,
            pred_velos,
            pred_accels,
        )
        pbar.update(1)
    pbar.close()
    objects = metrics_pb2.Objects()
    for obj in obj_list:
        objects.objects.append(obj)

    output_folder = os.path.join(output_folder, obj_type)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, "{:}.bin".format(output_file_name))
    f = open(output_path, "wb")
    f.write(objects.SerializeToString())
    f.close()
    return


def create_single_pred_bbox(
    id, bbox, type_token, time_stamp, context_name, inv_ego_motion, velo, accel
):
    o = metrics_pb2.Object()
    o.context_name = context_name
    o.frame_timestamp_micros = time_stamp
    box = label_pb2.Label.Box()

    proto_box = BBox.array2bbox(bbox)
    proto_box = BBox.bbox2world(inv_ego_motion, proto_box)
    bbox = BBox.bbox2array(proto_box)

    box.center_x, box.center_y, box.center_z, box.heading = bbox[:4]
    box.length, box.width, box.height = bbox[4:7]
    o.object.box.CopyFrom(box)
    o.score = bbox[-1]

    meta_data = label_pb2.Label.Metadata()
    if args.velo:
        meta_data.speed_x, meta_data.speed_y = velo[0], velo[1]
    if args.accel:
        meta_data.accel_x, meta_data.accel_y = accel[0], accel[1]
    o.object.metadata.CopyFrom(meta_data)

    o.object.id = "{:}_{:}".format(type_token, id)
    o.object.type = type_token
    return o


def create_sequence(
    pred_ids,
    pred_bboxes,
    type_token,
    context_name,
    time_stamps,
    ego_motions,
    pred_velos,
    pred_accels,
):
    frame_num = len(pred_ids)
    sequence_objects = list()
    for frame_index in range(frame_num):
        time_stamp = time_stamps[frame_index]
        frame_obj_num = len(pred_ids[frame_index])
        ego_motion = ego_motions["ego_infos"][frame_index][0][0]
        inv_ego_motion = np.linalg.inv(ego_motion)
        for obj_index in range(frame_obj_num):
            pred_id = pred_ids[frame_index][obj_index]
            pred_bbox = pred_bboxes[frame_index][obj_index]
            pred_velo, pred_accel = None, None
            if args.velo:
                pred_velo = pred_velos[frame_index][obj_index]
            if args.accel:
                pred_accel = pred_accels[frame_index][obj_index]
            sequence_objects.append(
                create_single_pred_bbox(
                    pred_id,
                    pred_bbox,
                    type_token,
                    time_stamp,
                    context_name,
                    inv_ego_motion,
                    pred_velo,
                    pred_accel,
                )
            )
    return sequence_objects


def merge_results(output_folder, obj_types, output_file_name):
    print("Merging different object types")
    result_objs = list()
    for obj_type in obj_types:
        bin_path = os.path.join(
            output_folder, obj_type, "{:}.bin".format(output_file_name)
        )
        f = open(bin_path, "rb")
        objects = metrics_pb2.Objects()
        objects.ParseFromString(f.read())
        f.close()
        objects = objects.objects
        result_objs += objects

    output_objs = metrics_pb2.Objects()
    for obj in result_objs:
        output_objs.objects.append(obj)

    output_path = os.path.join(output_folder, "{:}.bin".format(output_file_name))
    f = open(output_path, "wb")
    f.write(output_objs.SerializeToString())
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="MCTrack")
    parser.add_argument(
        "--obj_types", type=str, default="vehicle"
    )  # ,pedestrian')#,cyclist')
    parser.add_argument("--result_folder", type=str, default="./results/waymo/")
    parser.add_argument("--raw_data_folder", type=str, default="data/waymo/datasets")
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--src", type=str, default="summary")
    parser.add_argument("--velo", action="store_true", default=False)
    parser.add_argument("--accel", action="store_true", default=False)
    parser.add_argument("--output_file_name", type=str, default="pred")
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()

    if args.test:
        args.raw_data_folder = os.path.join(args.raw_data_folder, "test")
    else:
        args.raw_data_folder = os.path.join(args.raw_data_folder, "val")

    result_folder = args.result_folder
    output_folder = os.path.join(result_folder, "bin")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    obj_types = args.obj_types.split(",")
    for obj_type in obj_types:
        main(
            args.name,
            obj_type,
            result_folder,
            args.raw_data_folder,
            output_folder,
            args.output_file_name,
        )

    merge_results(output_folder, obj_types, args.output_file_name)
