'''
    modified from Immortaltracker
    https://github.com/esdolo/ImmortalTracker
    author: Qitai Wang
'''

""" Extract the ego location information from tfrecords
    Output file format: dict compressed in .npz files
    {
        st(frame_num): ego_info (4 * 4 matrix)
    }
"""
import argparse
import math
import numpy as np
import json
import os
import sys
sys.path.append('../')
from google.protobuf.descriptor import FieldDescriptor as FD
import tensorflow._api.v2.compat.v1 as tf
tf.enable_eager_execution()
import multiprocessing

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, help='location of tfrecords')
parser.add_argument('--output_folder', type=str, default='./data/waymo/',
    help='output folder')
parser.add_argument('--process', type=int, default=1, help='use multiprocessing for acceleration')
parser.add_argument('--test', action='store_true', default=False)
args = parser.parse_args()

class Matrixs:
    def __init__(self):
        self.ego2global = None
        self.CAM_FRONT_extrinsic = None
        self.CAM_FRONT_LEFT_extrinsic = None
        self.CAM_FRONT_RIGHT_extrinsic = None
        self.CAM_SIDE_LEFT_extrinsic = None
        self.CAM_SIDE_RIGHT_extrinsic = None
        self.CAM_FRONT_intrinsic = None
        self.CAM_FRONT_LEFT_intrinsic = None
        self.CAM_FRONT_RIGHT_intrinsic = None
        self.CAM_SIDE_LEFT_intrinsic = None
        self.CAM_SIDE_RIGHT_intrinsic = None
        self.LIDAR_FRONT_extrinsic = None
        self.LIDAR_REAR_extrinsic = None
        self.LIDAR_SIDE_LEFT_extrinsic = None
        self.LIDAR_SIDE_RIGHT_extrinsic = None
        self.LIDAR_TOP_extrinsic = None

    @classmethod
    def Matrix2array(cls, matrixs):
        return np.array([matrixs.ego2global, matrixs.CAM_FRONT_extrinsic, matrixs.CAM_FRONT_LEFT_extrinsic, 
                         matrixs.CAM_FRONT_RIGHT_extrinsic, matrixs.CAM_SIDE_LEFT_extrinsic, matrixs.CAM_SIDE_RIGHT_extrinsic, 
                         matrixs.CAM_FRONT_intrinsic, matrixs.CAM_FRONT_LEFT_intrinsic, matrixs.CAM_FRONT_RIGHT_intrinsic, 
                         matrixs.CAM_SIDE_LEFT_intrinsic, matrixs.CAM_SIDE_RIGHT_intrinsic, matrixs.LIDAR_FRONT_extrinsic, 
                         matrixs.LIDAR_REAR_extrinsic, matrixs.LIDAR_SIDE_LEFT_extrinsic, matrixs.LIDAR_SIDE_RIGHT_extrinsic, matrixs.LIDAR_TOP_extrinsic])

def pb2dict(obj):
    """
    Takes a ProtoBuf Message obj and convertes it to a dict.
    """
    adict = {}
    # if not obj.IsInitialized():
    #     return None
    for field in obj.DESCRIPTOR.fields:
        if not getattr(obj, field.name):
            continue
        if not field.label == FD.LABEL_REPEATED:
            if not field.type == FD.TYPE_MESSAGE:
                adict[field.name] = getattr(obj, field.name)
            else:
                value = pb2dict(getattr(obj, field.name))
                if value:
                    adict[field.name] = value
        else:
            if field.type == FD.TYPE_MESSAGE:
                adict[field.name] = [pb2dict(v) for v in getattr(obj, field.name)]
            else:
                adict[field.name] = [v for v in getattr(obj, field.name)]
    return adict


def main(token, process_num, data_folder, output_folder):
    tf_records = os.listdir(data_folder)
    tf_records = [x for x in tf_records if 'tfrecord' in x]
    tf_records = sorted(tf_records) 
    for record_index, tf_record_name in enumerate(tf_records):
        if record_index % process_num != token:
            continue
        print('starting for ego info ', record_index + 1, ' / ', len(tf_records), ' ', tf_record_name)
        FILE_NAME = os.path.join(data_folder, tf_record_name)
        dataset = tf.data.TFRecordDataset(FILE_NAME, compression_type='')
        segment_name = tf_record_name.split('.')[0]

        frame_num = 0
        ego_infos = []
        matrix_names = ["ego2global", "CAM_FRONT_extrinsic", "CAM_FRONT_LEFT_extrinsic", 
                         "CAM_FRONT_RIGHT_extrinsic", "CAM_SIDE_LEFT_extrinsic", "CAM_SIDE_RIGHT_extrinsic", 
                         "CAM_FRONT_intrinsic", "CAM_FRONT_LEFT_intrinsic", "CAM_FRONT_RIGHT_intrinsic", 
                         "CAM_SIDE_LEFT_intrinsic", "CAM_SIDE_RIGHT_intrinsic", "LIDAR_FRONT_extrinsic", 
                         "LIDAR_REAR_extrinsic", "LIDAR_SIDE_LEFT_extrinsic", "LIDAR_SIDE_RIGHT_extrinsic", "LIDAR_TOP_extrinsic"]

        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            matrixs = Matrixs()
     
            ego_info = np.reshape(np.array(frame.pose.transform), [4, 4])
            matrixs.ego2global = ego_info

            for cc in frame.context.camera_calibrations:
                # print(cc.name)
                extrinsic = tf.reshape(
                    tf.constant(list(cc.extrinsic.transform), dtype=tf.float32), # camera2ego
                    [4, 4])
                extrinsic = extrinsic.numpy()
                intrinsic = tf.constant(list(cc.intrinsic), dtype=tf.float32)
                intrinsic = intrinsic.numpy()
                NAME = open_dataset.CameraName.Name.Name(cc.name)
                # print(NAME)
                if NAME == "FRONT":
                    matrixs.CAM_FRONT_extrinsic = extrinsic
                    matrixs.CAM_FRONT_intrinsic = intrinsic
                elif NAME == "FRONT_LEFT":
                    matrixs.CAM_FRONT_LEFT_extrinsic = extrinsic
                    matrixs.CAM_FRONT_LEFT_intrinsic = intrinsic
                elif NAME == "FRONT_RIGHT":
                    matrixs.CAM_FRONT_RIGHT_extrinsic = extrinsic
                    matrixs.CAM_FRONT_RIGHT_intrinsic = intrinsic
                elif NAME == "SIDE_LEFT":
                    matrixs.CAM_SIDE_LEFT_extrinsic = extrinsic
                    matrixs.CAM_SIDE_LEFT_intrinsic = intrinsic
                elif NAME == "SIDE_RIGHT":
                    matrixs.CAM_SIDE_RIGHT_extrinsic = extrinsic
                    matrixs.CAM_SIDE_RIGHT_intrinsic = intrinsic

            for cc in frame.context.laser_calibrations:
                extrinsic = tf.reshape(
                    tf.constant(list(cc.extrinsic.transform), dtype=tf.float32), # camera2ego
                    [4, 4])
                extrinsic = extrinsic.numpy()
                # print(cc.name)
                NAME = open_dataset.LaserName.Name.Name(cc.name)
                # print(NAME)
                if NAME == "FRONT":
                    matrixs.LIDAR_FRONT_extrinsic = extrinsic
                if NAME == "REAR":  
                    matrixs.LIDAR_REAR_extrinsic = extrinsic
                if NAME == "SIDE_LEFT":  
                    matrixs.LIDAR_SIDE_LEFT_extrinsic = extrinsic
                if NAME == "SIDE_RIGHT":  
                    matrixs.LIDAR_SIDE_RIGHT_extrinsic = extrinsic
                if NAME == "TOP":      
                    matrixs.LIDAR_TOP_extrinsic = extrinsic
            matrixs = [Matrixs.Matrix2array(matrixs)]
            # print(matrixs)
            ego_infos.append(matrixs)
                
            frame_num += 1
            if frame_num % 10 == 0:
                print('record {:} / {:} frame number {:}'.format(record_index + 1, len(tf_records), frame_num))
        print('{:} frames in total'.format(frame_num))
        
        np.savez_compressed(os.path.join(output_folder, "{}.npz".format(segment_name)), ego_infos=ego_infos, matrix_names=matrix_names)


if __name__ == '__main__':
    if args.test:
        args.data_folder=os.path.join(args.data_folder, 'testing')
        args.output_folder=os.path.join(args.output_folder, 'testing')
    else:
        args.data_folder=os.path.join(args.data_folder, 'validation')
        args.output_folder=os.path.join(args.output_folder, 'validation')
    args.output_folder = os.path.join(args.output_folder, 'ego_info')
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(token, args.process, args.data_folder, args.output_folder))
        pool.close()
        pool.join()
    else:
        main(0, 1, args.data_folder, args.output_folder)
