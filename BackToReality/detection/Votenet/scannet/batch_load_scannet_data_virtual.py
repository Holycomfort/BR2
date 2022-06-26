# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations

Usage example: python ./batch_load_scannet_data.py
"""
import os
import sys
import datetime
import numpy as np
sys.path.append("../../../data_generation/ScanNet")
from scannet_scene_synthesis import export_for_md40


TRAIN_SCAN_NAMES = [sname[:-4] for sname in os.listdir('./augment_random_positions_scannet')]
LABEL_MAP_FILE = 'meta_data/scannetv2-labels.combined.tsv'
DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array([2,3,4,5,6,9,11,12,13,14,15,19,20,21,23,24,27,31,33,34,36,39])
MAX_NUM_POINT = 50000
OUTPUT_FOLDER = './scannet_train_detection_data_md40_obj_aug'

def export_one_scan(scan_name, output_filename_prefix):    
    mesh_vertices, semantic_labels, instance_labels, instance_bboxes = \
        export_for_md40(scan_name, Is_density=True, Is_HPR=False, Is_floor=True)

    mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
    mesh_vertices = mesh_vertices[mask,:]
    semantic_labels = semantic_labels[mask]
    instance_labels = instance_labels[mask]

    num_instances = len(np.unique(instance_labels))
    print('Num of instances: ', num_instances)

    bbox_mask = np.in1d(instance_bboxes[:,-1], OBJ_CLASS_IDS)
    instance_bboxes = instance_bboxes[bbox_mask,:]
    print('Num of care instances: ', instance_bboxes.shape[0])

    N = mesh_vertices.shape[0]
    if N > MAX_NUM_POINT:
        choices = np.random.choice(N, MAX_NUM_POINT, replace=False)
        mesh_vertices = mesh_vertices[choices, :]
        semantic_labels = semantic_labels[choices]
        instance_labels = instance_labels[choices]

    np.save(output_filename_prefix+'_vert.npy', mesh_vertices)
    np.save(output_filename_prefix+'_sem_label.npy', semantic_labels)
    np.save(output_filename_prefix+'_ins_label.npy', instance_labels)
    np.save(output_filename_prefix+'_bbox.npy', instance_bboxes)

def batch_export():
    if not os.path.exists(OUTPUT_FOLDER):
        print('Creating new data folder: {}'.format(OUTPUT_FOLDER))                
        os.mkdir(OUTPUT_FOLDER)        
        
    for scan_name in TRAIN_SCAN_NAMES:
        print('-'*20+'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name) 
        if os.path.isfile(output_filename_prefix+'_vert.npy'):
            print('File already exists. skipping.')
            print('-'*20+'done')
            continue
        export_one_scan(scan_name, output_filename_prefix)
        #try:            
        #    export_one_scan(scan_name, output_filename_prefix)
        #except:
        #    print('Failed export scan: %s'%(scan_name))            
        print('-'*20+'done')


if __name__=='__main__':    
    batch_export()