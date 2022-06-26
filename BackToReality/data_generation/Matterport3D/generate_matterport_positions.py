import open3d as o3d
import numpy as np
import cv2
from segment_tools import export_random
import os
import copy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from ins_ply import read_aggregation, read_segmentation, create_color_palette, get_id_to_label
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

import warnings
warnings.filterwarnings("ignore")



def get_MER(points):
    points = copy.deepcopy(points)
    xys = points[:, 0:2] * 1000
    xys = xys.astype('int')
    (x_center, y_center), (x_size, y_size), angle = cv2.minAreaRect(xys)
    x_center /= 1e3; y_center /= 1e3; y_size /= 1e3; x_size /= 1e3
    angle = angle / 180 * np.pi
    return (x_center, y_center), (x_size, y_size), angle


# MER: ((x, y), (long, short), theta)
def get_solid_MER(points):
    points = copy.deepcopy(points)
    xys = points[:, 0:2]
    xys *= 1000
    xys = xys.astype('int')
    rect = cv2.minAreaRect(xys)
    k_means = KMeans(n_clusters=2)
    k_means.fit(xys)
    cluster_label = k_means.predict(xys)
    choose0 = (sum(cluster_label == 0) < sum(cluster_label == 1))
    if choose0:
        xys_part = xys[cluster_label == 0]
        xys_other = xys[cluster_label == 1]
    else:
        xys_part = xys[cluster_label == 1]
        xys_other = xys[cluster_label == 0]
    rect_part = cv2.minAreaRect(xys_part)
    Is_solid = (rect_part[1][0] * rect_part[1][1] * 2.5 > rect[1][0] * rect[1][1])
    if Is_solid:
        pass
    else:
        rect = cv2.minAreaRect(xys_other)
    
    if rect[1][1] > rect[1][0]:
        l_s = (rect[1][1]/1000, rect[1][0]/1000)
    else:
        l_s = (rect[1][0]/1000, rect[1][1]/1000)
    if rect[1][0] >= rect[1][1]:
        theta = -rect[2]
        if theta == 0:
            theta = 180
    else:
        theta = -rect[2] + 90
    return ((rect[0][0] / 1000, rect[0][1] / 1000), l_s, theta)


def find_nearest_object(ls_ratio, info_dict, object_name, require_support=False):
    min_dis = 100
    min_code = ""
    for key, value in info_dict.items():
        if value[0][0][1] == 0:
            continue
        if abs(value[0][0][0] / value[0][0][1] - ls_ratio) < min_dis:
            if require_support == True and value[2] == False:
                continue
            min_dis = abs(value[0][0][0] / value[0][0][1] - ls_ratio)
            min_code = key
    if min_code == "" and require_support == True:
        return find_nearest_object(ls_ratio, info_dict, object_name)
    txt = object_name + "_" + min_code + ".txt"
    return txt, info_dict[min_code]
        

def generate_initial_random_positions(mesh_file, agg_file, seg_file, modelnet40_path, scan_name):
    xyz_obj_dict, scene_vertices = export_random(mesh_file, agg_file, seg_file, scan_name)
    obj_prop = np.load('CONFIG/object40_property.npy', allow_pickle=True, encoding='bytes').item()
    modelnet40_names = np.loadtxt('CONFIG/modelnet40_shape_names.txt', dtype='object')
    supporter_list = ["tv_stand", "desk", "bed", "bookshelf", "table", "night_stand"]
    # object_id: [(x,y,z), (sx,sy,sz), object_txt, Is_supporter, theta, support_MER/None, height/None]
    # theta is the orientation of the object (anticlockwise)
    # supporter has theta, support_MER and height
    # object with plane but not supporter has theta, None, None
    # object without plane has None, None, None
    # MER: ((x, y), (long, short), theta)
    positions = {}
    floors = []
    for key, value in xyz_obj_dict.items():
        obj_name = modelnet40_names[value[2] - 1]
        txts = os.listdir(os.path.join(modelnet40_path, obj_name))
        txts.remove("this_class_info.npy")
        this_class_info = np.load(os.path.join(modelnet40_path, obj_name,
         "this_class_info.npy"), allow_pickle=True).item()

        # choose xy or yx
        if np.random.rand() > 0.5:
            _, _, _, dxavg, dyavg, dzavg = obj_prop[value[2]][0:6]
        else:
            dxavg, dyavg, dzavg, _, _, _ = obj_prop[value[2]][0:6]
        dxmin, dymin, dzmin = 0.8 * dxavg, 0.8 * dyavg, 0.8 * dzavg
        dxmax, dymax, dzmax = 1.3 * dxavg, 1.3 * dyavg, 1.3 * dzavg
        dx = dxmin + np.random.rand() * (dxmax - dxmin)
        dy = dymin + np.random.rand() * (dymax - dymin)
        dz = dzmin + np.random.rand() * (dzmax - dzmin)
        x, y, z = value[0]
        ls_ratio = max(dx, dy) / min(dx, dy)
        if obj_name in supporter_list:
            txt, obj_info = find_nearest_object(ls_ratio, this_class_info, obj_name, require_support=True)
        else:
            txt, obj_info = find_nearest_object(ls_ratio, this_class_info, obj_name)
        obj_txt = os.path.join(modelnet40_path, obj_name, txt)
        pc_txt = np.loadtxt(obj_txt, delimiter=',')
        pc_txt[:, [1, 2]] = pc_txt[:, [2, 1]]
        pc_txt[:, [4, 5]] = pc_txt[:, [4, 5]]
        ddx, ddy, ddz = max(pc_txt[:, 0]) - min(pc_txt[:, 0]), max(pc_txt[:, 1]) - min(pc_txt[:, 1]), max(pc_txt[:, 2]) - min(pc_txt[:, 2])
        scale = (dx * dx * dz / ddx / ddy / ddz)**(1 / 3)
        # special category: door, curtain, ...
        # only constrain the height
        if obj_name in ["curtain", "door", "sofa", "desk"]:
            scale = dz / ddz
        # special category: keyboard
        # only constrain the horizontal property
        if obj_name in ["keyboard"]:
            scale = (dx * dy / ddx / ddy)**(1 / 2)
        theta = np.random.rand() * 360
        if obj_name in supporter_list:
            MER = ((x, y), (scale * max(ddx, ddy), scale * min(ddx, ddy)), (theta + obj_info[0][1]) % 180)
            height = z + scale * obj_info[1]
            positions[key] = [(x, y, z), (scale, scale, scale), obj_txt, True, theta, MER, height]
        else:
            positions[key] = [(x, y, z), (scale, scale, scale), obj_txt, False, theta, None, None]
        if not value[3]:
            temp_position = positions.pop(key)
            points_included_in_floor = position_to_xyz(temp_position, Is_density=True, S_larger=scale**2*ddx*ddy*ddz/min(ddx,ddy,ddz))
            floors.append(points_included_in_floor)

    '''
    # no floor/wall points
    wall_points = np.array([])
    floor_points = np.array([])
    '''
    
    # get floor/wall points
    map2nyu40 = np.load('CONFIG/map2nyu40.npy', allow_pickle=True).item()
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)
    for label, segs in label_to_segs.items():
        try:
            label_id = map2nyu40[label]
        except:
            label_id = 0
        if label_id not in ['1', '2']: continue
        for seg in segs:
            try:
                verts = seg_to_verts[seg]
            except:
                continue
            label_ids[verts] = label_id
    wall_points = scene_vertices[label_ids == 1]
    floors.append(scene_vertices[label_ids == 2])
    floor_points = np.concatenate(floors, axis=0)
    #floor_points = scene_vertices[abs(scene_vertices[:, 2]) < 0.05]
    
    return positions, (floor_points, wall_points)


def point_in_MER(x, y, MER):
    dx = x - MER[0][0]
    dy = abs(MER[0][1] - y)
    dd = (dx ** 2 + dy ** 2)** 0.5
    cosf = dx / dd
    f = np.arccos(cosf) / np.pi * 180
    if MER[2] >= 90:
        theta = f - MER[2] + 90
        dx_align = abs(dd * np.cos(theta / 180 * np.pi))
        dy_align = abs(dd * np.sin(theta / 180 * np.pi))
        if dx_align < MER[1][1] / 2 and dy_align < MER[1][0] / 2:
            return True
    else:
        theta = f - MER[2]
        dx_align = abs(dd * np.cos(theta / 180 * np.pi))
        dy_align = abs(dd * np.sin(theta / 180 * np.pi))
        if dx_align < MER[1][0] / 2 and dy_align < MER[1][1] / 2:
            return True
    return False


def generate_gravity_aware_positions(positions, floor_points):
    new_positions = positions.copy()
    if len(floor_points) == 0:
        ground_z = 0
    else:
        ground_z = np.mean(floor_points[:, 2])
    supporter_MER = {}  # id: MER
    # Stage1: not be supported, things on the ground or dangling (lamp/sink) 
    # Stage2: supported objects
    # stage1_id: [stage2_id, ...]
    stage_map = {}
    for key, value in positions.items():
        obj_name = value[2].split('/')[-2]
        if value[3] == True:
            supporter_MER[key] = value[5]
            stage_map[key] = []
            x, y, z = value[0]
            _, _, sz = value[1]
            pc_txt = np.loadtxt(value[2], delimiter=',')
            pc_txt[:, [1, 2]] = pc_txt[:, [2, 1]]
            pc_txt[:, [4, 5]] = pc_txt[:, [4, 5]]
            new_z = ground_z - sz * min(pc_txt[:, 2])
            new_positions[key][0] = (x, y, new_z)
            new_positions[key][6] = value[6] + (new_z - z)
    for key, value in positions.items():
        if value[3] == False:
            min_center_dis2 = 100
            x, y = value[0][0], value[0][1]
            choosed_supporter = -1
            for supporter_id, MER in supporter_MER.items():
                if point_in_MER(x, y, MER) and (x - MER[0][0])** 2 + (y - MER[0][1])** 2 < min_center_dis2:
                    choosed_supporter = supporter_id
                    min_center_dis2 = (x - MER[0][0])** 2 + (y - MER[0][1])** 2
            # some object will never be supported
            # so we need to correct the wrong choice if needed
            obj_name = value[2].split('/')[-2]
            if obj_name not in ["monitor", "plant", "lamp", "sink", "cup", "keyboard", "bottle", "laptop"]:
                choosed_supporter = -1
            
            if choosed_supporter == -1:
                stage_map[key] = []
            else:
                stage_map[choosed_supporter].append(key)

            x, y, z = value[0]
            _, _, sz = value[1]
            pc_txt = np.loadtxt(value[2], delimiter=',')
            pc_txt[:, [1, 2]] = pc_txt[:, [2, 1]]
            pc_txt[:, [4, 5]] = pc_txt[:, [4, 5]]
            # special category: sink, lamp
            if choosed_supporter == -1:
                if (obj_name == "lamp" and z > 1.2) or obj_name == "sink":
                    new_z = z
                else:
                    new_z = ground_z - sz * min(pc_txt[:, 2])
            else:
                new_z = new_positions[choosed_supporter][6] - sz * min(pc_txt[:, 2])
            new_positions[key][0] = (x, y, new_z)
    return new_positions, stage_map


def anticlock_rotate_matrix(theta):
    # anticlockwise means y-->x
    theta *= -1
    return np.array([[np.cos(np.pi / 180 * theta), np.sin(np.pi / 180 * theta)],
     [-np.sin(np.pi / 180 * theta), np.cos(np.pi / 180 * theta)]])


def position_to_xyz(position, Is_density=False, S_larger=0.1):
    # If consider density, the total number of points of a object is 10000*ratio
    obj_xyz = np.loadtxt(position[2], delimiter=',')[:, 0:3]
    obj_xyz[:, [1, 2]] = obj_xyz[:, [2, 1]]
    if Is_density:
        ds_k = int(100 / max(1, 10 * S_larger) + 1.5)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_xyz)
        down_pcd = pcd.uniform_down_sample(every_k_points=ds_k)
        obj_xyz = np.array(down_pcd.points)
        #print(position[2].split('/')[-2] + ": ", obj_xyz.shape[0], ", ", ds_k)
    # scale
    obj_xyz[:, 0] *= position[1][0]
    obj_xyz[:, 1] *= position[1][1]
    obj_xyz[:, 2] *= position[1][2]
    # rotate
    theta = position[4]
    obj_xyz[:, 0:2] = np.matmul(obj_xyz[:, 0:2], anticlock_rotate_matrix(theta))
    # translate
    obj_xyz[:, 0] += position[0][0]
    obj_xyz[:, 1] += position[0][1]
    obj_xyz[:, 2] += position[0][2]
    return obj_xyz
    


if __name__ == "__main__":
    MATTERPORT_DIR = "/opt/data2/MATTERPORT3D/for_scannet/scans"
    scan_names = os.listdir(MATTERPORT_DIR)
    for scan_name in scan_names:
        #now_dir = os.listdir('./matterport_positions_Prandom')
        #if scan_name+'_constant.npy' in now_dir:
        #   continue
        print(scan_name)
        scan_folder = "/opt/data2/MATTERPORT3D/for_scannet/scans/" + scan_name + "/"
        mesh_file = os.path.join(scan_folder, 'region{}.ply'.format(int(scan_name[-2:])))
        agg_file = os.path.join(scan_folder, 'region{}.semseg.json'.format(int(scan_name[-2:])))
        seg_file = os.path.join(scan_folder, 'region{}.vsegs.json'.format(int(scan_name[-2:])))
        modelnet40_path = "./modelnet40_normal_resampled"
        positions, fw_points = generate_initial_random_positions(mesh_file, agg_file, seg_file, modelnet40_path, scan_name)
        positions, stage_map = generate_gravity_aware_positions(positions, fw_points[0])
        floor_points = fw_points[0]
        wall_points = fw_points[1]
        constant_property = {}
        initial_variable = {}

        # object_id: ((x, y), OBJ_txt, minZ, Is_supporter, height)
        constant_property['floor'] = floor_points
        constant_property['wall'] = wall_points
        constant_property['stage_map'] = stage_map
        for key, value in positions.items():
            obj_xyz = np.loadtxt(value[2], delimiter=',')[:, 0:3]
            obj_xyz[:, [1, 2]] = obj_xyz[:, [2, 1]]
            minZ = min(obj_xyz[:, 2])
            Is_supporter = value[3]
            height = (value[6] - value[0][2]) / value[1][2] if Is_supporter else None
            constant_property[key] = ((value[0][0], value[0][1]), value[2], minZ, Is_supporter, height)
            initial_variable[key] = ((0, 0), value[1], value[4])
        np.save('./matterport_positions_Prandom/'+scan_name+'_constant.npy', constant_property)
        np.save('./matterport_positions_Prandom/'+scan_name+'_variable.npy', initial_variable)