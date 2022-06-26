from math import floor
import open3d as o3d
import numpy as np
import cv2
from segment_tools import export, export_random
import os
from sklearn.cluster import KMeans
from ins_ply import read_aggregation, read_segmentation, create_color_palette, get_id_to_label
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

import warnings
warnings.filterwarnings("ignore")


# MER: ((x, y), (long, short), theta)
def get_solid_MER(points):
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


def get_horizontal_area(points):
    xys = points[:,:2]
    xys *= 1000
    xys = xys.astype('int')
    hull = cv2.convexHull(xys, clockwise=True, returnPoints=True)
    area = cv2.contourArea(hull)
    return area/1000000


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


def generate_initial_positions(mesh_file, agg_file, seg_file, meta_file, modelnet40_path):
    hrz_obj_dict, segidx_to_seg, scene_vertices = export(mesh_file, agg_file, seg_file, meta_file)
    obj_prop = np.load('CONFIG/object40_property.npy', allow_pickle=True, encoding='bytes').item()
    modelnet40_names = np.loadtxt('CONFIG/modelnet40_shape_names.txt', dtype='object')
    supporter_list = ["tv_stand", "desk", "bed", "bookshelf", "table", "night_stand"]
    # object_id: [(x,y,z), (sx,sy,sz), object_txt, Is_supporter, theta/None, support_MER/None, height/None]
    # theta is the orientation of the object (anticlockwise)
    # supporter has theta, support_MER and height
    # object with plane but not supporter has theta, None, None
    # object without plane has None, None, None
    # MER: ((x, y), (long, short), theta)
    positions = {}
    for key, value in hrz_obj_dict.items():
        obj_name = modelnet40_names[value[4] - 1]
        txts = os.listdir(os.path.join(modelnet40_path, obj_name))
        txts.remove("this_class_info.npy")
        this_class_info = np.load(os.path.join(modelnet40_path, obj_name,
         "this_class_info.npy"), allow_pickle=True).item()
        # no plane
        if value[0] == False:
            # choose xy or yx
            if np.random.rand() > 0.5:
                _, _, _, dxavg, dyavg, dzavg = obj_prop[value[4]][0:6]
            else:
                dxavg, dyavg, dzavg, _, _, _ = obj_prop[value[4]][0:6]
            dxmin, dymin, dzmin = 0.8 * dxavg, 0.8 * dyavg, 0.8 * dzavg
            dxmax, dymax, dzmax = 1.3 * dxavg, 1.3 * dyavg, 1.3 * dzavg
            dx = dxmin + np.random.rand() * (dxmax - dxmin)
            dy = dymin + np.random.rand() * (dymax - dymin)
            dz = dzmin + np.random.rand() * (dzmax - dzmin)
            x, y, z = value[1]
            ls_ratio = max(dx, dy) / min(dx, dy)
            txt, obj_info = find_nearest_object(ls_ratio, this_class_info, obj_name)
            obj_txt = os.path.join(modelnet40_path, obj_name, txt)
            pc_txt = np.loadtxt(obj_txt, delimiter=',')
            pc_txt[:, [1, 2]] = pc_txt[:, [2, 1]]
            pc_txt[:, [4, 5]] = pc_txt[:, [4, 5]]
            ddx, ddy, ddz = max(pc_txt[:, 0]) - min(pc_txt[:, 0]), max(pc_txt[:, 1]) - min(pc_txt[:, 1]), max(pc_txt[:, 2]) - min(pc_txt[:, 2])
            scale = (dx * dx * dz / ddx / ddy / ddz)**(1 / 3)
            # special category: door, curtain, ...
            # only constrain the height
            if obj_name in ["curtain", "door"]:
                scale = dz / ddz
            # special category: keyboard
            # only constrain the horizontal property
            if obj_name in ["keyboard"]:
                scale = (dx * dy/ ddx / ddy)**(1 / 2)
            theta = np.random.rand() * 180
            positions[key] = [(x, y, z), (scale, scale, scale), obj_txt, False, theta, None, None]
        # has plane
        else:
            verts = []
            for segidx in value[1]:
                verts += list(scene_vertices[segidx_to_seg[segidx],:])
            verts = np.array(verts)
            MER = get_solid_MER(verts)
            x, y = MER[0]
            ls_ratio = MER[1][0] / MER[1][1]
            txt, obj_info = find_nearest_object(ls_ratio, this_class_info, obj_name, require_support=True)
            obj_txt = os.path.join(modelnet40_path, obj_name, txt)
            pc_txt = np.loadtxt(obj_txt, delimiter=',')
            pc_txt[:, [1, 2]] = pc_txt[:, [2, 1]]
            pc_txt[:, [4, 5]] = pc_txt[:, [4, 5]]
            theta = MER[2] - obj_info[0][1]
            if obj_name in supporter_list:
                if abs(obj_info[0][1]) < 10 or abs(obj_info[0][1] - 180) < 10:
                    dx, dy = 1.2 * MER[1][0], 1.2 * MER[1][1]
                else:
                    dy, dx = 1.2 * MER[1][0], 1.2 * MER[1][1]
                ddx, ddy = max(pc_txt[:, 0]) - min(pc_txt[:, 0]), max(pc_txt[:, 1]) - min(pc_txt[:, 1])
                sx, sy = dx / ddx, dy / ddy
                sz = (sx + sy) / 2
                z = value[2] - sz * obj_info[1]
                support_MER = ((MER[0][0], MER[0][1]), (1.2 * MER[1][0], 1.2 * MER[1][1]), MER[2])
                positions[key] = [(x, y, z), (sx, sy, sz), obj_txt, True, theta, support_MER, value[2]]
            # has plane but is not a supporter
            else:
                z = np.mean(verts[:, 2])
                # choose xy or yx
                if np.random.rand() > 0.5:
                    _, _, _, dxavg, dyavg, dzavg = obj_prop[value[4]][0:6]
                else:
                    dxavg, dyavg, dzavg, _, _, _ = obj_prop[value[4]][0:6]
                dxmin, dymin, dzmin = 0.8 * dxavg, 0.8 * dyavg, 0.8 * dzavg
                dxmax, dymax, dzmax = 1.3 * dxavg, 1.3 * dyavg, 1.3 * dzavg
                dx = dxmin + np.random.rand() * (dxmax - dxmin)
                dy = dymin + np.random.rand() * (dymax - dymin)
                dz = dzmin + np.random.rand() * (dzmax - dzmin)
                ddx, ddy, ddz = max(pc_txt[:, 0]) - min(pc_txt[:, 0]), max(pc_txt[:, 1]) - min(pc_txt[:, 1]), max(pc_txt[:, 2]) - min(pc_txt[:, 2])
                scale = (dx * dx * dz / ddx / ddy / ddz)**(1 / 3)
                positions[key] = [(x, y, z), (scale, scale, scale), obj_txt, False, theta, None, None]

    # get floor/wall points
    map2nyu40 = np.load('CONFIG/map2nyu40.npy', allow_pickle=True).item()
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)
    for label, segs in label_to_segs.items():
        label_id = map2nyu40[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    wall_points = scene_vertices[label_ids == 1]
    floor_points = scene_vertices[label_ids == 2]
    #floor_points = scene_vertices[abs(scene_vertices[:, 2]) < 0.05]
    
    return positions, (floor_points, wall_points)


def generate_initial_random_positions(mesh_file, agg_file, seg_file, meta_file, modelnet40_path):
    xyz_obj_dict, scene_vertices = export_random(mesh_file, agg_file, seg_file, meta_file)
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
        #theta = np.random.rand() * 180
        theta = np.random.choice([0, 90]) + (10 * np.random.rand() - 5)
        if obj_name in supporter_list:
            MER = ((x, y), (scale * max(ddx, ddy), scale * min(ddx, ddy)), (theta + obj_info[0][1]) % 180)
            height = z + scale * obj_info[1]
            positions[key] = [(x, y, z), (scale, scale, scale), obj_txt, True, theta, MER, height]
        else:
            positions[key] = [(x, y, z), (scale, scale, scale), obj_txt, False, theta, None, None]

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
        label_id = map2nyu40[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    wall_points = scene_vertices[label_ids == 1]
    floor_points = scene_vertices[label_ids == 2]
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


def position_to_xyz(position, Is_density=False, ratio=None):
    # If consider density, the total number of points of a object is 10000*ratio
    obj_xyz = np.loadtxt(position[2], delimiter=',')[:, 0:3]
    obj_xyz[:, [1, 2]] = obj_xyz[:, [2, 1]]
    if Is_density:
        ds_k = int(1 // ratio)
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
    

def positions_to_pcd(positions, fw_points, Is_floor=True, Is_wall=False, Is_density=False, Is_HPR=False):
    if len(positions) == 0:
        return None
    xyzrgb_list = []
    colors = create_color_palette()
    # floor
    if Is_floor:
        floor_points = fw_points[0]
        ground_z = np.mean(floor_points[:, 2])
        floor_points[:, 2] = ground_z
        floor_color = colors[0]
        floor_rgb = np.array([list(floor_color)] * floor_points.shape[0])
        floor_xyzrgb = np.concatenate([floor_points, floor_rgb], axis=1)
        xyzrgb_list += list(floor_xyzrgb)
    # wall
    if Is_wall:
        wall_points = fw_points[1]
        wall_color = colors[0]
        wall_rgb = np.array([list(wall_color)] * wall_points.shape[0])
        wall_xyzrgb = np.concatenate([wall_points, wall_rgb], axis=1)
        xyzrgb_list += list(wall_xyzrgb)
    # density
    if Is_density:
        S_dict = {}
        for key, value in positions.items():
            obj_xyz = np.loadtxt(value[2], delimiter=',')[:, 0:3]
            obj_xyz[:, [1, 2]] = obj_xyz[:, [2, 1]]
            ddx, ddy, ddz = max(obj_xyz[:, 0]) - min(obj_xyz[:, 0]), max(obj_xyz[:, 1]) - min(obj_xyz[:, 1]), max(obj_xyz[:, 2]) - min(obj_xyz[:, 2])
            dx, dy, dz = ddx * value[1][0], ddy * value[1][1], ddz * value[1][2]
            S_larger = dx * dy * dz / min(dx, dy, dz)
            S_dict[key] = S_larger
        S_max = max(S_dict.values())
    # position to points
    for key, value in positions.items():
        color = colors[key]
        if Is_density:
            obj_xyz = position_to_xyz(value, Is_density=True, ratio=S_dict[key]/S_max)
        else:
            obj_xyz = position_to_xyz(value)
        obj_rgb = np.array([list(color)] * obj_xyz.shape[0])
        obj_xyzrgb = np.concatenate([obj_xyz, obj_rgb], axis=1)
        xyzrgb_list += list(obj_xyzrgb)
    # points to pcd
    xyzrgb = np.array(xyzrgb_list)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:6] / 255)
    # HPR
    if Is_HPR:
        xs, ys = xyzrgb[:, 0], xyzrgb[:, 1]
        print("x in (", xs.min(), ",", xs.max(), "), y in (", ys.min(), ",", ys.max(), ")")
        _, pt_map = pcd.hidden_point_removal([0, 0, 2.5], 100)
        pcd = pcd.select_by_index(pt_map)
    return pcd


MDN_dict = {'airplane': 1, 'bathtub': 2, 'bed': 3, 'bench': 4, 'bookshelf': 5, 'bottle': 6, 'bowl': 7, 'car': 8, 'chair': 9, 'cone': 10, 'cup': 11, 'curtain': 12, 'desk': 13, 'door': 14, 'dresser': 15, 'flower_pot': 16, 'glass_box': 17, 'guitar': 18, 'keyboard': 19, 'lamp': 20, 'laptop': 21, 'mantel': 22, 'monitor': 23, 'night_stand': 24, 'person': 25, 'piano': 26, 'plant': 27, 'radio': 28, 'range_hood': 29, 'sink': 30, 'sofa': 31, 'stairs': 32, 'stool': 33, 'table': 34, 'tent': 35, 'toilet': 36, 'tv_stand': 37, 'vase': 38, 'wardrobe': 39, 'xbox': 40}

def export_for_md40(scan_name, Is_density=True, Is_HPR=False, Is_floor=False):
    positions = np.load('./augment_random_positions_scannet/'+scan_name+'.npy', allow_pickle=True).item()
    xyz_oid_list = []
    if Is_floor:
        floor_points = positions['floor_points']
        #floor_points[:, 2] = np.mean(floor_points[:, 2])
    positions.pop('floor_points')
    # density
    if Is_density:
        S_dict = {}
        for key, value in positions.items():
            obj_xyz = np.loadtxt(value[2], delimiter=',')[:, 0:3]
            obj_xyz[:, [1, 2]] = obj_xyz[:, [2, 1]]
            ddx, ddy, ddz = max(obj_xyz[:, 0]) - min(obj_xyz[:, 0]), max(obj_xyz[:, 1]) - min(obj_xyz[:, 1]), max(obj_xyz[:, 2]) - min(obj_xyz[:, 2])
            dx, dy, dz = ddx * value[1][0], ddy * value[1][1], ddz * value[1][2]
            S_larger = dx * dy * dz / min(dx, dy, dz)
            S_dict[key] = S_larger
        S_max = max(S_dict.values())
    # position to points
    instance_bboxes = np.zeros((len(positions),7))
    #label_to_nyuid = np.load('map2nyu40.npy', allow_pickle=True).item()
    label_to_modelnet40id = np.load("CONFIG/map2modelnet.npy", allow_pickle=True).item()
    obj_prop = np.load('CONFIG/object40_property.npy', allow_pickle=True, encoding='bytes').item()
    for key, value in positions.items():
        oid = key
        if Is_density:
            obj_xyz = position_to_xyz(value, Is_density=True, ratio=S_dict[key]/S_max)
        else:
            obj_xyz = position_to_xyz(value)
        obj_id = np.array([[oid]] * obj_xyz.shape[0])
        obj_xyzoid = np.concatenate([obj_xyz, obj_id], axis=1)
        xyz_oid_list += list(obj_xyzoid)
    if Is_floor:
        floor_xyzoid = np.concatenate([floor_points, np.array([[0]] * floor_points.shape[0])], axis=1)
        xyz_oid_list += list(floor_xyzoid)
    xyz_oid = np.array(xyz_oid_list)
    if Is_HPR:
        xyz = xyz_oid[:,:3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        xs, ys = xyz[:, 0], xyz[:, 1]
        delta_x, delta_y = (xs.max() - xs.min()) / 3, (ys.max() - ys.min()) / 3
        camera1 = [xs.min() + delta_x, ys.min() + delta_y, 2]
        camera2 = [xs.min() + 2 * delta_x, ys.min() + delta_y, 2]
        camera3 = [xs.min() + delta_x, ys.min() + 2 * delta_y, 2]
        camera4 = [xs.min() + 2 * delta_x, ys.min() + 2 * delta_y, 2]
        _, pt_map1 = pcd.hidden_point_removal(camera1, 100)
        _, pt_map2 = pcd.hidden_point_removal(camera2, 100)
        _, pt_map3 = pcd.hidden_point_removal(camera3, 100)
        _, pt_map4 = pcd.hidden_point_removal(camera4, 100)
        pt_map = np.unique(pt_map1 + pt_map2 + pt_map3 + pt_map4)
        xyz_oid = xyz_oid[pt_map]
    count_i = 0
    #np.save('xxw.npy', xyz_oid[:,:3])
    #sys.exit(0)
    oid_to_modelnet40id = {}
    for oid, value in positions.items():
        obj_xyz = xyz_oid[xyz_oid[:,3] == oid][:,:3]
        if len(obj_xyz) == 0:
            continue
        xmin = np.min(obj_xyz[:,0])
        ymin = np.min(obj_xyz[:,1])
        zmin = np.min(obj_xyz[:,2])
        xmax = np.max(obj_xyz[:,0])
        ymax = np.max(obj_xyz[:,1])
        zmax = np.max(obj_xyz[:,2])
        obj_name = value[2].split('/')[-2]
        sem_label = MDN_dict[obj_name]
        oid_to_modelnet40id[oid] = sem_label
        #if sem_label in [4,7,6,5,33,14,3,32,10,36]:
        #    dxavg_xy, dyavg_xy, dzavg_xy, dxavg_yx, dyavg_yx, dzavg_yx = obj_prop[label_to_modelnet40id[id_to_label[oid]]][0:6]
        #    if (xmax-xmin) < min(dxavg_xy, dxavg_yx)/2 and (ymax-ymin) < min(dyavg_xy, dyavg_yx)/2 and (zmax-zmin) < min(dzavg_xy, dzavg_yx)/2:
        #        sem_label = -1
        #        xyz_oid[xyz_oid[:,3] == oid][:,3] = -1
        bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin, sem_label])
        instance_bboxes[count_i,:] = bbox
        count_i += 1
    mesh_vertices = (xyz_oid.copy())[:,:3]
    instance_ids = xyz_oid[:, 3]
    semantic_ids = np.zeros_like(instance_ids)
    for i in range(len(semantic_ids)):
        if instance_ids[i] == 0:
            semantic_ids[i] = 0
        else:
            semantic_ids[i] = oid_to_modelnet40id[instance_ids[i]]
    return mesh_vertices, semantic_ids, instance_ids, instance_bboxes


if __name__ == "__main__":
    SCANNET_DIR = "/opt/data2/SCANNET/scannet/scans"
    scan_names = os.listdir(SCANNET_DIR)
    for scan_name in scan_names:
        #now_dir = os.listdir('./scannet_positions_P90')
        #if scan_name+'_constant.npy' in now_dir:
        #    continue
        print(scan_name)
        scan_folder = "/opt/data2/SCANNET/scannet/scans/" + scan_name + "/"
        mesh_file = scan_folder + scan_name + "_vh_clean_2.ply"
        agg_file = scan_folder + scan_name + ".aggregation.json"
        seg_file = scan_folder + scan_name + "_vh_clean_2.0.010000.segs.json"
        meta_file = scan_folder + scan_name + ".txt"
        modelnet40_path = "./modelnet40_normal_resampled"
        positions, fw_points = generate_initial_random_positions(mesh_file, agg_file, seg_file, meta_file, modelnet40_path)
        #positions, fw_points = generate_initial_positions(mesh_file, agg_file, seg_file, meta_file, modelnet40_path)
        positions, stage_map = generate_gravity_aware_positions(positions, fw_points[0])
        floor_points = fw_points[0]
        wall_points = fw_points[1]
        constant_property = {}
        initial_variable = {}

        # object_id: ((x, y), OBJ_txt, minZ, Is_supporter, height)
        constant_property['floor'] = floor_points
        constant_property['wall'] = wall_points
        constant_property['stage_map'] = stage_map
        #floor_area = get_horizontal_area(floor_points)
        #num_floor_points = floor_points.shape[0]
        for key, value in positions.items():
            obj_xyz = np.loadtxt(value[2], delimiter=',')[:, 0:3]
            obj_xyz[:, [1, 2]] = obj_xyz[:, [2, 1]]
            # ddx, ddy, ddz = max(obj_xyz[:, 0]) - min(obj_xyz[:, 0]), max(obj_xyz[:, 1]) - min(obj_xyz[:, 1]), max(obj_xyz[:, 2]) - min(obj_xyz[:, 2])
            # dx, dy, dz = ddx * value[1][0], ddy * value[1][1], ddz * value[1][2]
            # S_larger = dx * dy * dz / min(dx, dy, dz)
            # num_points = 500 * S_larger
            minZ = min(obj_xyz[:, 2])
            Is_supporter = value[3]
            height = (value[-1] - value[0][2]) / value[1][2] if Is_supporter else None
            constant_property[key] = ((value[0][0], value[0][1]), value[2], minZ, Is_supporter, height)
            initial_variable[key] = ((0, 0), value[1], value[4])
        np.save('./scannet_positions_P90/'+scan_name+'_constant.npy', constant_property)
        np.save('./scannet_positions_P90/'+scan_name+'_variable.npy', initial_variable)
        #positions['floor_points'] = fw_points[0]
        #np.save('/home/xxw/3D/BR2/BackToReality/data_generation/ScanNet/scannet_positions_P90/' + scan_name + '.npy', positions)