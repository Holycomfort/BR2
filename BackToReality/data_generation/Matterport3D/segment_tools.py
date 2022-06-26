import open3d as o3d
import numpy as np
from plyfile import PlyData
from ins_ply import read_aggregation, read_segmentation, visualize, get_id_to_label
import json
import cv2
import sys, os

### Definition
# seg: List of vertex's index
# segindices: data in json
# segidx: the element of segindices, the index of a seg

def get_MER(points):
    xys = points[:, 0:2] * 1000
    xys = xys.astype('int')
    (x_center, y_center), (x_size, y_size), angle = cv2.minAreaRect(xys)
    x_center /= 1e3; y_center /= 1e3; y_size /= 1e3; x_size /= 1e3
    angle = angle / 180 * np.pi
    return (x_center, y_center), (x_size, y_size), angle


def get_mesh(scene_ply):
    mesh = o3d.io.read_triangle_mesh(scene_ply)
    return mesh


def get_normal(mesh):
    mesh.compute_vertex_normals()
    return np.array(mesh.triangle_normals)


def Is_horizontal(scene_vertices, seg):
    # scene_vertices: Nx3
    zs = scene_vertices[seg][:, 2]
    zs.sort()
    num = zs.shape[0]
    return zs[-1] - zs[num // 2] < 0.2 or zs[num // 2] - zs[0] < 0.2


def get_height(scene_vertices, scene_normal, seg):
    # scene_vertices: Nx3
    z_list = []
    for idx in seg:
        normal = scene_normal[idx]
        if abs(np.dot(normal, [0, 0, 1])) > 0.88:
            z_list.append(scene_vertices[idx][2])
    return np.mean(z_list)


def generate_seg_adjacency_matrix(plydata, segindices):
    # idx_to_segidx
    # idx: 0->num_segs-1
    segidxs = np.unique(segindices)
    seg_num = len(segidxs)
    idx_to_segidx = {}
    for i in range(seg_num):
        idx_to_segidx[i] = segidxs[i]
    segidx_to_idx = {value:key for key,value in idx_to_segidx.items()}
    adjacency_matrix = np.zeros([seg_num, seg_num])
    num_faces = plydata['face'].count
    for i in range(num_faces):
        face = plydata['face']['vertex_indices'][i]
        for idx in [[0,1], [0,2], [1,2]]:
            seg1 = segindices[face[idx[0]]]
            seg2 = segindices[face[idx[1]]]
            if seg1 != seg2:
                adjacency_matrix[segidx_to_idx[seg1]][segidx_to_idx[seg2]] = 1
                adjacency_matrix[segidx_to_idx[seg2]][segidx_to_idx[seg1]] = 1
    return adjacency_matrix, idx_to_segidx, segidx_to_idx


def get_neighbor(adj_matrix, idx_to_segidx, segidx_to_idx, segidx):
    idxs = list(np.reshape(np.argwhere(adj_matrix[segidx_to_idx[segidx]] == 1), -1))
    return [idx_to_segidx[idx] for idx in idxs]


def get_horizontal_area(scene_vertices, seg):
    xys = (np.array(scene_vertices)[:,:2])[seg]
    xys *= 1000
    xys = xys.astype('int')
    hull = cv2.convexHull(xys, clockwise=True, returnPoints=True)
    area = cv2.contourArea(hull)
    return area/1000000


def export_random(mesh_file, agg_file, seg_file, scan_name):
    ## correct some misleading things
    label_map = np.load("CONFIG/map2modelnet.npy", allow_pickle=True).item()
    for key, value in label_map.items():
        if "door" in key and key != "door":
            label_map[key] = 0
        if "cabinet" in key:
            label_map[key] = 5
    label_map["countertop"] = 34
    label_map["shelf"] = 5
    label_map["sink"] = 0
    
    # ## for WSD, we only need furniture, so objects should be ignored
    # ## c10 is the 10 category in SUN-RGBD detection
    # c10_in_modelnet40 = [3, 34, 31, 9, 36, 13, 15, 24, 5, 2]
    # for key, value in label_map.items():
    #     if value not in c10_in_modelnet40:
    #         label_map[key] = 0

    mesh = get_mesh(mesh_file)
    mesh_vertices = np.array(mesh.vertices)
    scene_normal = get_normal(mesh)
    with open(seg_file) as fp:
        j = json.load(fp)
    segindices = j['segIndices']

    plydata = PlyData.read(mesh_file)

    ## Load semantic and instance labels
    # note that seg_to_verts means segidx_to_seg according to the definition above
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    object_id_to_label_id = {}
    error_segs = []
    for label, segs in label_to_segs.items():
        try:
            label_id = label_map[label]
        except:
            label_map[label] = 0
            label_id = 0
        for segidx in segs:
            try:
                verts = seg_to_verts[segidx]
            except:
                error_segs.append(segidx)
                continue
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    
    # obj_id: [(x, y, z), label, modelnet_id, Is_labeled]
    xyz_obj_dict = {}
    id_to_label = get_id_to_label(agg_file)

    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            try:
                verts = seg_to_verts[seg]
            except:
                continue
            if label_ids[verts][0] == 0:
                instance_ids[verts] = 0
            else:
                instance_ids[verts] = object_id
    
    obj_idx = -1
    for object_id, segs in object_id_to_segs.items():
        modelnet_id = label_map[id_to_label[object_id]]
        obj_pc = mesh_vertices[instance_ids==object_id, 0:3]
        if len(obj_pc) == 0 or modelnet_id == 0: continue
        #if modelnet_id == 0: continue
        #if modelnet_id in [2,3,4,9,12,13,14,15,24,31,33,34,36]:
        #    obj_idx += 1
        Is_labeled = False
        if modelnet_id in [2,3,4,9,12,13,14,15,24,31,33,34,36]:
            obj_idx += 1
            Is_labeled = True
        zmin = np.min(obj_pc[:,2])
        zmax = np.max(obj_pc[:,2])
        (x_center, y_center), (x_size, y_size), angle = get_MER(obj_pc)
        x, y, z = x_center, y_center, (zmin+zmax)/2
        error = np.load('CONFIG/annotation_error.npy')
        scan_name_to_idx = np.load('CONFIG/name2idx.npy', allow_pickle=True).item()
        if scan_name in scan_name_to_idx.keys() and Is_labeled:
            dx, dy, dz = np.array([x_size, y_size, zmax-zmin]) * error[scan_name_to_idx[scan_name], obj_idx, :]
        else:
            dx, dy, dz = 0, 0, 0
        x += dx; y += dy; z += dz
        xyz_obj_dict[object_id] = [(x, y, z), id_to_label[object_id], modelnet_id, Is_labeled]

    return xyz_obj_dict, mesh_vertices
