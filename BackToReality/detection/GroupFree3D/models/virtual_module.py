import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
sys.path.append(os.path.join(ROOT_DIR, 'matterport'))
import pc_util
from model_util_scannet import ScannetDatasetConfig_md40
from model_util_matterport import MatterportDatasetConfig_md40


def get_MER(points):
    xys = points[:, 0:2] * 1000
    xys = xys.astype('int')
    (x_center, y_center), (x_size, y_size), angle = cv2.minAreaRect(xys)
    x_center /= 1e3; y_center /= 1e3; y_size /= 1e3; x_size /= 1e3
    angle = angle / 180 * np.pi
    return (x_center, y_center), (x_size, y_size), angle


# y-->x rotate t
def rotz(t, device):
    """Rotation about the z-axis."""
    c = torch.cos(t)
    s = torch.sin(t)
    row1 = torch.cat([c, -s, torch.Tensor([0]).to(device)])
    row2 = torch.cat([s, c, torch.Tensor([0]).to(device)])
    row3 = torch.Tensor([0, 0, 1]).to(device)
    return torch.stack([row1, row2, row3])


def create_color_palette():
    return [
       (255, 255, 255),
       (152, 223, 138),
       (31, 119, 180),
       (255, 187, 120),
       (188, 189, 34),
       (140, 86, 75),
       (255, 152, 150),
       (214, 39, 40),
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),  
       (247, 182, 210),
       (66, 188, 102), 
       (219, 219, 141),
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14),
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),
       (112, 128, 144),
       (96, 207, 209), 
       (227, 119, 194),
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),
       (100, 85, 144),
       (214, 39, 40),
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),  
       (247, 182, 210),
       (66, 188, 102), 
       (219, 219, 141),
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14),
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),
       (112, 128, 144),
       (96, 207, 209), 
       (227, 119, 194),
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),
       (100, 85, 144),
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),  
       (247, 182, 210),
       (66, 188, 102), 
       (219, 219, 141),
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14),
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),
       (112, 128, 144),
       (96, 207, 209), 
       (227, 119, 194),
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),
       (100, 85, 144),
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),  
       (247, 182, 210),
       (66, 188, 102), 
       (219, 219, 141),
       (140, 57, 197), 
       (202, 185, 52),
       (152, 223, 138),
       (31, 119, 180),
       (255, 187, 120),
       (188, 189, 34),
       (140, 86, 75),
       (255, 152, 150),
       (214, 39, 40),
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),  
       (247, 182, 210),
       (66, 188, 102), 
       (219, 219, 141),
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14),
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),
       (112, 128, 144),
       (96, 207, 209), 
       (227, 119, 194),
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),
       (100, 85, 144),
       (214, 39, 40),
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),  
       (247, 182, 210),
       (66, 188, 102), 
       (219, 219, 141),
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14),
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),
       (112, 128, 144),
       (96, 207, 209), 
       (227, 119, 194),
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),
       (100, 85, 144),
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),  
       (247, 182, 210),
       (66, 188, 102), 
       (219, 219, 141),
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14),
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),
       (112, 128, 144),
       (96, 207, 209), 
       (227, 119, 194),
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),
       (100, 85, 144),
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),  
       (247, 182, 210),
       (66, 188, 102), 
       (219, 219, 141),
       (140, 57, 197), 
       (202, 185, 52)
    ]


class virtual_generator(nn.Module):
    def __init__(self, num_scenes, num_points=40000, dataset='scannet'):
        super().__init__()
        self.DC = ScannetDatasetConfig_md40() if dataset == 'scannet' else MatterportDatasetConfig_md40()
        self.MAX_NUM_OBJ = 64 if dataset == 'scannet' else 256

        self.num_points = num_points
        self.rotate_gt = False if dataset == 'scannet' else True
        self.key2id = [{} for i in range(num_scenes)]

        self.floor_points = [None for i in range(num_scenes)]
        self.wall_points = [None for i in range(num_scenes)]
        self.stage_map = [None for i in range(num_scenes)]
        self.object_constant = [None for i in range(num_scenes)]

        self.object_dxy = torch.zeros(num_scenes, self.MAX_NUM_OBJ, 2)
        self.object_scale = torch.zeros(num_scenes, self.MAX_NUM_OBJ, 3)
        self.object_theta = torch.zeros(num_scenes, self.MAX_NUM_OBJ, 1)
        self.object_mask = torch.zeros(num_scenes, self.MAX_NUM_OBJ)

        if dataset == 'scannet':
            self.storage_path = '/home/xxw/3D/BR2/BackToReality/data_generation/ScanNet/'
            self.dir_path = 'scannet_positions_P90/'
        else:
            self.storage_path = '/home/xxw/3D/BR2/BackToReality/data_generation/Matterport3D/'
            self.dir_path = 'matterport_positions_Prandom/'
        self.scan_idx_to_name = {}
        scan_name_to_idx = np.load(self.storage_path+'CONFIG/name2idx.npy', allow_pickle=True).item()
        initial_variables = os.listdir(self.storage_path+self.dir_path)
        for f in initial_variables:
            try:
                scan_idx = scan_name_to_idx[f[:12]]
                self.scan_idx_to_name[str(scan_idx)] = f[:12]
            except:
                continue
            if 'constant' in f:
                c = np.load(self.storage_path+self.dir_path+f, allow_pickle=True).item()
                self.floor_points[scan_idx] = c['floor']
                self.wall_points[scan_idx] = c['wall']
                self.stage_map[scan_idx] = c['stage_map']
                c.pop('floor'); c.pop('wall'); c.pop('stage_map')
                self.object_constant[scan_idx] = c
            else:
                v = np.load(self.storage_path+self.dir_path+f, allow_pickle=True).item()
                for i, (key, value) in enumerate(v.items()):
                    self.key2id[scan_idx][key] = i
                    self.object_dxy[scan_idx][i] = torch.Tensor(value[0])
                    self.object_scale[scan_idx][i] = torch.Tensor(value[1])
                    self.object_theta[scan_idx][i] = value[2]
                self.object_mask[scan_idx][:len(v)] = 1
        self.object_dxy = nn.Parameter(self.object_dxy)
        self.object_scale = nn.Parameter(self.object_scale)
        self.object_theta = nn.Parameter(self.object_theta / 180 * np.pi, requires_grad=self.rotate_gt) # y-->x (anticlockwise)

    def generate_one_batch(self, flip_YZ, flip_XZ, rot_angle, s_idx, device, Is_floor, Is_wall, Is_vis, use_height):
        stage_map = self.stage_map[s_idx]
        floor = self.floor_points[s_idx]
        if len(floor) == 0:
            floor = np.array([[0, 0, 0]])
        wall = self.wall_points[s_idx]
        object_constant = self.object_constant[s_idx]
        # DataAug for Background. Note rotation is x-->y (clockwise)!
        if flip_YZ:
            floor[:, 0] = -1 * floor[:, 0]
            wall[:, 0] = -1 * wall[:, 0]
        if flip_XZ:
            floor[:, 1] = -1 * floor[:, 1]
            wall[:, 1] = -1 * wall[:, 1]
        rot_mat = pc_util.rotz(rot_angle)
        floor = np.dot(floor, np.transpose(rot_mat))
        wall = np.dot(wall, np.transpose(rot_mat))
        # compute the height of support surface
        z_dict = {}
        for key_id in stage_map.keys():
            obj_id = self.key2id[s_idx][key_id]
            z_dict[key_id] = np.mean(floor[:,2]) - self.object_scale[s_idx][obj_id][2]*object_constant[key_id][2]
            for value_id in stage_map[key_id]:
                vobj_id = self.key2id[s_idx][value_id]
                z_dict[value_id] = z_dict[key_id] + self.object_scale[s_idx][obj_id][2]*object_constant[key_id][4] \
                  - self.object_scale[s_idx][vobj_id][2]*object_constant[value_id][2]
        # point_clouds, center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label
        # size_gts, sem_cls_label, box_label_mask, scan_idx
        scene_points = []
        if Is_vis: scene_color = []
        centers = torch.zeros((self.MAX_NUM_OBJ, 3))
        angle_classes = torch.zeros((self.MAX_NUM_OBJ,))
        angle_residuals = torch.zeros((self.MAX_NUM_OBJ,))
        angle = torch.zeros((self.MAX_NUM_OBJ,))
        size_classes = torch.zeros((self.MAX_NUM_OBJ,))
        size_residuals = torch.zeros((self.MAX_NUM_OBJ, 3))
        size_residuals_normalized = torch.zeros((self.MAX_NUM_OBJ, 3))
        size_gts = torch.zeros((self.MAX_NUM_OBJ, 3))
        sem_cls = torch.zeros((self.MAX_NUM_OBJ,))
        target_bboxes_mask = self.object_mask[s_idx]
        for key, obj in object_constant.items():
            obj_id = self.key2id[s_idx][key]
            obj_xyz = pd.read_csv('/opt/data2/MODELNET40/'+obj[1], sep=',').values[:, 0:3]
            obj_xyz[:, [1, 2]] = obj_xyz[:, [2, 1]]
            obj_xyz[:, 0] -= np.mean(obj_xyz[:, 0]); obj_xyz[:, 1] -= np.mean(obj_xyz[:, 1])
            ddx, ddy, ddz = np.max(obj_xyz[:, 0]) - np.min(obj_xyz[:, 0]), np.max(obj_xyz[:, 1]) - np.min(obj_xyz[:, 1]), np.max(obj_xyz[:, 2]) - np.min(obj_xyz[:, 2])
            dx, dy, dz = ddx * self.object_scale[s_idx][obj_id][0], ddy * self.object_scale[s_idx][obj_id][1], ddz * self.object_scale[s_idx][obj_id][2]
            S_larger = dx * dy * dz / min(dx, dy, dz)
            num_point = max(int(S_larger * 1000), 100)
            obj_xyz = torch.Tensor(pc_util.random_sampling(obj_xyz, num_point)).to(device)
            obj_xyz *= self.object_scale[s_idx][obj_id]
            center = torch.cat([(obj[0][0]+self.object_dxy[s_idx][obj_id][0]).unsqueeze(-1), (obj[0][1]+self.object_dxy[s_idx][obj_id][1]).unsqueeze(-1), z_dict[key].unsqueeze(-1)])
            theta = self.object_theta[s_idx][obj_id]
            '''Warning: torch.mm on GPU is slower than on CPU'''
            obj_xyz = torch.mm(obj_xyz.cpu(), rotz(theta, device).cpu()).to(device) + center
            # DataAug for Obj. Note rotation is x-->y (clockwise)!
            if flip_YZ:
                obj_xyz[:, 0] = -1 * obj_xyz[:, 0]
                center[0] = -1 * center[0]
            if flip_XZ:
                obj_xyz[:, 1] = -1 * obj_xyz[:, 1]
                center[1] = -1 * center[1]
            obj_xyz = torch.mm(obj_xyz, torch.Tensor(np.transpose(rot_mat)).to(device))
            center = torch.mm(center.unsqueeze(0), torch.Tensor(np.transpose(rot_mat)).to(device)).squeeze(0)
            scene_points.append(obj_xyz)
            if Is_vis: scene_color.append(torch.Tensor(create_color_palette()[key]).repeat(len(obj_xyz),1)/255.0)
            # Ground-truth
            if self.rotate_gt:
                type_name = obj[1].split('/')[-2]
                zmin = torch.min(obj_xyz[:,2])
                zmax = torch.max(obj_xyz[:,2])
                sem_cls[obj_id] = self.DC.type2class[type_name]
                centers[obj_id] = torch.cat([center[0].unsqueeze(-1), center[1].unsqueeze(-1), ((zmin+zmax)/2).unsqueeze(-1)])
                angle[obj_id] = (np.pi / 2 - theta + rot_angle) % np.pi
                if angle[obj_id] > np.pi / 2:
                    angle[obj_id] -= np.pi / 2
                    size_gts[obj_id] = torch.cat([dx.unsqueeze(-1), dy.unsqueeze(-1), dz.unsqueeze(-1)])
                else:
                    size_gts[obj_id] = torch.cat([dy.unsqueeze(-1), dx.unsqueeze(-1), dz.unsqueeze(-1)])
                size_classes[obj_id], size_residuals[obj_id], size_residuals_normalized[obj_id] = self.DC.size2class(size_gts[obj_id], type_name, require_grad=True)
                angle_classes[obj_id], angle_residuals[obj_id] = self.DC.angle2class(angle[obj_id].detach())
            else:
                type_name = obj[1].split('/')[-2]
                sem_cls[obj_id] = self.DC.type2class[type_name]
                xmin = torch.min(obj_xyz[:,0])
                ymin = torch.min(obj_xyz[:,1])
                zmin = torch.min(obj_xyz[:,2])
                xmax = torch.max(obj_xyz[:,0])
                ymax = torch.max(obj_xyz[:,1])
                zmax = torch.max(obj_xyz[:,2])
                centers[obj_id] = torch.cat([((xmin+xmax)/2).unsqueeze(-1), ((ymin+ymax)/2).unsqueeze(-1), ((zmin+zmax)/2).unsqueeze(-1)])
                size_gts[obj_id] = torch.cat([(xmax-xmin).unsqueeze(-1), (ymax-ymin).unsqueeze(-1), (zmax-zmin).unsqueeze(-1)])
                size_classes[obj_id], size_residuals[obj_id], size_residuals_normalized[obj_id] = self.DC.size2class(size_gts[obj_id], type_name, require_grad=True)
        if Is_floor:
            scene_points = torch.cat(scene_points + [torch.Tensor(floor).to(device)])
            if Is_vis: scene_color = torch.cat(scene_color + [torch.Tensor([[0.596,0.875,0.541]]).repeat(len(floor),1)])
        if Is_wall:
            scene_points = torch.cat(scene_points + [torch.Tensor(wall).to(device)])
            if Is_vis: scene_color = torch.cat(scene_color + [torch.Tensor([[0.682,0.780,0.910]]).repeat(len(wall),1)])
        if not (Is_floor or Is_wall):
            scene_points = torch.cat(scene_points)
        if Is_vis: scene_points = torch.cat([scene_points, scene_color], dim=-1)
        point_clouds = scene_points[torch.randint(len(scene_points), (self.num_points,))] if len(scene_points) < self.num_points \
          else scene_points[torch.randperm(len(scene_points))[:self.num_points]]
        if Is_vis: point_clouds, scene_color = point_clouds[:, 0:3], point_clouds[:, 3:6].detach()
        if use_height:
            # Use height
            height = point_clouds[:,2] - np.mean(floor[:,2])
            point_clouds = torch.cat([point_clouds, height.unsqueeze(-1)],1)
        return {'point_clouds': point_clouds, 'center_label': centers.detach().to(device), 'heading_class_label': angle_classes.long().to(device), 
          'heading_residual_label': angle_residuals.to(device), 'size_class_label': size_classes.long().to(device),
          'size_residual_label': size_residuals.to(device), 'size_gts': size_gts.detach().to(device), 'sem_cls_label': sem_cls.long().to(device),
          'box_label_mask': target_bboxes_mask.to(device), 'scan_idx': torch.Tensor([s_idx]).to(device), 'virtual_centers': centers.to(device), 
          'virtual_size_residuals_normalized': size_residuals_normalized.to(device), 'virtual_angle': angle.to(device),
          'color': scene_color if Is_vis else torch.Tensor([])}

    def forward(self, data_augmentation, scan_idx, Is_floor=True, Is_wall=False, Is_vis=False, use_height=False):
        # data_augmentation: (flip_YZ, flip_XZ, rot_angle)
        flip_YZs, flip_XZs, rot_angles = list(data_augmentation)
        batch_size = scan_idx.shape[0]
        ret_dicts = []
        for i in range(batch_size):
            flip_YZ, flip_XZ, rot_angle = flip_YZs[i], flip_XZs[i], rot_angles[i]
            device = rot_angle.device
            flip_YZ, flip_XZ, rot_angle = flip_YZ.cpu(), flip_XZ.cpu(), rot_angle.cpu()
            s_idx = scan_idx[i]
            ret = self.generate_one_batch(flip_YZ, flip_XZ, rot_angle, s_idx, device, Is_floor, Is_wall, Is_vis, use_height)
            ret_dicts.append(ret)
        keys = ret.keys()
        return {key: torch.stack([ret_dict[key] for ret_dict in ret_dicts]) for key in keys}


if __name__ == "__main__":
    '''
    G = virtual_generator(1201)
    print(G.object_dxy, G.object_scale, G.object_theta)
    print('Init!')
    data = G(([False], [False], [0]), np.array([827]), Is_floor=True, Is_wall=False, Is_vis=True)
    np.savetxt('scene0000_00_P90BF.txt', torch.cat([data['point_clouds'][0][:,:3].detach().cpu(), data['color'][0]], dim=-1).numpy())
    checkpoint = torch.load('../logs/log_scannet/log_Votenet_BRDLE_iou2/G.tar')
    G.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(G.object_dxy, G.object_scale, G.object_theta)
    data = G(([False], [False], [0]), np.array([827]), Is_floor=True, Is_wall=False, Is_vis=True)
    np.savetxt('scene0000_00_P90AF.txt', torch.cat([data['point_clouds'][0][:,:3].detach().cpu(), data['color'][0]], dim=-1).numpy())
    '''
    G = virtual_generator(1543, dataset='matterport')
    print('Init!')
    data = G(([False], [False], [0]), np.array([0]), Is_floor=True, Is_wall=False, Is_vis=True)
    np.savetxt('scene0000_00_PrandomBF.txt', torch.cat([data['point_clouds'][0][:,:3].detach().cpu(), data['color'][0]], dim=-1).numpy())
    checkpoint = torch.load('../logs/log_matterport/log_Votenet_BRDLE/G.tar')
    G.load_state_dict(checkpoint['model_state_dict'], strict=False)
    data = G(([False], [False], [0]), np.array([0]), Is_floor=True, Is_wall=False, Is_vis=True)
    np.savetxt('scene0000_00_PrandomAF.txt', torch.cat([data['point_clouds'][0][:,:3].detach().cpu(), data['color'][0]], dim=-1).numpy())
    