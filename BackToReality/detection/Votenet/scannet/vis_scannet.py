import torch
import sys, os
import numpy as np
from scannet_detection_dataset import ScannetDetectionDataset, ScannetDatasetConfig_md40
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util



DUMP_CONF_THRESH = 0.5 # Dump boxes with obj prob larger than that.

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def dump_results(end_points, dump_dir, config, inference_switch=False):
    ''' Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    '''
    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))

    # INPUT
    point_clouds = end_points['point_clouds']
    colors = end_points['pcl_color']
    batch_size = point_clouds.shape[0]

    # # NETWORK OUTPUTS
    # seed_xyz = end_points['seed_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    # if 'vote_xyz' in end_points:
    #     aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
    #     vote_xyz = end_points['vote_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    #     aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
    # objectness_scores = end_points['objectness_scores'].detach().cpu().numpy() # (B,K,2)
    # pred_center = end_points['center'].detach().cpu().numpy() # (B,K,3)
    # pred_heading_class = torch.argmax(end_points['heading_scores'], -1) # B,num_proposal
    # pred_heading_residual = torch.gather(end_points['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    # pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
    # pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
    # pred_size_class = torch.argmax(end_points['size_scores'], -1) # B,num_proposal
    # pred_size_residual = torch.gather(end_points['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    # pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

    # # OTHERS
    # pred_mask = end_points['pred_mask'] # B,num_proposal
    idx_beg = 0

    pc = point_clouds
    cl = colors
    #     objectness_prob = softmax(objectness_scores[i,:,:])[:,1] # (K,)

    #     # Dump various point clouds
    pc_util.write_ply_rgb(pc, cl, os.path.join(dump_dir, '%03d_pc.obj'%(idx_beg)))
    #     pc_util.write_ply(seed_xyz[i,:,:], os.path.join(dump_dir, '%02d_seed_pc.ply'%(idx_beg+i)))
    #     if 'vote_xyz' in end_points:
    #         pc_util.write_ply(end_points['vote_xyz'][i,:,:], os.path.join(dump_dir, '%02d_vgen_pc.ply'%(idx_beg+i)))
    #         pc_util.write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%02d_aggregated_vote_pc.ply'%(idx_beg+i)))
    #         pc_util.write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%02d_aggregated_vote_pc.ply'%(idx_beg+i)))
    #     pc_util.write_ply(pred_center[i,:,0:3], os.path.join(dump_dir, '%02d_proposal_pc.ply'%(idx_beg+i)))
    #     if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
    #         pc_util.write_ply(pred_center[i,objectness_prob>DUMP_CONF_THRESH,0:3], os.path.join(dump_dir, '%02d_confident_proposal_pc.ply'%(idx_beg+i)))

    #     # Dump predicted bounding boxes
    #     if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
    #         num_proposal = pred_center.shape[1]
    #         obbs = []
    #         for j in range(num_proposal):
    #             obb = config.param2obb(pred_center[i,j,0:3], pred_heading_class[i,j], pred_heading_residual[i,j],
    #                             pred_size_class[i,j], pred_size_residual[i,j])
    #             obbs.append(obb)
    #         if len(obbs)>0:
    #             obbs = np.vstack(tuple(obbs)) # (num_proposal, 7)
    #             pc_util.write_oriented_bbox(obbs[objectness_prob>DUMP_CONF_THRESH,:], os.path.join(dump_dir, '%02d_pred_confident_bbox.ply'%(idx_beg+i)))
    #             pc_util.write_oriented_bbox(obbs[np.logical_and(objectness_prob>DUMP_CONF_THRESH, pred_mask[i,:]==1),:], os.path.join(dump_dir, '%02d_pred_confident_nms_bbox.ply'%(idx_beg+i)))
    #             pc_util.write_oriented_bbox(obbs[pred_mask[i,:]==1,:], os.path.join(dump_dir, '%02d_pred_nms_bbox.ply'%(idx_beg+i)))
    #             pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%02d_pred_bbox.ply'%(idx_beg+i)))

    # # Return if it is at inference time. No dumping of groundtruths
    # if inference_switch:
    #     return

    # LABELS
    gt_center = end_points['center_label'] # (B,MAX_NUM_OBJ,3)
    gt_mask = end_points['box_label_mask'] # B,K2
    gt_heading_class = end_points['heading_class_label'] # B,K2
    gt_heading_residual = end_points['heading_residual_label'] # B,K2
    gt_size_class = end_points['size_class_label'] # B,K2
    gt_size_residual = end_points['size_residual_label'] # B,K2,3

    # Dump GT bounding boxes
    obbs = []
    for j in range(gt_center.shape[0]):
        if gt_mask[j] == 0: continue
        obb = config.param2obb(gt_center[j,0:3], gt_heading_class[j], gt_heading_residual[j],
                        gt_size_class[j], gt_size_residual[j])
        obbs.append(obb)

    if len(obbs)>0:
        obbs = np.vstack(tuple(obbs)) # (num_gt_objects, 7)
        pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%02d_gt_bbox.ply'%(idx_beg)))

    # OPTIONALL, also dump prediction and gt details
    if 'batch_pred_map_cls' in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, '%02d_pred_map_cls.txt'%(ii)), 'w')
            for t in end_points['batch_pred_map_cls'][ii]:
                fout.write(str(t[0])+' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write(' '+str(t[2]))
                fout.write('\n')
            fout.close()
    if 'batch_gt_map_cls' in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, '%02d_gt_map_cls.txt'%(ii)), 'w')
            for t in end_points['batch_gt_map_cls'][ii]:
                fout.write(str(t[0])+' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write('\n')
            fout.close()


if __name__=='__main__': 
    dset = ScannetDetectionDataset(use_height=True, num_points=50000)
    config = ScannetDatasetConfig_md40()
    for d in dset:
        idx = d['scan_idx']
        scan_name = dset.scan_names[idx]
        if scan_name != 'scene0038_00':
            continue
        else:
            dump_results(d, './xxwddd', config)
