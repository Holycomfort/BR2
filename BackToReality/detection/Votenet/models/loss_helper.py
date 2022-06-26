# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from utils.nms import lhs_3d_faster_samecls, nms_3d_faster_samecls
from models.ap_helper import flip_axis_to_camera
from utils.box_util import get_3d_box
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
import mmd
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from nn_distance import nn_distance, huber_loss

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness
MAX_NUM_OBJ = 64


def compute_vote_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'].repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_weak_vote_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3

    gt_center = end_points['center_label'][:,:,0:3] # B,K2,3

    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz, gt_center, l1=True) # dist1: B,num_seed*vote_factor, dist2: B,K2
    dist1 = dist1.view(batch_size, num_seed, -1) # dist1: B,num_seed,vote_factor
    votes_dist, _ = torch.min(dist1, dim=2) # (B,num_seed,vote_factor) to (B,num_seed,)
    box_label_mask = end_points['box_label_mask'] # B,K2
    vote_loss = torch.mean(votes_dist) + torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    
    return vote_loss

def compute_objectness_loss(end_points):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = end_points['aggregated_vote_xyz']
    # aggregated_vote_xyz = end_points['center']
    gt_center = end_points['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = end_points['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(end_points, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = end_points['center']
    gt_center = end_points['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = end_points['box_label_mask']
    objectness_label = end_points['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(end_points['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(end_points['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(end_points['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(end_points['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(end_points['size_scores'].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(end_points['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized']*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss


def smoothl1_loss(error, delta=1.0):
    """Smooth L1 loss.
    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    |x| - 0.5 * d               if |x|>d
    """
    diff = torch.abs(error)
    loss = torch.where(diff < delta, 0.5 * diff * diff / delta, diff - 0.5 * delta)
    return loss


def compute_center_and_sem_cls_loss(end_points, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = end_points['center']
    gt_center = end_points['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = end_points['box_label_mask']
    objectness_label = end_points['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2
    '''
    pred_center = end_points['center']
    gt_center = end_points['center_label'][:, :, 0:3]
    size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment)  # select (B,K) from (B,K2)
    center_margin = torch.from_numpy(0.05 * mean_size_arr[size_class_label.cpu(), :]).cuda()   # (B,K,3)

    objectness_label = end_points['objectness_label'].float()
    object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
    assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
    center_loss = smoothl1_loss(assigned_gt_center - pred_center)  # (B,K)
    center_loss -= center_margin
    center_loss[center_loss < 0] = 0
    center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
    '''
    
    # Compute size loss
    size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(end_points['size_scores'].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, size_class_loss, sem_cls_loss


def get_loss(end_points, config):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
        compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
        torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    end_points['neg_ratio'] = \
        torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
        compute_box_and_sem_cls_loss(end_points, config)
    end_points['center_loss'] = center_loss
    end_points['heading_cls_loss'] = heading_cls_loss
    end_points['heading_reg_loss'] = heading_reg_loss
    end_points['size_cls_loss'] = size_cls_loss
    end_points['size_reg_loss'] = size_reg_loss
    end_points['sem_cls_loss'] = sem_cls_loss
    box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss
    end_points['box_loss'] = box_loss

    # Final loss function
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss
    loss *= 10
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc

    return loss, end_points


def get_loss_weak(end_points, config, without_size=False):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Vote loss
    vote_loss = compute_weak_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
        compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
        torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    end_points['neg_ratio'] = \
        torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    # Box loss and sem cls loss
    center_loss, size_cls_loss, sem_cls_loss = compute_center_and_sem_cls_loss(end_points, config)
    end_points['center_loss'] = center_loss
    if not without_size:
        end_points['size_cls_loss'] = size_cls_loss
    end_points['sem_cls_loss'] = sem_cls_loss
    
    if not without_size:
        box_loss = center_loss + 0.1*size_cls_loss
    else:
        box_loss = center_loss
    sem_cls_loss = sem_cls_loss

    # Final loss function
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss
    loss *= 10
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc

    return loss, end_points


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5),
                                   putting more focus on hard, misclassi?ed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True,sigmoid=False,reduce=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.sigmoid = sigmoid
        self.reduce = reduce
    def forward(self, inputs, targets, global_weight=None):
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        if self.sigmoid:
            P = F.sigmoid(inputs)
            #F.softmax(inputs)
            if targets == 0:
                probs = 1 - P#(P * class_mask).sum(1).view(-1, 1)
                probs = probs.clamp(min=1e-4, max=1.0)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
            if targets == 1:
                probs = P  # (P * class_mask).sum(1).view(-1, 1)
                probs = probs.clamp(min=1e-4, max=1.0)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
        else:
            #inputs = F.sigmoid(inputs)
            P = F.softmax(inputs, dim=-1)
            class_mask = inputs.data.new(N, C).fill_(0)
            class_mask = Variable(class_mask)
            ids = targets.view(-1, 1)
            class_mask.scatter_(1, ids.data, 1.)
            # print(class_mask)


            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            alpha = self.alpha[ids.data.view(-1)]

            probs = (P * class_mask).sum(1).view(-1, 1)
            probs = probs.clamp(min=1e-4, max=1.0)

            log_p = probs.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)

            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
            # print('-----bacth_loss------')
            # print(batch_loss)

        if not self.reduce:
            return batch_loss
        if self.size_average:
            if global_weight is not None:
                global_weight = global_weight.view(-1, 1)
                batch_loss = batch_loss * global_weight
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def compute_iou_labels(end_points, pred_votes, pred_center, pred_sem_cls, pred_objectness, pred_heading_scores,
                       pred_heading_residuals, pred_size_scores, pred_size_residuals, config_dict, reverse=False):

    # the end_points labels are not transformed
    center_label = end_points['center_label']
    zero_mask = (1 - end_points['box_label_mask']).unsqueeze(-1).expand(-1, -1, 3).bool()
    center_label[zero_mask] = -1000
    heading_class_label = end_points['heading_class_label']
    heading_residual_label = end_points['heading_residual_label']
    size_class_label = end_points['size_class_label']
    size_residual_label = end_points['size_residual_label']

    pred_heading_class = torch.argmax(pred_heading_scores, -1)
    pred_heading_residual = torch.gather(pred_heading_residuals, 2, pred_heading_class.unsqueeze(-1)).squeeze(2)
    pred_size_class = torch.argmax(pred_size_scores, -1)
    pred_size_class_inds = pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3)
    pred_size_residual = torch.gather(pred_size_residuals, 2, pred_size_class_inds).squeeze(2) # B, num_proposals, 3

    dist1, object_assignment, _, _ = nn_distance(pred_votes, center_label)
    euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
    batch_size, num_proposal = euclidean_dist1.shape[:2]
    objectness_label = torch.zeros((batch_size, num_proposal), dtype=torch.long).cuda()
    objectness_label[euclidean_dist1 < NEAR_THRESHOLD] = 1

    # ------------------------- GT BBOX ----------------------------------------
    gt_size = config_dict['dataset_config'].class2size_gpu(size_class_label, size_residual_label)
    gt_angle = config_dict['dataset_config'].class2angle_gpu(heading_class_label, heading_residual_label)
    gt_bbox = torch.cat([center_label, gt_size, -gt_angle[:, :, None]], dim=2)

    pred_size = config_dict['dataset_config'].class2size_gpu(pred_size_class.detach(), pred_size_residual)
    pred_size[pred_size <= 0] = 1e-6

    if config_dict['dataset_config'].num_heading_bin == 1:
        pred_angle = torch.zeros(pred_size.shape[:2]).cuda()
    else:
        pred_angle = config_dict['dataset_config'].class2angle_gpu(pred_heading_class.detach(), pred_heading_residual)
    pred_bbox = torch.cat([pred_center, pred_size, -pred_angle[:, :, None]], axis=2)

    end_points['pred_bbox'] = pred_bbox
    pred_num = pred_bbox.shape[1]
    gt_num = gt_bbox.shape[1]

    # start = time.time()
    gt_bbox_ = gt_bbox.view(-1, 7)
    pred_bbox_ = pred_bbox.view(-1, 7)
    iou_labels = boxes_iou3d_gpu(pred_bbox_, gt_bbox_)
    iou_labels, object_assignment = iou_labels.view(batch_size * pred_num, batch_size, -1).max(dim=2)
    inds = torch.arange(batch_size).cuda().unsqueeze(1).expand(-1, pred_num).contiguous().view(-1, 1)
    iou_labels = iou_labels.gather(dim=1, index=inds).view(batch_size, -1)
    iou_labels = iou_labels.detach()
    object_assignment = object_assignment.gather(dim=1, index=inds).view(batch_size, -1)
    return iou_labels, objectness_label, object_assignment


def get_loss_DA(end_points_S, end_points_T, config, Is_mmd=False, epoch=-1):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, global_d_pred, vote_xyz, local_d_pred,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    vote_coefficient = 0.1
    obj_coefficient = 0.1
    cls_coefficient = 0.1

    # Vote loss
    vote_loss_S = compute_weak_vote_loss(end_points_S)
    vote_loss_T = compute_weak_vote_loss(end_points_T)
    vote_loss = vote_coefficient*vote_loss_S + vote_loss_T
    end_points_S['vote_loss'] = vote_loss_S
    end_points_T['vote_loss'] = vote_loss_T

    # Obj loss
    objectness_loss_S, objectness_label_S, objectness_mask_S, object_assignment = \
        compute_objectness_loss(end_points_S)
    end_points_S['objectness_loss'] = objectness_loss_S
    end_points_S['objectness_label'] = objectness_label_S
    end_points_S['objectness_mask'] = objectness_mask_S
    end_points_S['object_assignment'] = object_assignment
    total_num_proposal = objectness_label_S.shape[0]*objectness_label_S.shape[1]
    end_points_S['pos_ratio'] = \
        torch.sum(objectness_label_S.float().cuda())/float(total_num_proposal)
    end_points_S['neg_ratio'] = \
        torch.sum(objectness_mask_S.float())/float(total_num_proposal) - end_points_S['pos_ratio']
    
    objectness_loss_T, objectness_label_T, objectness_mask_T, object_assignment = \
        compute_objectness_loss(end_points_T)
    end_points_T['objectness_loss'] = objectness_loss_T
    end_points_T['objectness_label'] = objectness_label_T
    end_points_T['objectness_mask'] = objectness_mask_T
    end_points_T['object_assignment'] = object_assignment
    total_num_proposal = objectness_label_T.shape[0]*objectness_label_T.shape[1]
    end_points_T['pos_ratio'] = \
        torch.sum(objectness_label_T.float().cuda())/float(total_num_proposal)
    end_points_T['neg_ratio'] = \
        torch.sum(objectness_mask_T.float())/float(total_num_proposal) - end_points_T['pos_ratio']
    
    objectness_loss = obj_coefficient*objectness_loss_S + objectness_loss_T

    # Box loss and sem cls loss
    center_loss_S, heading_cls_loss, heading_reg_loss, size_cls_loss_S, size_reg_loss, sem_cls_loss_S = \
        compute_box_and_sem_cls_loss(end_points_S, config)
    end_points_S['center_loss'] = center_loss_S
    end_points_S['heading_cls_loss'] = heading_cls_loss
    end_points_S['heading_reg_loss'] = heading_reg_loss
    end_points_S['size_cls_loss'] = size_cls_loss_S
    end_points_S['size_reg_loss'] = size_reg_loss
    end_points_S['sem_cls_loss'] = sem_cls_loss_S
    box_loss_S = center_loss_S + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss_S + size_reg_loss
    end_points_S['box_loss'] = box_loss_S
    
    center_loss_T, size_cls_loss_T, sem_cls_loss_T = compute_center_and_sem_cls_loss(end_points_T, config)
    end_points_T['center_loss'] = center_loss_T
    end_points_T['size_cls_loss'] = size_cls_loss_T
    end_points_T['sem_cls_loss'] = sem_cls_loss_T
    box_loss_T = center_loss_T + 0.1*size_cls_loss_T

    box_loss = box_loss_S + box_loss_T
    sem_cls_loss = cls_coefficient*sem_cls_loss_S + sem_cls_loss_T

    ## Domain Align Loss
    FL_global = FocalLoss(class_num=2, gamma=3)

    da_coefficient = 0.5
    
    # Source domain
    global_d_pred_S = end_points_S['global_d_pred']
    local_d_pred_S = end_points_S['local_d_pred'].transpose(1,2).contiguous()
    domain_S = Variable(torch.zeros(global_d_pred_S.size(0)).long().cuda())
    #object_weight_local_S = F.softmax(end_points_S['objectness_scores'], dim=-1)[:,:,1:]
    object_weight_local_S = end_points_S['objectness_label'].unsqueeze(-1)
    source_dloss = da_coefficient * torch.mean(local_d_pred_S**2 * object_weight_local_S) + da_coefficient * FL_global(global_d_pred_S, domain_S)
    
    # Target domain
    global_d_pred_T = end_points_T['global_d_pred']
    local_d_pred_T = end_points_T['local_d_pred'].transpose(1,2).contiguous()
    domain_T = Variable(torch.ones(global_d_pred_T.size(0)).long().cuda())
    #object_weight_local_T = F.softmax(end_points_T['objectness_scores'], dim=-1)[:,:,1:]
    object_weight_local_T = end_points_T['objectness_label'].unsqueeze(-1)
    target_dloss = da_coefficient * torch.mean((1-local_d_pred_T)**2 * object_weight_local_T) + da_coefficient * FL_global(global_d_pred_T, domain_T)
    DA_loss = source_dloss + target_dloss

    ## MMD
    if Is_mmd:
        sigma_list = [0.1,1,10]
        mmd_dloss = mmd.mix_rbf_mmd2(end_points_S['shared_local_features'], end_points_T['shared_local_features'], sigma_list)

    ## IoU Loss
    if epoch > 120:
        iou_labels, _, iou_assignment = compute_iou_labels(
            # aggregated_vote_xyz -> center
            end_points_S, end_points_S['aggregated_vote_xyz'], end_points_S['center'], None, None,
            end_points_S['heading_scores'], end_points_S['heading_residuals'], end_points_S['size_scores'],
            end_points_S['size_residuals'], config_dict={'dataset_config': config})
        iou_pred = nn.Sigmoid()(end_points_S['iou_scores'])
        # gt sem cls
        iou_sem_cls_label = torch.gather(end_points_S['sem_cls_label'], 1, iou_assignment)
        iou_pred = torch.gather(iou_pred, 2, iou_sem_cls_label.unsqueeze(-1)).squeeze(-1)  # use pred semantic labels
        iou_loss = huber_loss(iou_pred - iou_labels.detach(), delta=1.0)  # (B, K, 1)
        iou_loss = iou_loss.mean()
        end_points_S['iou_loss'] = iou_loss
    else:
        iou_loss = 0.0

    ## Final loss function
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + DA_loss + iou_loss
    loss *= 10
    end_points_S['loss'] = loss
    if Is_mmd:
        mmd_dloss *= 10
        end_points_S['mmd_loss'] = mmd_dloss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points_S['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label_S.long()).float()*objectness_mask_S)/(torch.sum(objectness_mask_S)+1e-6)
    end_points_S['obj_acc'] = obj_acc

    if Is_mmd:
        return loss, mmd_dloss, end_points_S, end_points_T
    return loss, end_points_S, end_points_T


def get_pseudo_labels(end_points, pred_center, pred_sem_cls, pred_size, pred_objectness, pred_heading_scores,
                      pred_heading_residuals,
                      pred_size_scores, pred_size_residuals, pred_vote_xyz, config_dict):
    batch_size, num_proposal = pred_center.shape[:2]
    label_mask = torch.zeros((batch_size, MAX_NUM_OBJ), dtype=torch.long).cuda()

    # obj score threshold
    pred_objectness = nn.Softmax(dim=2)(pred_objectness)
    # the second element is positive score
    pos_obj = pred_objectness[:, :, 1]
    neg_obj = pred_objectness[:, :, 0]
    neg_objectness_mask = neg_obj > 0.9  # deprecated

    # Position-level annotation guided selection
    gt_center = end_points['center_label'][:,:,0:3]
    dist1, object_assignment, _, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    max_cls, argmax_cls = torch.max(pred_sem_cls, dim=2)
    iou_pred = nn.Sigmoid()(end_points['iou_scores'])
    if iou_pred.shape[2] > 1:
        iou_pred = torch.gather(iou_pred, 2, argmax_cls.unsqueeze(-1)).squeeze(-1)  # use pred semantic labels
    else:
        iou_pred = iou_pred.squeeze(-1)
    pred_sem_cls = nn.Softmax(dim=2)(pred_sem_cls)
    sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    cls_mask = (argmax_cls == sem_cls_label)
    dis_mask = (euclidean_dist1 < config_dict['size_ratio'] * torch.norm(pred_size, p=2, dim=-1))
    objectness_mask = pos_obj > config_dict['obj_threshold']
    iou_mask = iou_pred > config_dict['iou_threshold']
    final_mask = (cls_mask * dis_mask * objectness_mask * iou_mask)
    #print(final_mask.sum(-1))
    #object_assignment[~final_mask] = -1
    #print(object_assignment, end_points['box_label_mask'], end_points['sem_cls_label'])
    #sys.exit(0)

    # we only keep MAX_NUM_OBJ predictions
    # however, after filtering the number can still exceed this
    # so we keep the ones with larger pos_obj * max_cls
    inds = torch.argsort(pos_obj * max_cls * final_mask, dim=1, descending=True)
    inds = inds[:, :MAX_NUM_OBJ].long()
    final_mask_sorted = torch.gather(final_mask, dim=1, index=inds)

    neg_objectness_mask = torch.gather(neg_objectness_mask, dim=1, index=inds)

    max_size, argmax_size = torch.max(pred_size_scores, dim=2)
    size_inds = argmax_size.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 3)
    max_heading, argmax_heading = torch.max(pred_heading_scores, dim=2)
    heading_inds = argmax_heading.unsqueeze(-1)

    # now only one class residuals
    
    '''
    ### Adjust box size ###
    mean_size_arr = config_dict['dataset_config'].mean_size_arr
    mean_size_arr = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
    pred_size_residuals = 0.95 * pred_size_residuals - 0.05 * mean_size_arr
    #######################
    '''
    
    pred_heading_residuals = torch.gather(pred_heading_residuals, dim=2, index=heading_inds).squeeze(2)
    pred_size_residuals = torch.gather(pred_size_residuals, dim=2, index=size_inds).squeeze(2)

    if config_dict['use_lhs']:
        pred_center_ = torch.gather(pred_center, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
        pred_heading_class_ = torch.gather(argmax_heading, dim=1, index=inds)
        pred_heading_residual_ = torch.gather(pred_heading_residuals, dim=1, index=inds)
        pred_size_class_ = torch.gather(argmax_size, dim=1, index=inds)
        pred_size_residual_ = torch.gather(pred_size_residuals, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
        num_proposal = pred_center_.shape[1]
        bsize = pred_center_.shape[0]
        pred_box_parameters = np.zeros((bsize, num_proposal, 7), dtype=np.float32)
        pred_box_parameters[:, :, 0:3] = pred_center_.detach().cpu().numpy()
        pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3), dtype=np.float32)
        pred_center_upright_camera = flip_axis_to_camera(pred_center_.detach().cpu().numpy())
        for i in range(bsize):
            for j in range(num_proposal):
                heading_angle = config_dict['dataset_config'].class2angle( \
                    pred_heading_class_[i, j].detach().cpu().numpy(),
                    pred_heading_residual_[i, j].detach().cpu().numpy())
                box_size = config_dict['dataset_config'].class2size( \
                    int(pred_size_class_[i, j].detach().cpu().numpy()),
                    pred_size_residual_[i, j].detach().cpu().numpy())
                pred_box_parameters[i, j, 3:6] = box_size
                pred_box_parameters[i, j, 6] = heading_angle
                corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i, j, :])
                pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

        # pred_corners_3d_upright_camera, _ = predictions2corners3d(end_points, config_dict)
        pred_mask = np.ones((batch_size, MAX_NUM_OBJ))
        nonempty_box_mask = np.ones((batch_size, MAX_NUM_OBJ))
        pos_obj_numpy = torch.gather(pos_obj, dim=1, index=inds).detach().cpu().numpy()
        iou_numpy = torch.gather(iou_pred, dim=1, index=inds).detach().cpu().numpy()
        pred_sem_cls_numpy = torch.gather(argmax_cls, dim=1, index=inds).detach().cpu().numpy()
        for i in range(batch_size):
            boxes_3d_with_prob = np.zeros((MAX_NUM_OBJ, 8))
            for j in range(MAX_NUM_OBJ):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = pos_obj_numpy[i, j] * iou_numpy[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls_numpy[i, j]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]

            # here we do not consider orientation, in accordance to test time nms
            pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                         config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 0
        # end_points['pred_mask'] = pred_mask
        final_mask_sorted[torch.from_numpy(pred_mask).bool().cuda()] = 0

    label_mask[final_mask_sorted] = 1
    heading_label = torch.gather(argmax_heading, dim=1, index=inds)
    heading_residual_label = torch.gather(pred_heading_residuals.squeeze(-1), dim=1, index=inds)
    size_label = torch.gather(argmax_size, dim=1, index=inds)
    size_residual_label = torch.gather(pred_size_residuals, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
    sem_cls_label = torch.gather(argmax_cls, dim=1, index=inds)
    center_label = torch.gather(pred_center, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
    center_label[(1 - label_mask).unsqueeze(-1).expand(-1, -1, 3).bool()] = -1000
    false_center_label = torch.gather(pred_vote_xyz, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
    false_center_label[torch.logical_not(neg_objectness_mask).unsqueeze(-1).expand(-1, -1, 3).bool()] = -1000
    object_assignment_label = torch.gather(object_assignment, dim=1, index=inds)

    return label_mask, center_label, sem_cls_label, heading_label, heading_residual_label, size_label, size_residual_label, false_center_label, object_assignment_label


def get_pseudo_detection_loss(end_points, config, config_dict, include_weak=False):
    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    aggregated_vote_xyz = end_points['aggregated_vote_xyz']
    gt_center = end_points['unlabeled_center_label'][:,:,0:3]
    box_label_mask = end_points['unlabeled_box_label_mask'] # B,K2
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, _, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1
    object_assignment = ind1
    batch_size = object_assignment.shape[0]

    if include_weak:
        '''
        ## Compute vote loss
        num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
        vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
        dist1, _, dist2, _ = nn_distance(vote_xyz, gt_center, l1=True) # dist1: B,num_seed*vote_factor, dist2: B,K2
        dist1 = dist1.view(batch_size, num_seed, -1) # dist1: B,num_seed,vote_factor
        votes_dist, _ = torch.min(dist1, dim=2) # (B,num_seed,vote_factor) to (B,num_seed,)
        vote_loss = torch.mean(votes_dist) + torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
        end_points['unlabeled_vote_loss'] = vote_loss

        ## Compute objectness loss
        objectness_scores = end_points['objectness_scores']
        criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
        objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
        objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)
        end_points['unlabeled_objectness_loss'] = objectness_loss
        '''

        ## Compute center loss
        pred_center = end_points['center']
        dist1, _, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
        centroid_reg_loss1 = \
            torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
        centroid_reg_loss2 = \
            torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
        center_loss = centroid_reg_loss1 + centroid_reg_loss2
        end_points['unlabeled_center_loss'] = center_loss
    

    # Compute heading class loss
    heading_class_label = torch.gather(end_points['unlabeled_heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(end_points['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)
    end_points['unlabeled_heading_class_loss'] = heading_class_loss

    # Compute heading residual loss
    heading_residual_label = torch.gather(end_points['unlabeled_heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(end_points['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)
    end_points['unlabeled_heading_residual_loss'] = heading_residual_normalized_loss

    if include_weak:
        ## Compute size class loss
        size_class_label = torch.gather(end_points['unlabeled_size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
        criterion_size_class = nn.CrossEntropyLoss(reduction='none')
        size_class_loss = criterion_size_class(end_points['size_scores'].transpose(2,1), size_class_label) # (B,K)
        size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)
        end_points['unlabeled_size_class_loss'] = size_class_loss
    

    # Compute size residual loss
    size_class_label = torch.gather(end_points['unlabeled_size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    size_residual_label = torch.gather(end_points['unlabeled_size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized']*size_label_one_hot_tiled, 2) # (B,K,3)
    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)
    end_points['unlabeled_size_residual_loss'] = size_residual_normalized_loss

    if include_weak:
        ## Compute semantic cls loss
        sem_cls_label = torch.gather(end_points['unlabeled_sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
        criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
        sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
        sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)
        end_points['unlabeled_sem_cls_loss'] = sem_cls_loss
    

    if include_weak:
        box_loss = center_loss + 0.1*heading_class_loss + heading_residual_normalized_loss + 0.1*size_class_loss + size_residual_normalized_loss
    else:
        box_loss = 0.1*heading_class_loss + heading_residual_normalized_loss + size_residual_normalized_loss
    end_points['unlabeled_box_loss'] = box_loss

    if include_weak:
        loss = box_loss + 0.1*sem_cls_loss
    else:
        loss = box_loss
    loss *= 10
    end_points['unlabeled_detection_loss'] = loss

    return loss, end_points


def get_loss_ST(end_points, ema_end_points, config, config_dict, include_weak=False):
    labeled_num = torch.nonzero(end_points['supervised_mask']).squeeze(1).shape[0]
    pred_center = ema_end_points['center'][labeled_num:]
    pred_sem_cls = ema_end_points['sem_cls_scores'][labeled_num:]
    pred_size = ema_end_points['pred_size'][labeled_num:]
    pred_objectness = ema_end_points['objectness_scores'][labeled_num:]
    pred_heading_scores = ema_end_points['heading_scores'][labeled_num:]
    pred_heading_residuals = ema_end_points['heading_residuals'][labeled_num:]
    pred_size_scores = ema_end_points['size_scores'][labeled_num:]
    pred_size_residuals = ema_end_points['size_residuals'][labeled_num:]
    pred_vote_xyz = ema_end_points['aggregated_vote_xyz'][labeled_num:]

    # generate pseudo labels
    label_mask, center_label, sem_cls_label, \
    heading_label, heading_residual_label, \
    size_label, size_residual_label, false_center_label, object_assignment_label = \
        get_pseudo_labels(end_points, pred_center, pred_sem_cls, pred_size, pred_objectness, pred_heading_scores,
                          pred_heading_residuals, pred_size_scores, pred_size_residuals, pred_vote_xyz, config_dict)

    end_points['unlabeled_center_label'] = center_label
    end_points['unlabeled_box_label_mask'] = label_mask
    end_points['unlabeled_sem_cls_label'] = sem_cls_label
    end_points['unlabeled_heading_class_label'] = heading_label
    end_points['unlabeled_heading_residual_label'] = heading_residual_label
    end_points['unlabeled_size_class_label'] = size_label
    end_points['unlabeled_size_residual_label'] = size_residual_label
    end_points['unlabeled_false_center_label'] = false_center_label
    end_points['unlabeled_object_assignment_label'] = object_assignment_label

    consistency_loss, end_points = get_pseudo_detection_loss(end_points, config, config_dict, include_weak=include_weak)

    return consistency_loss, end_points, ema_end_points


def get_loss_ST_for_virtual(end_points, ema_end_points, config, config_dict):
    labeled_num = torch.nonzero(ema_end_points['supervised_mask']).squeeze(1).shape[0]
    pred_center = ema_end_points['center'][labeled_num:]
    pred_sem_cls = ema_end_points['sem_cls_scores'][labeled_num:]
    pred_size = ema_end_points['pred_size'][labeled_num:]
    pred_objectness = ema_end_points['objectness_scores'][labeled_num:]
    pred_heading_scores = ema_end_points['heading_scores'][labeled_num:]
    pred_heading_residuals = ema_end_points['heading_residuals'][labeled_num:]
    pred_size_scores = ema_end_points['size_scores'][labeled_num:]
    pred_size_residuals = ema_end_points['size_residuals'][labeled_num:]
    pred_vote_xyz = ema_end_points['aggregated_vote_xyz'][labeled_num:]

    # generate pseudo labels
    label_mask, center_label, sem_cls_label, \
    heading_label, heading_residual_label, \
    size_label, size_residual_label, false_center_label, object_assignment_label = \
        get_pseudo_labels(ema_end_points, pred_center, pred_sem_cls, pred_size, pred_objectness, pred_heading_scores,
                          pred_heading_residuals, pred_size_scores, pred_size_residuals, pred_vote_xyz, config_dict)

    ######### ST for virtual #########
    virtual_centers = end_points['virtual_centers']
    virtual_size_residuals_normalized = end_points['virtual_size_residuals_normalized']
    virtual_angle = end_points['virtual_angle']
    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr
    batch_size = label_mask.shape[0]


    ## Compute center loss
    dist1, _, _, _ = nn_distance(center_label, virtual_centers)
    centroid_reg_loss1 = \
        torch.sum(dist1*label_mask)/(torch.sum(label_mask)+1e-6)
    center_loss = centroid_reg_loss1
    end_points['unlabeled_center_loss'] = center_loss

    # Compute angle loss
    virtual_angle = torch.gather(virtual_angle, 1, object_assignment_label) % np.pi
    virtual_angle_normalized = virtual_angle / np.pi
    angle_per_class = 2*np.pi/float(num_heading_bin)
    angle_center = heading_label * angle_per_class
    angle_label = (angle_center + heading_residual_label) % np.pi
    angle_label_normalized = angle_label / np.pi
    angle_normalized_loss = huber_loss(angle_label_normalized - virtual_angle_normalized, delta=1.0)
    angle_normalized_loss = torch.sum(angle_normalized_loss*label_mask)/(torch.sum(label_mask)+1e-6)
    end_points['unlabeled_angle_loss'] = angle_normalized_loss

    # Compute size residual loss
    virtual_size_residuals_normalized = torch.gather(virtual_size_residuals_normalized, 1, object_assignment_label.unsqueeze(-1).repeat(1,1,3))
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_label.unsqueeze(-1), 1)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3)
    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)
    size_residual_label_normalized = size_residual_label / mean_size_label
    size_residual_normalized_loss = torch.mean(huber_loss(size_residual_label_normalized - virtual_size_residuals_normalized, delta=1.0), -1)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*label_mask)/(torch.sum(label_mask)+1e-6)
    end_points['unlabeled_size_residual_loss'] = size_residual_normalized_loss

    consistency_loss = center_loss + size_residual_normalized_loss

    return consistency_loss, end_points, ema_end_points