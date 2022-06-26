import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from .grid_conv_module import GridConv
from .backbone_module import Pointnet2Backbone, Pointnet2Backbone_jitter
from .transformer import TransformerDecoderLayer
from .modules import PointsObjClsModule, FPSModule, GeneralSamplingModule, PositionEmbeddingLearned, PredictHead
from torch.autograd import Variable, Function
from pointnet2_utils import furthest_point_sample, gather_operation


class GradReverse(Function):
    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        #pdb.set_trace()
        return (grad_output * -1.0)


def grad_reverse(x):
    return GradReverse.apply(x)


class Attention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.convQK = nn.Conv1d(channel, channel // 2, 1)
        self.convV = nn.Conv1d(channel, channel // 2, 1)
        self.convFinal = nn.Conv1d(channel // 2, channel, 1)

    def forward(self, features):
        Q = self.convQK(features).permute(0,2,1) # BxNxC
        K = self.convQK(features) # BxCxN
        V = self.convV(features).permute(0,2,1) # BxNxC
        att = torch.matmul(Q, K)
        att = F.softmax(att, dim=-1)
        feats = torch.matmul(att, V).permute(0,2,1).contiguous() # NxCxN
        feats = self.convFinal(feats)
        return feats + features


class GroupFreeDetector_DA(nn.Module):
    r"""
        A Group-Free detector for 3D object detection via Transformer.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        width: (default: 1)
            PointNet backbone width ratio
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        sampling: (default: kps)
            Initial object candidate sampling method
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, width=1, bn_momentum=0.1, sync_bn=False, num_proposal=128, sampling='kps',
                 dropout=0.1, activation="relu", nhead=8, num_decoder_layers=6, dim_feedforward=2048,
                 self_position_embedding='xyz_learned', cross_position_embedding='xyz_learned', dataset_config=None):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.bn_momentum = bn_momentum
        self.sync_bn = sync_bn
        self.width = width
        self.nhead = nhead
        self.sampling = sampling
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.self_position_embedding = self_position_embedding
        self.cross_position_embedding = cross_position_embedding
        if dataset_config:
            self.dataset_config = dataset_config

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim, width=self.width)

        if self.sampling == 'fps':
            self.fps_module = FPSModule(num_proposal)
        elif self.sampling == 'kps':
            self.points_obj_cls = PointsObjClsModule(288)
            self.gsample_module = GeneralSamplingModule()
        else:
            raise NotImplementedError
        # Proposal
        self.proposal_head = PredictHead(num_class, num_heading_bin, num_size_cluster,
                                         mean_size_arr, num_proposal, 288)
        if self.num_decoder_layers <= 0:
            # stop building if has no decoder layer
            return

        # Transformer Decoder Projection
        self.decoder_key_proj = nn.Conv1d(288, 288, kernel_size=1)
        self.decoder_query_proj = nn.Conv1d(288, 288, kernel_size=1)

        # Position Embedding for Self-Attention
        if self.self_position_embedding == 'none':
            self.decoder_self_posembeds = [None for i in range(num_decoder_layers)]
        elif self.self_position_embedding == 'xyz_learned':
            self.decoder_self_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(3, 288))
        elif self.self_position_embedding == 'loc_learned':
            self.decoder_self_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(6, 288))
        else:
            raise NotImplementedError(f"self_position_embedding not supported {self.self_position_embedding}")

        # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            self.decoder_cross_posembeds = [None for i in range(num_decoder_layers)]
        elif self.cross_position_embedding == 'xyz_learned':
            self.decoder_cross_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_cross_posembeds.append(PositionEmbeddingLearned(3, 288))
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")

        # Transformer decoder layers
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    288, nhead, dim_feedforward, dropout, activation,
                    self_posembed=self.decoder_self_posembeds[i],
                    cross_posembed=self.decoder_cross_posembeds[i],
                ))

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.prediction_heads.append(PredictHead(num_class, num_heading_bin, num_size_cluster,
                                                     mean_size_arr, num_proposal, 288))

        # Init
        self.init_weights()
        self.init_bn_momentum()
        if self.sync_bn:
            nn.SyncBatchNorm.convert_sync_batchnorm(self)

        # global domain prediction
        self.global_netD1 = nn.Sequential(
            nn.Conv1d(288,256,1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU()
            )
        self.global_netD2 = nn.Linear(128, 2)

        # local domain prediction
        self.decoder_netD = nn.Sequential(
            nn.Conv1d(288,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,1,1)
            )
        
        # IoU branch
        self.grid_conv = GridConv(
            num_class, num_heading_bin, num_size_cluster,
            mean_size_arr, num_proposal, sampling, query_feats='seed',
        )

        # attention module, map the local features to the shared space
        self.attention = Attention(288)
    
    def calculate_bbox(self, end_points, prefix):
        # calculate size and center
        size_scores = end_points[f'{prefix}size_scores']
        size_residuals = end_points[f'{prefix}size_residuals']
        B, K = size_scores.shape[:2]
        mean_size_arr = self.mean_size_arr
        mean_size_arr = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda()  # (num_size_cluster,3)
        size_class = torch.argmax(size_scores, -1)  # B,num_proposal
        size_residual = torch.gather(size_residuals, 2,
                                     size_class.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 3))  # B,num_proposal,1,3
        size_residual = size_residual.squeeze(2)
        size_base = torch.index_select(mean_size_arr, 0, size_class.view(-1))
        size_base = size_base.view(B, K, 3)
        size = (size_base + size_residual) / 2  # half of the size
        size[size < 0] = 1e-6
        center = end_points[f'{prefix}center']

        heading_scores = end_points[f'{prefix}heading_scores']
        heading_class = torch.argmax(heading_scores, -1)  # B,num_proposal
        heading_residuals = end_points[f'{prefix}heading_residuals']
        heading_residual = torch.gather(heading_residuals, 2, heading_class.unsqueeze(-1))  # B,num_proposal,1
        heading_residual = heading_residual.squeeze(2)
        heading = self.dataset_config.class2angle_gpu(heading_class, heading_residual)

        end_points[f'{prefix}size'] = size
        end_points[f'{prefix}heading'] = heading
        return center, size, heading

    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]

        end_points = self.backbone_net(inputs['point_clouds'], end_points)

        # Query Points Generation
        points_xyz = end_points['fp2_xyz']
        points_features = end_points['fp2_features']
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        if self.sampling == 'fps':
            xyz, features, sample_inds = self.fps_module(xyz, features)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        elif self.sampling == 'kps':
            points_obj_cls_logits = self.points_obj_cls(features)  # (batch_size, 1, num_seed)
            end_points['seeds_obj_cls_logits'] = points_obj_cls_logits
            points_obj_cls_scores = torch.sigmoid(points_obj_cls_logits).squeeze(1)
            sample_inds = torch.topk(points_obj_cls_scores, self.num_proposal)[1].int()
            xyz, features, sample_inds = self.gsample_module(xyz, features, sample_inds)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        else:
            raise NotImplementedError

        # Proposal
        proposal_center, proposal_size = self.proposal_head(cluster_feature,
                                                            base_xyz=cluster_xyz,
                                                            end_points=end_points,
                                                            prefix='proposal_')  # N num_proposal 3

        base_xyz = proposal_center.detach().clone()
        base_size = proposal_size.detach().clone()

        # Transformer Decoder and Prediction
        if self.num_decoder_layers > 0:
            query = self.decoder_query_proj(cluster_feature)
            key = self.decoder_key_proj(points_features) if self.decoder_key_proj is not None else None
        # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            key_pos = None
        elif self.cross_position_embedding in ['xyz_learned']:
            key_pos = points_xyz
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")

        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'

            # Position Embedding for Self-Attention
            if self.self_position_embedding == 'none':
                query_pos = None
            elif self.self_position_embedding == 'xyz_learned':
                query_pos = base_xyz
            elif self.self_position_embedding == 'loc_learned':
                query_pos = torch.cat([base_xyz, base_size], -1)
            else:
                raise NotImplementedError(f"self_position_embedding not supported {self.self_position_embedding}")

            # Transformer Decoder Layer
            query = self.decoder[i](query, key, query_pos, key_pos)

            # Prediction
            base_xyz, base_size = self.prediction_heads[i](query,
                                                           base_xyz=cluster_xyz,
                                                           end_points=end_points,
                                                           prefix=prefix)

            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()

            # ## IoU Prediction for Virtual Scenes
            # center, size, heading = self.calculate_bbox(end_points, prefix)
            # end_points = self.grid_conv(center.detach(), size.detach(), heading.detach(), end_points, prefix)

            if prefix == 'last_':
                ## IoU Prediction for Virtual Scenes
                center, size, heading = self.calculate_bbox(end_points, prefix)
                end_points = self.grid_conv(center.detach(), size.detach(), heading.detach(), end_points, prefix)
                ## domain prediction
                local_d_pred = self.decoder_netD(grad_reverse(query)) # Bx1x256
                local_d_pred = torch.sigmoid(local_d_pred)
                end_points[f'{prefix}local_d_pred'] = local_d_pred
                ## shared space feature
                local_features = gather_operation(query.contiguous(), furthest_point_sample(base_xyz, 64)) # Bx288x256-->Bx288x64
                local_features = self.attention(local_features)
                local_features = local_features.transpose(1, 2).contiguous().view(-1, 128) # Bx288x64-->Bx64x288-->-1x288
                end_points['shared_local_features'] = local_features
            else:
                pass

        ## domain prediction
        # global
        global_d_pred = self.global_netD1(grad_reverse(end_points['seed_features'])) # Bx128x1024
        global_d_pred = torch.mean(global_d_pred, dim=2) # Bx128
        global_d_pred = self.global_netD2(global_d_pred) # Bx2
        end_points['global_d_pred'] = global_d_pred

        return end_points

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum


class GroupFreeDetector_DA_jitter(nn.Module):
    r"""
        A Group-Free detector for 3D object detection via Transformer.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        width: (default: 1)
            PointNet backbone width ratio
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        sampling: (default: kps)
            Initial object candidate sampling method
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, width=1, bn_momentum=0.1, sync_bn=False, num_proposal=128, sampling='kps',
                 dropout=0.1, activation="relu", nhead=8, num_decoder_layers=6, dim_feedforward=2048,
                 self_position_embedding='xyz_learned', cross_position_embedding='xyz_learned'
                 ):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.bn_momentum = bn_momentum
        self.sync_bn = sync_bn
        self.width = width
        self.nhead = nhead
        self.sampling = sampling
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.self_position_embedding = self_position_embedding
        self.cross_position_embedding = cross_position_embedding

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone_jitter(input_feature_dim=self.input_feature_dim, width=self.width)

        if self.sampling == 'fps':
            self.fps_module = FPSModule(num_proposal)
        elif self.sampling == 'kps':
            self.points_obj_cls = PointsObjClsModule(288)
            self.gsample_module = GeneralSamplingModule()
        else:
            raise NotImplementedError
        # Proposal
        self.proposal_head = PredictHead(num_class, num_heading_bin, num_size_cluster,
                                         mean_size_arr, num_proposal, 288)
        if self.num_decoder_layers <= 0:
            # stop building if has no decoder layer
            return

        # Transformer Decoder Projection
        self.decoder_key_proj = nn.Conv1d(288, 288, kernel_size=1)
        self.decoder_query_proj = nn.Conv1d(288, 288, kernel_size=1)

        # Position Embedding for Self-Attention
        if self.self_position_embedding == 'none':
            self.decoder_self_posembeds = [None for i in range(num_decoder_layers)]
        elif self.self_position_embedding == 'xyz_learned':
            self.decoder_self_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(3, 288))
        elif self.self_position_embedding == 'loc_learned':
            self.decoder_self_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(6, 288))
        else:
            raise NotImplementedError(f"self_position_embedding not supported {self.self_position_embedding}")

        # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            self.decoder_cross_posembeds = [None for i in range(num_decoder_layers)]
        elif self.cross_position_embedding == 'xyz_learned':
            self.decoder_cross_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_cross_posembeds.append(PositionEmbeddingLearned(3, 288))
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")

        # Transformer decoder layers
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    288, nhead, dim_feedforward, dropout, activation,
                    self_posembed=self.decoder_self_posembeds[i],
                    cross_posembed=self.decoder_cross_posembeds[i],
                ))

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.prediction_heads.append(PredictHead(num_class, num_heading_bin, num_size_cluster,
                                                     mean_size_arr, num_proposal, 288))

        # Init
        self.init_weights()
        self.init_bn_momentum()
        if self.sync_bn:
            nn.SyncBatchNorm.convert_sync_batchnorm(self)

        # global domain prediction
        self.global_netD1 = nn.Sequential(
            nn.Conv1d(288,256,1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU()
            )
        self.global_netD2 = nn.Linear(128, 2)

        
        # local domain prediction
        self.decoder_netD = nn.Sequential(
            nn.Conv1d(288,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,1,1)
            )

        # jitter prediction
        self.jitter_net = nn.Sequential(
            nn.Conv1d(150,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,3,1)
            )
        

    def forward(self, inputs, center_xyz=None, center_cls=None):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]

        end_points = self.backbone_net(inputs['point_clouds'], center_xyz, center_cls, end_points)
        if center_xyz is not None:
            end_points['jitter_pred'] = self.jitter_net(end_points['center_features']) # B, 3, 64

        # Query Points Generation
        points_xyz = end_points['fp2_xyz']
        points_features = end_points['fp2_features']
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        if self.sampling == 'fps':
            xyz, features, sample_inds = self.fps_module(xyz, features)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        elif self.sampling == 'kps':
            points_obj_cls_logits = self.points_obj_cls(features)  # (batch_size, 1, num_seed)
            end_points['seeds_obj_cls_logits'] = points_obj_cls_logits
            points_obj_cls_scores = torch.sigmoid(points_obj_cls_logits).squeeze(1)
            sample_inds = torch.topk(points_obj_cls_scores, self.num_proposal)[1].int()
            xyz, features, sample_inds = self.gsample_module(xyz, features, sample_inds)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        else:
            raise NotImplementedError

        # Proposal
        proposal_center, proposal_size = self.proposal_head(cluster_feature,
                                                            base_xyz=cluster_xyz,
                                                            end_points=end_points,
                                                            prefix='proposal_')  # N num_proposal 3

        base_xyz = proposal_center.detach().clone()
        base_size = proposal_size.detach().clone()

        # Transformer Decoder and Prediction
        if self.num_decoder_layers > 0:
            query = self.decoder_query_proj(cluster_feature)
            key = self.decoder_key_proj(points_features) if self.decoder_key_proj is not None else None
        # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            key_pos = None
        elif self.cross_position_embedding in ['xyz_learned']:
            key_pos = points_xyz
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")

        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'

            # Position Embedding for Self-Attention
            if self.self_position_embedding == 'none':
                query_pos = None
            elif self.self_position_embedding == 'xyz_learned':
                query_pos = base_xyz
            elif self.self_position_embedding == 'loc_learned':
                query_pos = torch.cat([base_xyz, base_size], -1)
            else:
                raise NotImplementedError(f"self_position_embedding not supported {self.self_position_embedding}")

            # Transformer Decoder Layer
            query = self.decoder[i](query, key, query_pos, key_pos)

            # local
            if prefix == 'last_':
                local_d_pred = self.decoder_netD(grad_reverse(query)) # Bx1x256
                local_d_pred = torch.sigmoid(local_d_pred)
                end_points[f'{prefix}local_d_pred'] = local_d_pred
            else:
                pass

            # Prediction
            base_xyz, base_size = self.prediction_heads[i](query,
                                                           base_xyz=cluster_xyz,
                                                           end_points=end_points,
                                                           prefix=prefix)

            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()


        ## domain prediction
        # global
        global_d_pred = self.global_netD1(grad_reverse(end_points['seed_features'])) # Bx128x1024
        global_d_pred = torch.mean(global_d_pred, dim=2) # Bx128
        global_d_pred = self.global_netD2(global_d_pred) # Bx2
        end_points['global_d_pred'] = global_d_pred

        return end_points

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum
