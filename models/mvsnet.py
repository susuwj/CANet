import torch
import torch.nn as nn
import torch.nn.functional as F

from models.submodules import homo_warping, init_inverse_range, schedule_inverse_range, FeatExt3_ref, Refinement, RegVis, Reg2d_GCA, UNet_A

class MVSNet(nn.Module):
    def __init__(self, levels, hypo_plane_num_stages, depth_interval_ratio_stages, 
                    feat_base_channel, reg_base_channel, group_cor_dim_stages, qkv_dim_stages, heads_stages):
        super(MVSNet, self).__init__()
        
        self.levels = levels
        self.hypo_plane_num_stages = hypo_plane_num_stages
        self.depth_interval_ratio_stages = depth_interval_ratio_stages
        self.qkv_dim_stages = qkv_dim_stages
        self.heads_stages = heads_stages

        # feature settings
        self.FeatureNet = UNet_A(base_channels=feat_base_channel)
        self.context_feature = FeatExt3_ref()
        self.context_feature_global = UNet_A(base_channels=feat_base_channel)

        # cost regularization settings
        self.Reg_stages = nn.ModuleList()
        self.group_cor_dim_stages = group_cor_dim_stages
        self.att_feat_channels = [64, 64, 32, 32, 16]
        for stage_idx in range(self.levels):
            in_dim = group_cor_dim_stages[stage_idx]
            self.Reg_stages.append(StageNet(in_dim, reg_base_channel, in_dim, feat_dim=self.att_feat_channels[stage_idx], qkv_dim=self.qkv_dim_stages[stage_idx], heads=self.heads_stages[stage_idx], depth_num=self.hypo_plane_num_stages[stage_idx], cor_weight_est=(stage_idx==0)))

        self.context_feat_channel = [0, 32, 0, 16, 0] 
        self.residualnet = nn.ModuleList([ResidualNet(self.context_feat_channel[i]) for i in range(self.levels)])

    def forward(self, imgs, proj_matrices, depth_values, filename=None):

        context_feats = self.context_feature(imgs[0])
        context_feats_global = self.context_feature_global(imgs[0])
        features = []
        for nview_idx in range(len(imgs)):
            img = imgs[nview_idx]
            features.append(self.FeatureNet(img))   # B C H W
        
        cor_weights = None
        # coarse-to-fine
        outputs = {}
        for stage_idx in range(self.levels):
            if stage_idx<(self.levels-1):
               idx = stage_idx//2
            else:
               idx = stage_idx - 2 
            stage_name = "stage{}".format(idx + 1)
            B, C, H, W = features[0][stage_name].shape
            proj_matrices_stage = proj_matrices[stage_name]
            features_stage = [feat[stage_name] for feat in features]
            global_feat = context_feats_global["stage{}".format(str(idx+1))]

            if self.context_feat_channel[stage_idx] != 0 :
               context_feat = context_feats["stage{}".format(str(idx+1))]
            else:
               context_feat = None

            # @Note depth hypos
            if stage_idx == 0:
                depth_hypo = init_inverse_range(depth_values, self.hypo_plane_num_stages[stage_idx], img[0].device, img[0].dtype, H, W)
                last_depth_itv = (1./depth_hypo[:,2,0,0] - 1./depth_hypo[:,1,0,0]).view(-1, 1, 1)
            else:
                depth = outputs_stage['depth'].detach().squeeze(1)
                cor_weights = F.interpolate(cor_weights, [H, W], mode='bilinear', align_corners=True)
                last_depth_itv = self.depth_interval_ratio_stages[stage_idx-1] * last_depth_itv
                inverse_min_depth = 1/depth + last_depth_itv  # B H W
                inverse_max_depth = 1/depth - last_depth_itv  # B H W
                depth_hypo = schedule_inverse_range(inverse_min_depth, inverse_max_depth, self.hypo_plane_num_stages[stage_idx], H, W)  # B D H W
  
            outputs_stage = self.Reg_stages[stage_idx](
            features_stage, proj_matrices_stage, global_feat, depth_hypo=depth_hypo, cor_weights=cor_weights)

            if stage_idx != self.levels - 1:
                depth = torch.zeros_like(outputs_stage['depth'])
                for i in range(B):
                   depth[i] = torch.clamp(outputs_stage['depth'][i], min=depth_values[i,0].cpu().item(), max=depth_values[i,-1].cpu().item())
                outputs_stage['depth'] = depth

            if self.context_feat_channel[stage_idx] != 0:
               outputs_stage = self.residualnet[stage_idx](outputs_stage, context_feat)

            if stage_idx == 0:
                cor_weights = outputs_stage['cor_weights']
                #print(cor_weights.shape, len(imgs))

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs

class ResidualNet(nn.Module):
    def __init__(self, residual):
        super(ResidualNet, self).__init__()
        if residual != 0:
           self.residual_net = Refinement(residual)

    def forward(self, inputs, feat):

        depth_init_res = inputs["depth"]
        inputs["depth_init_res"] = inputs["depth"]
        res, depth = self.residual_net(depth_init_res.detach(), feat)
        inputs["depth"] = depth
        inputs["res"] = res

        return inputs

class StageNet(nn.Module):
    def __init__(self, in_dim, reg_base_channel, group_cor_dim, feat_dim, qkv_dim, heads, depth_num, cor_weight_est=False):
        super(StageNet, self).__init__()
        self.group_cor_dim = group_cor_dim
        self.regnet = Reg2d_GCA(input_channel=in_dim, base_channel=reg_base_channel, feat_dim=feat_dim, qkv_dim=qkv_dim, heads=heads)
        self.cor_weight_est = cor_weight_est
        if self.cor_weight_est:
            self.reg_vis = RegVis()

    def forward(self, features, proj_matrices, global_feat, depth_hypo, cor_weights=None):

        # @Note step1: feature extraction
        proj_matrices = torch.unbind(proj_matrices, 1)
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        B, D, H, W = depth_hypo.shape
        C = ref_feature.shape[1]

        if self.cor_weight_est:
            cor_weights = []

        # @Note step2: cost aggregation
        ref_volume = ref_feature.unsqueeze(2).reshape(B, self.group_cor_dim, C//self.group_cor_dim, 1, H, W)
        cor_weight_sum = 1e-8
        cor_feats = 0
        for src_idx, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_src = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_hypo)  # B C D H W

            warped_src = warped_src.reshape(B, self.group_cor_dim, C//self.group_cor_dim, D, H, W)
            cor_feat = (warped_src * ref_volume).mean(2)  # B G D H W
            del warped_src, src_proj, src_fea

            if self.cor_weight_est:
                cor_weight = self.reg_vis(cor_feat)
                cor_weight = torch.sigmoid(cor_weight.squeeze(1))
                cor_weight, _ = torch.max(cor_weight, dim=1, keepdim=True)
                cor_weights.append(cor_weight)
                cor_weight_sum += cor_weight  # B D H W
            else:
                cor_weight = cor_weights[:, src_idx].unsqueeze(1)
                cor_weight_sum += cor_weight  # B D H W

            cor_feats += cor_weight.unsqueeze(1) * cor_feat  # B C D H W
            del cor_weight, cor_feat

        cost_volume = cor_feats / cor_weight_sum.unsqueeze(1)  # B C D H W
        del cor_weight_sum, src_features, cor_feats

        # @Note step3: cost regularization
        cost_int = None
        cost_reg, cost_int = self.regnet(cost_volume, global_feat)
        del cost_volume
        prob_volume = F.softmax(cost_reg, dim=1)  # B D H W

        # @Note step4: depth regression
        prob_max_indices = prob_volume.max(1, keepdim=True)[1]  # B 1 H W
        depth = torch.gather(depth_hypo, 1, prob_max_indices).squeeze(1)  # B H W

        with torch.no_grad():
            photometric_confidence = prob_volume.max(1)[0]  # B H W
            #photometric_confidence = F.interpolate(photometric_confidence.unsqueeze(1), scale_factor=1, mode='bilinear', align_corners=True).squeeze(1)
        
        output_stage = {
            "depth": depth,  
            "photometric_confidence": photometric_confidence, 
            "depth_hypo": depth_hypo, 
            "prob_volume": prob_volume,
        }

        if cost_int is not None:
            prob_volume = F.softmax(cost_int, dim=1)  # B D H W
            prob_max_indices = prob_volume.max(1, keepdim=True)[1]  # B 1 H W
            depth = torch.gather(depth_hypo, 1, prob_max_indices).squeeze(1)  # B H W
            output_stage["depth_int"] =  depth
            output_stage["prob_int"] =  prob_volume
        if self.cor_weight_est:
            output_stage["cor_weights"] = torch.cat(cor_weights, dim=1).detach()
        return output_stage
