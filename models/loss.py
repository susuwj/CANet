import torch
import torch.nn.functional as F

def mvsnet_loss(inputs, depth_gt_ms, mask_ms, imgs, proj_matrices_ms, l1, **kwargs):

    stage_lw = kwargs.get("stage_lw")
    depth_values = kwargs.get("depth_values")
    depth_min, depth_max = depth_values[:,0], depth_values[:,-1]
    depth_interval = (depth_max - depth_min)/128
    
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    pw_loss_stages = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        if stage_idx<4:
           idx = stage_idx//2
        else:
           idx = stage_idx - 2 
        stage_key = "stage{}".format(idx + 1)
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key] > 0.5

        prob_volume = stage_inputs['prob_volume']
        depth_value = stage_inputs['depth_hypo']
        pw_loss = pixel_wise_loss(prob_volume, depth_gt, mask, depth_value)

        pw_loss_stages.append(pw_loss)


        if "depth_int" in stage_inputs:
            int_loss = pixel_wise_loss(stage_inputs["prob_int"], depth_gt, mask, depth_value)
            total_loss = total_loss + stage_lw[stage_idx] * int_loss

        if (stage_idx == 2) or (stage_idx == 4):
           depth_est = inputs["stage{}".format(str(stage_idx))]["depth"]
           res_loss = depth_loss(depth_gt, depth_est, mask, depth_interval, l1)
           total_loss = total_loss + stage_lw[stage_idx] * res_loss
        
        # total loss
        total_loss = total_loss + stage_lw[stage_idx] * pw_loss

    depth_pred = stage_inputs['depth']
    depth_gt = depth_gt_ms[stage_key]
    epe = cal_metrics(depth_pred, depth_gt, mask, depth_min, depth_max)
    
    return total_loss, epe, pw_loss_stages, res_loss, int_loss

def depth_loss(depth_gt, depth_est, mask, depth_interval, l1):
    if l1:
       depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
    else:
        depth_loss = (depth_est - depth_gt).abs()
        depth_loss_scale = depth_loss / depth_interval.unsqueeze(-1).unsqueeze(-1)
        depth_loss = depth_loss_scale[mask].mean()
    return depth_loss

def pixel_wise_loss(prob_volume, depth_gt, mask, depth_value):
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1,2])+1e-12

    shape = depth_gt.shape

    depth_num = depth_value.shape[1]
    depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)

    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume+1e-12), dim=1).squeeze(1)
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image)
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)
    
    pw_loss = masked_cross_entropy
    return pw_loss

def cal_metrics(depth_pred, depth_gt, mask, depth_min, depth_max):
    depth_pred_norm = depth_pred * 128 / (depth_max - depth_min)[:,None,None]
    depth_gt_norm = depth_gt * 128 / (depth_max - depth_min)[:,None,None]

    abs_err = torch.abs(depth_pred_norm[mask] - depth_gt_norm[mask])
    epe = abs_err.mean()
    
    return epe
