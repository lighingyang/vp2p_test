import torch
import torch.nn as nn
import torch.nn.functional as F
# from mdistiller.engine.att_sampler import AttentionSampler


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def average_norm(t):
    return t / t.square().sum(1, keepdim=True).sqrt().mean()


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)

# ----------------------------------------------------------------------------------------------------------------------

def attention_sampling(input_tensor, out_channels, scale_factor, coords: torch.Tensor):
    """
    使用注意力机制对输入张量进行采样。
    
    参数:
    - input_tensor: 输入特征图，维度为[batch_size, channel, h, w]。
    - out_channels: 注意力网络的输出通道数。
    - scale_factor: 上采样的尺度因子。
    
    返回:
    - 输出特征图，维度为[batch_size, channel, n, n]。
    """
    # 注意力网络
    attention_net = torch.nn.Sequential(
        torch.nn.Conv2d(input_tensor.shape[1], out_channels, kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(out_channels, 1, kernel_size=1)
    ).to(input_tensor.device)
    
    # 计算注意力权重
    attn_weights = attention_net(input_tensor)  # [64, 1, 32, 32]
    attn_weights = torch.softmax(attn_weights, dim=-1)
    

    
    weighted_features = input_tensor * attn_weights
    
    # 创建采样网格
    grid = _create_grid(input_tensor.size(2), input_tensor.size(3), scale_factor, input_tensor.size(0)).to(input_tensor.device)
    
    # 使用F.grid_sample进行采样
    # sampled_features = F.grid_sample(weighted_features, grid, padding_mode='zeros', align_corners=False)
    
    # test sampling
    sampled_features = F.grid_sample(weighted_features, coords.permute(0, 2, 1, 3), padding_mode='zeros', align_corners=False)
    # print("attention_sampler")
    
    # 使用F.interpolate进行尺寸调整
    # sampled_features = F.interpolate(sampled_features, scale_factor=scale_factor, mode='bilinear', align_corners=True)
    
    return sampled_features

def _create_grid(h, w, scale_factor, batch_size):
    """
    创建用于F.grid_sample的网格。
    
    参数:
    - h: 输入特征图的高度。
    - w: 输入特征图的宽度。
    - scale_factor: 上采样的尺度因子。
    
    返回:
    - 网格张量，维度为[1, h * scale_factor, w * scale_factor, 2]。
    """
    y, x = torch.meshgrid(torch.linspace(0, h - 1, scale_factor),
                           torch.linspace(0, w - 1, scale_factor))
    grid = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1).float()
    return grid

# ----------------------------------------------------------------------------------------------------------------------
# def mask_sample(t: torch.Tensor, mask: torch.Tensor):
#     mask = mask.float()
#     mask = mask / mask.max()

#     sampled_features = t * mask
#     aggredgated_features = sampled_features.sum(dim=(2, 3))

#     return aggredgated_features
 

def sample_nonzero_locations(t, target_size):
    nonzeros = torch.nonzero(t)
    coords = torch.zeros(target_size, dtype=nonzeros.dtype, device=nonzeros.device)
    n = target_size[1] * target_size[2]
    for i in range(t.shape[0]):
        selected_nonzeros = nonzeros[nonzeros[:, 0] == i]
        if selected_nonzeros.shape[0] == 0:
            selected_coords = torch.randint(t.shape[1], size=(n, 2), device=nonzeros.device)
        else:
            selected_coords = selected_nonzeros[torch.randint(len(selected_nonzeros), size=(n,)), 1:]
        coords[i, :, :, :] = selected_coords.reshape(target_size[1], target_size[2], 2)
    coords = coords.to(torch.float32) / t.shape[1]
    coords = coords * 2 - 1
    return torch.flip(coords, dims=[-1])

def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


def CORR_loss(orig_feats: torch.Tensor, orig_feats_pos: torch.Tensor,orig_code: torch.Tensor, orig_code_pos: torch.Tensor):
        feature_samples = 40
        coord_shape = [orig_feats.shape[0], feature_samples, feature_samples, 2]      # [64, 40, 40, 2]
        coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
        coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
        # print(coords2)
        # print(coords2.shape)
        # while(1):
        #     pass
        feats = sample(orig_feats, coords1)     # orig_feats: [64, 32, 32, 32] -> feats: [64, 32, 40, 40]
        # feats_mask = mask_sample(orig_feats, coords1)
        code = sample(orig_code, coords1)       # orig_code: [64, 32, 32, 32] -> code: [64, 32, 40, 40]

        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)

        # pos_intra_loss, pos_intra_cd = self.helper(feats, feats_pos, code, code_pos)
        # pos_intra_loss, pos_intra_cd = self.helper(feats, feats_pos, code, code_pos)
        
        fd = tensor_correlation(norm(feats), norm(feats_pos))

        old_mean = fd.mean()
        fd -= fd.mean([3, 4], keepdim=True)
        fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(code), norm(code_pos))

        loss = (- cd.clamp(0) * (fd)).mean()

        return loss

class ContrastiveCorrelationLoss(nn.Module):

    def __init__(self):
        super(ContrastiveCorrelationLoss, self).__init__()
        self.inter_cal = None
        self.intra_cal = None
        self.neg_cal = None

        self.feature_samples = 40
        
    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, c1 , c2):

        fd = tensor_correlation(norm(f1), norm(f2))

        old_mean = fd.mean()
        fd -= fd.mean([3, 4], keepdim=True)
        fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))

        loss = (- cd.clamp(0) * (fd)).mean()

        return loss, cd
    
    def corr_helper_2(self, f1, f2, c1, c2, temperature=0.5, margin=0.1, poly_degree=2, reg_weight=0.001):
            
            pos_similarity = tensor_correlation(norm(f1), norm(f2))
            pos_similarity = torch.exp(pos_similarity / temperature)
            

            neg_similarity = tensor_correlation(norm(c1), norm(c2))
            neg_similarity = torch.exp(neg_similarity / temperature)
            
            # 对比损失，惩罚正样本对的相似度小于负样本对的相似度
            # torch.clamp确保相似度差异至少为0，从而避免负损失
            contrast_loss = torch.clamp(margin + pos_similarity - neg_similarity, min=0)
            
            # 多项式惩罚项，目的是减少正样本对的相似度
            poly_penalty = (pos_similarity - 1).pow(poly_degree).mean()
            
            # 正则化项，可以是参数的L2范数或其他形式的正则化
            # 这里对教师模型的特征向量应用L2正则化
            reg_loss = reg_weight * (f1.pow(2).sum() + f2.pow(2).sum())
            
            # 总损失是对比损失、多项式惩罚项和正则化项的和
            loss = contrast_loss.mean() + poly_penalty + reg_loss
            
            return loss

    def corr_helper(self, f1, f2, c1, c2, degree=2, margin=0.3, poly_weight=0.5):

        fd = tensor_correlation(norm(f1), norm(f2))     # [64, 128, 128, 128, 128]

        old_mean = fd.mean()
        fd -= fd.mean([3, 4], keepdim=True)
        fd = fd - fd.mean() + old_mean


        cd = tensor_correlation(norm(c1), norm(c2))

        # 原始的损失项，提高学生模型和教师模型对应特征向量间的相似性
        loss_corr = - cd.clamp(0) * (fd)    # shape: [64, 40, 40, 40, 40], type: <class 'torch.Tensor'>

        # 多项式惩罚项，目的是减少学生模型内部非对应元素的相似性
        # 使用clamp确保相似度非负，然后计算多项式和margin
        poly_penalty = torch.clamp(fd, min=0).pow(degree).mean(dim=[3, 4])  # shape: [64, 40, 40], type: <class 'torch.Tensor'>
        poly_penalty = (poly_penalty - margin).clamp(min=0).mean()  # 

        loss = loss_corr + poly_weight * poly_penalty     # shape: [64, 40, 40, 40, 40], type: <class 'torch.Tensor'>

        return loss, cd





    def forward(self,
                orig_feats: torch.Tensor, orig_feats_pos: torch.Tensor,orig_code: torch.Tensor, orig_code_pos: torch.Tensor,
                ):
        coord_shape = [orig_feats.shape[0], self.feature_samples, self.feature_samples, 2]      # [64, 40, 40, 2]
        coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
        coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
        # print(coords2)
        # print(coords2.shape)
        # while(1):
        #     pass
        feats = sample(orig_feats, coords1)     # orig_feats: [64, 32, 32, 32] -> feats: [64, 32, 40, 40]
        # feats_mask = mask_sample(orig_feats, coords1)
        code = sample(orig_code, coords1)       # orig_code: [64, 32, 32, 32] -> code: [64, 32, 40, 40]

        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)

        # pos_intra_loss, pos_intra_cd = self.helper(feats, feats_pos, code, code_pos)
        pos_intra_loss, pos_intra_cd = self.helper(feats, feats_pos, code, code_pos)

        return pos_intra_loss
    
    # use AttentionSampler 
    # def forward(self,
    #             orig_feats: torch.Tensor, orig_feats_pos: torch.Tensor,orig_code: torch.Tensor, orig_code_pos: torch.Tensor,
    #             ):
        
    #     scale_factor = 2
        
    #     coord_shape = [orig_feats.shape[0], self.feature_samples, self.feature_samples, 2]      # [64, 40, 40, 2]
    #     coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
    #     coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

    #     # attention_sampler = AttentionSampler(orig_feats[1], )

    #     f1 = attention_sampling(orig_feats, orig_feats.size(1), scale_factor, coords1)     # [64, 16, 128, 128]
    #     f2 = attention_sampling(orig_feats_pos, orig_feats_pos.size(1), scale_factor, coords2)       # [64, 16, 128, 128]

    #     c1 = attention_sampling(orig_code, orig_code.size(1), scale_factor, coords1)     # [64, 32, 64, 64]
    #     c2 = attention_sampling(orig_code_pos, orig_code_pos.size(1), scale_factor, coords2)     # [64, 32, 64, 64]
       


    #     # pos_intra_loss, pos_intra_cd = self.corr_helper(f1, f2, c1, c2)
    #     pos_intra_loss, pos_intra_cd = self.helper(f1, f2, c1, c2)

    #     return pos_intra_loss
if __name__ == '__main__':
    # a = torch.rand(8,64,512)
    # b = torch.rand(8,64,512)
    a = torch.rand(8,64,24,24)
    b = torch.rand(8,64,100,1)
    c = torch.rand(8,32,24,24)
    d = torch.rand(8,32,100,1)
    # c = torch.sum(a.unsqueeze(-1)*b.unsqueeze(-2),dim=1)
    # d = a.permute(0, 2,1) @ b
    corr_loss = ContrastiveCorrelationLoss()
    loss0 = CORR_loss(a,b,c,d)
    loss = corr_loss(a,b,c,d)
    print(loss0.shape)
    print(loss0)
    print(loss.shape)
    print(loss)