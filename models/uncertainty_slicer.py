"""
Uncertainty-Guided Probabilistic Slicer
Implements Proposal 1: Probabilistic patch selection based on uncertainty

Replaces the binary thresholding in AdaSlicer with probabilistic criterion:
E_rel(d_i) = ∫_d_i M_μ(x,y) · (1 - M_σ²(x,y)) dx dy > τ
"""

import torch
import torch.nn.functional as F
import math
from utils.general import make_divisible


class UncertaintySlicer:
    """
    Probabilistic slicer that uses uncertainty-aware patch selection
    
    Instead of binary thresholding, selects patches based on:
    E_rel(d_i) = ∫ M_μ · (1 - σ²) dx dy > τ
    
    This prioritizes patches with:
    - High mean objectness (M_μ)
    - Low uncertainty (σ²)
    """
    
    def __init__(self, ratio=8, threshold=0.3, uncertainty_weight=1.0):
        """
        Args:
            ratio: Downsampling ratio
            threshold: Threshold for reliable expected objectness
            uncertainty_weight: Weight for uncertainty term (1.0 means full uncertainty discounting)
        """
        self.ratio = ratio
        self.threshold = threshold
        self.uncertainty_weight = uncertainty_weight
        self.grid = None
        self.grid_vtx = None
    
    @torch.no_grad()
    def probabilistic_slice(self, mean_mask: torch.Tensor, uncertainty_mask: torch.Tensor, 
                          feat: torch.Tensor, ratio=8, threshold=0.3):
        """
        Probabilistic patch selection using uncertainty-guided criterion
        
        Args:
            mean_mask: Mean objectness M_μ [B, H, W]
            uncertainty_mask: Uncertainty (variance) M_σ² [B, H, W]
            feat: Feature map to slice [B, C, H, W]
            ratio: Downsampling ratio
            threshold: Threshold for reliable expected objectness
        
        Returns:
            total_clusters: List of patch coordinates [(x1, y1, x2, y2), ...] per batch
        """
        bs, height, width = mean_mask.shape
        device, dtype = mean_mask.device, mean_mask.dtype
        
        # Normalize uncertainty to [0, 1] range for weighting
        # Higher uncertainty -> lower confidence weight
        uncertainty_normalized = torch.clamp(uncertainty_mask, 0, 1)
        confidence_weight = 1.0 - self.uncertainty_weight * uncertainty_normalized
        
        # Compute reliable expected objectness: E_rel = M_μ · (1 - σ²)
        reliable_objectness = mean_mask * confidence_weight
        
        # Compute patch dimensions
        cluster_w, cluster_h = make_divisible(width / ratio, 4), make_divisible(height / ratio, 4)
        half_clus_w, half_clus_h = cluster_w // 2, cluster_h // 2
        
        # Find local maxima in reliable objectness (not just mean)
        # Max pooling to find local peaks
        maxima = F.max_pool2d(reliable_objectness, 3, stride=1, padding=1) == reliable_objectness
        
        # Threshold on reliable objectness (not binary threshold on mean)
        activated = reliable_objectness >= threshold
        obj_centers = activated & maxima
        
        # Average pool to get patch-level reliable objectness scores
        padding = half_clus_w // 2
        patch_scores = F.avg_pool2d(reliable_objectness, padding * 2 + 1, stride=1, padding=padding)
        
        # Get center coordinates
        cb, cy, cx = obj_centers.nonzero(as_tuple=True)
        patch_scores_at_centers = patch_scores[cb, cy, cx]
        
        outs = []
        for bi in range(bs):
            ci = cb == bi
            cn = ci.sum().item()
            if cn == 0:
                outs.append(torch.zeros((0, 4), device=device, dtype=dtype))
                continue
            
            if bs == 1:
                scores = patch_scores_at_centers
                cy_bi, cx_bi = cy, cx
            else:
                scores = patch_scores_at_centers[ci]
                cy_bi, cx_bi = cy[ci], cx[ci]
            
            # Initialize patches around centers
            init_x1 = cx_bi.clamp(half_clus_w, width - half_clus_w) - half_clus_w
            init_y1 = cy_bi.clamp(half_clus_h, height - half_clus_h) - half_clus_h
            
            # Create grid for patch refinement
            if not hasattr(self, 'grid') or self.grid is None or self.grid[0].shape[-1] != cluster_h * cluster_w:
                gy, gx = torch.meshgrid(torch.arange(cluster_h), torch.arange(cluster_w), indexing='ij')
                self.grid = (gy.reshape(1, -1).to(device), gx.reshape(1, -1).to(device))
            gy, gx = self.grid
            
            # Check activation within each patch using reliable_objectness
            act_x, act_y = (init_x1.view(-1, 1) + gx).view(-1), (init_y1.view(-1, 1) + gy).view(-1)
            act = activated[bi, act_y, act_x].view(cn, cluster_h, cluster_w)
            
            # Refine patch boundaries based on activation
            act_x, act_y = act.any(dim=1).long(), act.any(dim=2).long()
            dx1, dx2 = (1 - act_x).argmin(dim=1), -(1 - act_x.flip((1,))).argmin(dim=1)
            dy1, dy2 = (1 - act_y).argmin(dim=1), -(1 - act_y.flip((1,))).argmin(dim=1)
            dx = torch.where(dx1.abs() > dx2.abs(), dx1, dx2)
            dy = torch.where(dy1.abs() > dy2.abs(), dy1, dy2)
            
            refine_x1 = (init_x1 + dx).clamp(0, width - cluster_w).to(dtype)
            refine_y1 = (init_y1 + dy).clamp(0, height - cluster_h).to(dtype)
            refine_x2, refine_y2 = refine_x1 + cluster_w, refine_y1 + cluster_h
            total_clusters = torch.stack((refine_x1, refine_y1, refine_x2, refine_y2), dim=1).long()
            
            # Filter patches by reliable expected objectness
            # Compute integral: E_rel = mean(M_μ · (1 - σ²)) over patch area
            patch_reliable_scores = []
            for i, (x1, y1, x2, y2) in enumerate(total_clusters):
                patch_mean = mean_mask[bi, y1:y2, x1:x2].mean()
                patch_uncertainty = uncertainty_mask[bi, y1:y2, x1:x2].mean()
                reliable_score = patch_mean * (1.0 - self.uncertainty_weight * torch.clamp(patch_uncertainty, 0, 1))
                patch_reliable_scores.append(reliable_score)
            
            patch_reliable_scores = torch.stack(patch_reliable_scores)
            
            # NMS on patches with reliable objectness > threshold
            overlap = (refine_x1[:, None] <= cx_bi[None, :]) & (cx_bi[None, :] < refine_x2[:, None]) & \
                     (refine_y1[:, None] <= cy_bi[None, :]) & (cy_bi[None, :] < refine_y2[:, None])
            
            clusters = []
            contained = torch.full_like(overlap[0], False)
            
            # Sort by reliable score (descending)
            sorted_indices = torch.argsort(patch_reliable_scores, descending=True)
            
            for idx in sorted_indices:
                if contained[idx] or patch_reliable_scores[idx] < threshold:
                    continue
                clusters.append(total_clusters[idx])
                contained |= overlap[idx]
            
            outs.append(torch.stack(clusters) if len(clusters) else torch.zeros((0, 4), device=device, dtype=dtype))
        
        return outs
    
    @torch.no_grad()
    def probabilistic_slice_fast(self, mean_mask: torch.Tensor, uncertainty_mask: torch.Tensor,
                                 feat: torch.Tensor, ratio=8, threshold=0.3):
        """
        Faster version using uniform grid (similar to ada_slicer_fast)
        """
        bs, height, width = mean_mask.shape
        device, dtype = mean_mask.device, mean_mask.dtype
        
        # Compute reliable objectness
        uncertainty_normalized = torch.clamp(uncertainty_mask, 0, 1)
        confidence_weight = 1.0 - self.uncertainty_weight * uncertainty_normalized
        reliable_objectness = mean_mask * confidence_weight
        
        cluster_w, cluster_h = make_divisible(width / ratio, 4), make_divisible(height / ratio, 4)
        ratio_x, ratio_y = int(math.ceil(width / cluster_w)), int(math.ceil(height / cluster_h))
        half_clus_w, half_clus_h = cluster_w // 2, cluster_h // 2
        
        # Pre-compute grid vertices
        if getattr(self, 'grid_vtx', None) is None or self.grid_vtx.size(0) != ratio_x * ratio_y * bs:
            gy, gx = torch.meshgrid(torch.arange(ratio_y), torch.arange(ratio_x), indexing='ij')
            gxy = torch.stack((gy.reshape(-1), gx.reshape(-1)), dim=1).unsqueeze(0).repeat(bs, 1, 1).view(-1, 2)
            gb = torch.arange(bs).view(-1, 1).repeat(1, ratio_x * ratio_y).view(-1, 1)
            self.grid_vtx = torch.cat((gb, gxy), dim=1).to(device)
        
        rb, ry, rx = self.grid_vtx.T
        
        # Find local maxima in reliable objectness
        maxima = F.max_pool2d(reliable_objectness, 3, stride=1, padding=1) == reliable_objectness
        activated = reliable_objectness >= threshold
        obj_centers = activated & maxima
        
        if (~obj_centers).all():
            return [torch.zeros((0, 4), device=device, dtype=dtype) for _ in range(bs)]
        
        cb, cy, cx = obj_centers.nonzero(as_tuple=True)
        
        # Compute patch-level reliable scores
        padding = half_clus_w // 2
        patch_scores = F.avg_pool2d(reliable_objectness, padding * 2 + 1, stride=1, padding=padding)
        patch_scores_at_centers = patch_scores[cb, cy, cx]
        
        # Initialize patches
        init_x1 = (cx - half_clus_w).clamp(0, width - cluster_w)
        init_y1 = (cy - half_clus_h).clamp(0, height - cluster_h)
        
        # Create grid
        if getattr(self, 'grid', None) is None or self.grid[0].shape[-1] != cluster_h * cluster_w:
            gy, gx = torch.meshgrid(torch.arange(cluster_h), torch.arange(cluster_w), indexing='ij')
            self.grid = (gy.reshape(1, -1).to(device), gx.reshape(1, -1).to(device))
        gy, gx = self.grid
        
        # Check activation and refine
        act_x, act_y = (init_x1.view(-1, 1) + gx).view(-1), (init_y1.view(-1, 1) + gy).view(-1)
        act = activated[cb, act_y, act_x].view(len(cb), cluster_h, cluster_w)
        
        act_x, act_y = act.any(dim=1).long(), act.any(dim=2).long()
        dx1, dx2 = (1 - act_x).argmin(dim=1), -(1 - act_x.flip((1,))).argmin(dim=1)
        dy1, dy2 = (1 - act_y).argmin(dim=1), -(1 - act_y.flip((1,))).argmin(dim=1)
        dx = torch.where(dx1.abs() > dx2.abs(), dx1, dx2)
        dy = torch.where(dy1.abs() > dy2.abs(), dy1, dy2)
        
        refine_x1 = (init_x1 + dx).clamp(0, width - cluster_w).to(dtype)
        refine_y1 = (init_y1 + dy).clamp(0, height - cluster_h).to(dtype)
        refine_x2, refine_y2 = refine_x1 + cluster_w, refine_y1 + cluster_h
        total_clusters = torch.stack((refine_x1, refine_y1, refine_x2, refine_y2), dim=1).long()
        
        # Filter by reliable score and NMS
        outs = []
        for bi in range(bs):
            ci = cb == bi
            if not ci.any():
                outs.append(torch.zeros((0, 4), device=device, dtype=dtype))
                continue
            
            clusters_bi = total_clusters[ci]
            scores_bi = patch_scores_at_centers[ci]
            
            # Compute reliable scores for each patch
            patch_reliable_scores = []
            for (x1, y1, x2, y2) in clusters_bi:
                patch_mean = mean_mask[bi, y1:y2, x1:x2].mean()
                patch_uncertainty = uncertainty_mask[bi, y1:y2, x1:x2].mean()
                reliable_score = patch_mean * (1.0 - self.uncertainty_weight * torch.clamp(patch_uncertainty, 0, 1))
                patch_reliable_scores.append(reliable_score)
            
            patch_reliable_scores = torch.stack(patch_reliable_scores)
            
            # NMS
            overlap = (clusters_bi[:, 0:1] <= cx[ci][None, :]) & (cx[ci][None, :] < clusters_bi[:, 2:3]) & \
                     (clusters_bi[:, 1:2] <= cy[ci][None, :]) & (cy[ci][None, :] < clusters_bi[:, 3:4])
            
            clusters = []
            contained = torch.full_like(overlap[0], False)
            sorted_indices = torch.argsort(patch_reliable_scores, descending=True)
            
            for idx in sorted_indices:
                if contained[idx] or patch_reliable_scores[idx] < threshold:
                    continue
                clusters.append(clusters_bi[idx])
                contained |= overlap[idx]
            
            outs.append(torch.stack(clusters) if len(clusters) else torch.zeros((0, 4), device=device, dtype=dtype))
        
        return outs

