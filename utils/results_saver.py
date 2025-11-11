"""
Results Saver for structured output during testing
Saves visual and quantitative results in organized folder structure
"""

import csv
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import warnings

from utils.general import xyxy2xywh, xywh2xyxy
from utils.plots import plot_one_box, colors


def create_results_structure(base_dir: Path, dataset_name: str, model_name: str) -> Dict[str, Path]:
    """
    Create structured directory for results
    
    Returns:
        Dictionary with paths to different result directories
    """
    results_root = base_dir / 'results' / dataset_name / model_name
    dirs = {
        'root': results_root,
        'individual_results': results_root / 'individual_results',
        'qualitative_top5': results_root / 'individual_results' / 'top5_best',
        'quantitative_csv': results_root / 'individual_results' / 'quantitative_all.csv',
        'comparative': results_root / 'comparative_results',  # For multi-model comparisons
    }
    
    # Only create directories, not files
    directory_paths = [
        dirs['root'],
        dirs['individual_results'],
        dirs['qualitative_top5'],
        dirs['comparative'],
    ]
    
    for dir_path in directory_paths:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def save_image_result(image: np.ndarray, pred_boxes: torch.Tensor, gt_boxes: Optional[torch.Tensor],
                     image_path: Path, save_path: Path, names: Dict, conf_thres: float = 0.25):
    """
    Save a single image with predictions and ground truth
    
    Args:
        image: Original image (H, W, C) - can be RGB or BGR
        pred_boxes: Predicted boxes tensor/array [N, 6] (x1, y1, x2, y2, conf, cls)
        gt_boxes: Ground truth boxes tensor/array [M, 6] (x1, y1, x2, y2, conf=1.0, cls)
        image_path: Original image path
        save_path: Path to save visualization
        names: Class names dictionary or list
        conf_thres: Confidence threshold for visualization
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Ensure image is contiguous for OpenCV operations
    image = np.ascontiguousarray(image)
    
    # Get original image dimensions BEFORE any color conversion
    orig_img_height, orig_img_width = image.shape[:2]
    
    # Images from dataset are typically RGB after transforms, but OpenCV uses BGR
    # We'll assume input is RGB (common after PIL/transforms), convert to BGR for OpenCV drawing
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert RGB to BGR for OpenCV operations
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        needs_rgb_conversion = True  # Convert back to RGB for PIL saving
    else:
        image_bgr = image.copy()
        needs_rgb_conversion = False
    
    # Get image dimensions (should be same as orig after color conversion)
    img_height, img_width = image_bgr.shape[:2]
    
    # Ensure image is contiguous for OpenCV operations (required by plot_one_box)
    image_bgr = np.ascontiguousarray(image_bgr)
    
    # Debug: Check dimensions match
    if img_height != orig_img_height or img_width != orig_img_width:
        print(f"WARNING: Image dimensions changed after color conversion: {orig_img_height}x{orig_img_width} -> {img_height}x{img_width}")
    
    # Debug: Draw a test box at known locations to verify coordinate system
    # Draw boxes at corners to verify alignment
    if 'top1' in str(save_path) or 'top2' in str(save_path):  # Only for first 2 images to avoid clutter
        # Top-left corner (small box)
        cv2.rectangle(image_bgr, (10, 10), (50, 50), (255, 255, 0), 3)  # Cyan box in BGR
        # Bottom-right corner (small box)  
        cv2.rectangle(image_bgr, (img_width - 50, img_height - 50), (img_width - 10, img_height - 10), (255, 255, 0), 3)  # Cyan box
        print(f"DEBUG: Test boxes drawn at corners on {save_path.name} (img: {img_height}x{img_width})")
    
    # Helper function to get class name (shortened for display)
    def get_class_name(cls_id, short=True):
        if isinstance(names, dict):
            name = names.get(cls_id, str(cls_id))
        elif isinstance(names, (list, tuple)) and cls_id < len(names):
            name = names[cls_id]
        else:
            name = str(cls_id)
        
        # Shorten class names to reduce label size
        if short:
            # Take first 3 characters or first letter if name is long
            if len(name) > 3:
                # For multi-word names, take first letter of each word
                if ' ' in name:
                    name = ''.join([w[0].upper() for w in name.split()])
                else:
                    name = name[:3].upper()
        return name
    
    # Draw predicted boxes FIRST (red in BGR = (0, 0, 255))
    # So GT boxes drawn later will be on top and more visible
    if pred_boxes is not None and len(pred_boxes) > 0:
        # Convert tensor to numpy if needed
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes_np = pred_boxes.cpu().numpy()
        else:
            pred_boxes_np = np.array(pred_boxes)
        
        # Ensure 2D array
        if len(pred_boxes_np.shape) == 1:
            pred_boxes_np = pred_boxes_np.reshape(1, -1)
        
        boxes_drawn = 0
        boxes_out_of_bounds = 0
        boxes_invalid = 0
        boxes_below_thresh = 0
        for box_idx, box in enumerate(pred_boxes_np):
            if len(box) >= 6:
                conf = float(box[4])
                if conf >= conf_thres:
                    x1, y1, x2, y2 = box[:4].astype(float)
                    
                    # Check if boxes are out of bounds before clipping
                    if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
                        boxes_out_of_bounds += 1
                    
                    # Clip coordinates to image boundaries
                    x1_clipped = max(0, min(x1, img_width))
                    y1_clipped = max(0, min(y1, img_height))
                    x2_clipped = max(0, min(x2, img_width))
                    y2_clipped = max(0, min(y2, img_height))
                    
                    # Only draw if box is valid (has positive area)
                    if x2_clipped > x1_clipped and y2_clipped > y1_clipped:
                        x1, y1, x2, y2 = int(x1_clipped), int(y1_clipped), int(x2_clipped), int(y2_clipped)
                        cls = int(box[5])
                        label = get_class_name(cls, short=True)
                        color = (0, 0, 255)  # Red in BGR for predictions
                        # Draw box - ensure coordinates are valid
                        # Note: OpenCV uses (x, y) where x is width (column) and y is height (row)
                        if 0 <= x1 < img_width and 0 <= y1 < img_height and 0 < x2 <= img_width and 0 < y2 <= img_height:
                            # Debug: Print first few boxes for top images
                            if boxes_drawn < 3 and ('top1' in str(save_path) or 'top2' in str(save_path)):
                                print(f"  Pred box {boxes_drawn}: ({x1}, {y1}) to ({x2}, {y2}), conf={conf:.3f}, img: {img_height}x{img_width}")
                            # Short label: just class abbreviation and confidence (1 decimal)
                            short_label = f'{label} {conf:.1f}' if conf >= 0.1 else f'{label}'
                            plot_one_box((x1, y1, x2, y2), image_bgr, 
                                        label=short_label, 
                                        color=color, line_thickness=2)
                            boxes_drawn += 1
                        else:
                            boxes_invalid += 1
                            if boxes_invalid <= 3:
                                print(f"  Pred box invalid: ({x1}, {y1}) to ({x2}, {y2}), img: {img_height}x{img_width}")
                    else:
                        boxes_invalid += 1
                else:
                    boxes_below_thresh += 1
        # Debug: print if no boxes were drawn or if boxes are out of bounds
        if boxes_drawn == 0 and len(pred_boxes_np) > 0:
            print(f"WARNING: {len(pred_boxes_np)} pred boxes provided but none were drawn for {save_path.name} (conf_thres={conf_thres})")
            if len(pred_boxes_np) > 0:
                sample_box = pred_boxes_np[0]
                print(f"  Image size: {img_height}x{img_width}, Sample box: {sample_box[:4]}, conf: {sample_box[4]}")
                print(f"  Below thresh: {boxes_below_thresh}, Invalid: {boxes_invalid}, Out of bounds: {boxes_out_of_bounds}, Drawn: {boxes_drawn}")
        elif boxes_drawn > 0:
            print(f"DEBUG: Drawn {boxes_drawn} pred boxes on {save_path.name} (img: {img_height}x{img_width}, conf_thres={conf_thres})")
        if boxes_out_of_bounds > 0:
            print(f"WARNING: {boxes_out_of_bounds} pred boxes out of bounds for {save_path.name} (img: {img_height}x{img_width})")
    
    # Draw ground truth boxes AFTER predictions (green in BGR = (0, 255, 0))
    # This ensures GT boxes are on top and more visible
    if gt_boxes is not None and len(gt_boxes) > 0:
        # Convert tensor to numpy if needed
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes_np = gt_boxes.cpu().numpy()
        else:
            gt_boxes_np = np.array(gt_boxes)
        
        # Ensure 2D array
        if len(gt_boxes_np.shape) == 1:
            gt_boxes_np = gt_boxes_np.reshape(1, -1)
        
        boxes_drawn = 0
        boxes_out_of_bounds = 0
        boxes_invalid = 0
        for box_idx, box in enumerate(gt_boxes_np):
            if len(box) >= 4:
                x1, y1, x2, y2 = box[:4].astype(float)
                
                # Check if boxes are out of bounds before clipping
                if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
                    boxes_out_of_bounds += 1
                
                # Clip coordinates to image boundaries
                x1_clipped = max(0, min(x1, img_width))
                y1_clipped = max(0, min(y1, img_height))
                x2_clipped = max(0, min(x2, img_width))
                y2_clipped = max(0, min(y2, img_height))
                
                # Only draw if box is valid (has positive area)
                if x2_clipped > x1_clipped and y2_clipped > y1_clipped:
                    x1, y1, x2, y2 = int(x1_clipped), int(y1_clipped), int(x2_clipped), int(y2_clipped)
                    cls = int(box[5]) if len(box) > 5 else 0
                    label = get_class_name(cls, short=True)
                    # Draw box - ensure coordinates are valid
                    # Note: OpenCV uses (x, y) where x is width (column) and y is height (row)
                    # So (x1, y1) is top-left and (x2, y2) is bottom-right
                    if 0 <= x1 < img_width and 0 <= y1 < img_height and 0 < x2 <= img_width and 0 < y2 <= img_height:
                        # Debug: Print first few boxes for top images
                        if boxes_drawn < 3 and ('top1' in str(save_path) or 'top2' in str(save_path)):
                            print(f"  GT box {boxes_drawn}: ({x1}, {y1}) to ({x2}, {y2}), img: {img_height}x{img_width}")
                        # Use thicker line (4) and bright green/yellow color to make GT boxes more visible
                        # Bright green-yellow in BGR = (0, 255, 255) for better visibility
                        # Very short label: just "GT" or class abbreviation only
                        gt_label = 'GT' if len(label) > 2 else f'GT:{label}'
                        plot_one_box((x1, y1, x2, y2), image_bgr, label=gt_label, 
                                    color=(0, 255, 255), line_thickness=4)  # Thicker, brighter color
                        boxes_drawn += 1
                    else:
                        boxes_invalid += 1
                        if boxes_invalid <= 3:
                            print(f"  GT box invalid: ({x1}, {y1}) to ({x2}, {y2}), img: {img_height}x{img_width}")
                else:
                    boxes_invalid += 1
        # Debug: print if no boxes were drawn or if boxes are out of bounds
        if boxes_drawn == 0 and len(gt_boxes_np) > 0:
            print(f"WARNING: {len(gt_boxes_np)} GT boxes provided but none were drawn for {save_path.name}")
            if len(gt_boxes_np) > 0:
                sample_box = gt_boxes_np[0]
                print(f"  Image size: {img_height}x{img_width}, Sample box: {sample_box[:4]}")
                print(f"  Invalid: {boxes_invalid}, Out of bounds: {boxes_out_of_bounds}, Drawn: {boxes_drawn}")
        elif boxes_drawn > 0:
            print(f"DEBUG: Drawn {boxes_drawn} GT boxes on {save_path.name} (img: {img_height}x{img_width})")
        if boxes_out_of_bounds > 0:
            print(f"WARNING: {boxes_out_of_bounds} GT boxes out of bounds for {save_path.name} (img: {img_height}x{img_width})")
    
    # Convert back to RGB for PIL saving
    if needs_rgb_conversion:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_bgr
    
    # Ensure image is uint8 and contiguous for PIL
    if image_rgb.dtype != np.uint8:
        image_rgb = image_rgb.astype(np.uint8)
    image_rgb = np.ascontiguousarray(image_rgb)
    
    # Verify final dimensions match original
    final_h, final_w = image_rgb.shape[:2]
    if final_h != orig_img_height or final_w != orig_img_width:
        print(f"ERROR: Final image dimensions don't match original: {orig_img_height}x{orig_img_width} -> {final_h}x{final_w} for {save_path.name}")
    
    # Save image with PIL (same as batch images)
    Image.fromarray(image_rgb).save(save_path)
    return save_path


def compute_image_metrics(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor,
                         iouv: torch.Tensor, conf_thres: float = 0.001) -> Dict:
    """
    Compute per-image quantitative metrics
    
    Returns:
        Dictionary with metrics for this image
    """
    metrics = {
        'num_gt': len(gt_boxes) if gt_boxes is not None else 0,
        'num_pred': 0,
        'num_tp': 0,
        'num_fp': 0,
        'num_fn': 0,
        'mean_iou': 0.0,
        'mean_conf': 0.0,
        'best_iou': 0.0,
        'worst_iou': 1.0,
    }
    
    if pred_boxes is None or len(pred_boxes) == 0:
        metrics['num_fn'] = metrics['num_gt']
        return metrics
    
    # Filter by confidence
    pred_filtered = pred_boxes[pred_boxes[:, 4] >= conf_thres]
    metrics['num_pred'] = len(pred_filtered)
    
    if metrics['num_pred'] == 0:
        metrics['num_fn'] = metrics['num_gt']
        return metrics
    
    # Compute IoUs
    if gt_boxes is not None and len(gt_boxes) > 0:
        from utils.general import box_iou
        # pred_filtered: [N, 6] -> [N, 4] for IoU
        # gt_boxes: [M, 6] -> [M, 4] for IoU
        pred_xyxy = pred_filtered[:, :4]
        gt_xyxy = gt_boxes[:, :4]
        
        # Filter out invalid boxes (x2 <= x1 or y2 <= y1) before IoU calculation
        pred_valid = (pred_xyxy[:, 2] > pred_xyxy[:, 0]) & (pred_xyxy[:, 3] > pred_xyxy[:, 1])
        gt_valid = (gt_xyxy[:, 2] > gt_xyxy[:, 0]) & (gt_xyxy[:, 3] > gt_xyxy[:, 1])
        
        if not pred_valid.all():
            invalid_count = (~pred_valid).sum().item()
            if invalid_count > 0:
                print(f"WARNING: {invalid_count} invalid pred boxes (x2<=x1 or y2<=y1) - filtering out")
            pred_xyxy = pred_xyxy[pred_valid]
            pred_filtered = pred_filtered[pred_valid]
        
        if not gt_valid.all():
            invalid_count = (~gt_valid).sum().item()
            if invalid_count > 0:
                print(f"WARNING: {invalid_count} invalid GT boxes (x2<=x1 or y2<=y1) - filtering out")
            gt_xyxy = gt_xyxy[gt_valid]
            gt_boxes = gt_boxes[gt_valid]
        
        # Skip IoU calculation if no valid boxes
        if len(pred_xyxy) == 0 or len(gt_xyxy) == 0:
            return {
                'num_gt': len(gt_boxes) if len(gt_boxes) > 0 else 0,
                'num_pred': len(pred_filtered) if len(pred_filtered) > 0 else 0,
                'num_tp': 0,
                'num_fp': len(pred_filtered) if len(pred_filtered) > 0 else 0,
                'num_fn': len(gt_boxes) if len(gt_boxes) > 0 else 0,
                'mean_iou': 0.0
            }
        
        ious = box_iou(pred_xyxy, gt_xyxy)  # [N, M]
        max_ious, _ = ious.max(dim=1)  # Best IoU for each prediction
        
        # DEBUG: Check if all IoUs are 0
        if len(max_ious) > 0 and max_ious.max() == 0.0:
            # Check if boxes are in completely different ranges
            pred_x_center = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
            pred_y_center = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
            gt_x_center = (gt_xyxy[:, 0] + gt_xyxy[:, 2]) / 2
            gt_y_center = (gt_xyxy[:, 1] + gt_xyxy[:, 3]) / 2
            
            print(f"DEBUG: All IoUs are 0!")
            print(f"  Pred center range: X=[{pred_x_center.min():.1f}, {pred_x_center.max():.1f}], Y=[{pred_y_center.min():.1f}, {pred_y_center.max():.1f}]")
            print(f"  GT center range:   X=[{gt_x_center.min():.1f}, {gt_x_center.max():.1f}], Y=[{gt_y_center.min():.1f}, {gt_y_center.max():.1f}]")
            print(f"  Pred box range: X=[{pred_xyxy[:, 0].min():.1f}, {pred_xyxy[:, 2].max():.1f}], Y=[{pred_xyxy[:, 1].min():.1f}, {pred_xyxy[:, 3].max():.1f}]")
            print(f"  GT box range:   X=[{gt_xyxy[:, 0].min():.1f}, {gt_xyxy[:, 2].max():.1f}], Y=[{gt_xyxy[:, 1].min():.1f}, {gt_xyxy[:, 3].max():.1f}]")
        
        matched_gt = set()
        
        # Count TP/FP
        for i, iou in enumerate(max_ious):
            if iou > iouv[0]:  # TP if IoU > 0.5
                gt_idx = ious[i].argmax().item()
                if gt_idx not in matched_gt:
                    matched_gt.add(gt_idx)
                    metrics['num_tp'] += 1
                else:
                    metrics['num_fp'] += 1
            else:
                metrics['num_fp'] += 1
        
        metrics['num_fn'] = metrics['num_gt'] - len(matched_gt)
        metrics['mean_iou'] = float(max_ious.mean()) if len(max_ious) > 0 else 0.0
        metrics['best_iou'] = float(max_ious.max()) if len(max_ious) > 0 else 0.0
        metrics['worst_iou'] = float(max_ious.min()) if len(max_ious) > 0 else 1.0
    else:
        # No ground truth
        metrics['num_fp'] = metrics['num_pred']
    
    metrics['mean_conf'] = float(pred_filtered[:, 4].mean()) if len(pred_filtered) > 0 else 0.0
    
    return metrics


def save_quantitative_csv(image_metrics_list: List[Dict], save_path: Path):
    """
    Save quantitative metrics for all images to CSV
    
    Args:
        image_metrics_list: List of dictionaries, each containing metrics for one image
        save_path: Path to save CSV file
    """
    if not image_metrics_list:
        return
    
    # Ensure parent directory exists
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if save_path is actually a directory (from previous buggy runs)
    if save_path.exists() and save_path.is_dir():
        raise ValueError(f"Save path '{save_path}' is a directory, not a file. Please remove it and try again.")
    
    fieldnames = ['image_path', 'image_name', 'num_gt', 'num_pred', 'num_tp', 'num_fp', 'num_fn',
                 'precision', 'recall', 'f1', 'mean_iou', 'mean_conf', 'best_iou', 'worst_iou']
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for img_result in image_metrics_list:
            # Extract metrics from nested structure
            metrics = img_result.get('metrics', {})
            image_path = img_result.get('image_path', metrics.get('image_path', ''))
            
            # Get metric values with defaults
            num_gt = metrics.get('num_gt', 0)
            num_pred = metrics.get('num_pred', 0)
            num_tp = metrics.get('num_tp', 0)
            num_fp = metrics.get('num_fp', 0)
            num_fn = metrics.get('num_fn', 0)
            
            # Use precomputed precision/recall if available, otherwise compute
            precision = metrics.get('precision')
            if precision is None:
                precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0.0
            
            recall = metrics.get('recall')
            if recall is None:
                recall = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0.0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            row = {
                'image_path': image_path,
                'image_name': Path(image_path).name if image_path else '',
                'num_gt': num_gt,
                'num_pred': num_pred,
                'num_tp': num_tp,
                'num_fp': num_fp,
                'num_fn': num_fn,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mean_iou': metrics.get('mean_iou', 0.0),
                'mean_conf': metrics.get('mean_conf', 0.0),
                'best_iou': metrics.get('best_iou', 0.0),
                'worst_iou': metrics.get('worst_iou', 1.0),
            }
            writer.writerow(row)


def save_top5_results(image_results: List[Dict], save_dir: Path, names: Dict, vis_conf_thres: float = 0.25):
    """
    Save top 5 best results as qualitative visualizations
    
    Args:
        image_results: List of dicts with keys: 'image', 'pred_boxes', 'gt_boxes', 'image_path', 'metrics'
        save_dir: Directory to save top 5 images
        names: Class names dictionary
        vis_conf_thres: Confidence threshold for visualization (default: 0.25 to show only confident detections)
    """
    # Sort by mean IoU (best first)
    sorted_results = sorted(image_results, 
                           key=lambda x: x['metrics'].get('mean_iou', 0.0), 
                           reverse=True)
    
    top5 = sorted_results[:5]
    
    for idx, result in enumerate(top5, 1):
        image = result['image']
        pred_boxes = result.get('pred_boxes')
        gt_boxes = result.get('gt_boxes')
        image_path = result['image_path']
        metrics = result['metrics']
        
        image_name = Path(image_path).stem
        save_path = save_dir / f'top{idx}_{image_name}_iou{metrics["mean_iou"]:.3f}.jpg'
        
        # Load image fresh from disk to ensure exact format match with boxes
        # Boxes are computed for the original image, so we should use the original
        import cv2
        img_from_disk = cv2.imread(str(image_path))  # Loads in BGR format
        if img_from_disk is None:
            # Fallback to using provided image if file not found
            print(f"WARNING: Could not load image from {image_path}, using provided image")
            if isinstance(image, np.ndarray):
                img_with_metrics = image.copy()
            elif isinstance(image, torch.Tensor):
                img_with_metrics = image.cpu().numpy()
            else:
                img_with_metrics = np.array(image)
            
            # Ensure image is in correct format (H, W, C) and uint8
            if len(img_with_metrics.shape) == 3 and img_with_metrics.shape[0] == 3:
                # CHW -> HWC
                img_with_metrics = img_with_metrics.transpose(1, 2, 0)
            
            if img_with_metrics.dtype != np.uint8:
                if img_with_metrics.max() <= 1.0:
                    img_with_metrics = (img_with_metrics * 255).astype(np.uint8)
                else:
                    img_with_metrics = img_with_metrics.astype(np.uint8)
            
            # Convert to RGB (assuming it's in RGB format from dataset)
            if len(img_with_metrics.shape) == 3 and img_with_metrics.shape[2] == 3:
                img_with_metrics = cv2.cvtColor(img_with_metrics, cv2.COLOR_BGR2RGB)
        else:
            # Use image from disk - convert BGR to RGB
            img_with_metrics = cv2.cvtColor(img_from_disk, cv2.COLOR_BGR2RGB)
            print(f"DEBUG: Loaded image from disk: {image_path}, size: {img_with_metrics.shape}")
        
        # Store original dimensions for debugging
        orig_h, orig_w = img_with_metrics.shape[:2]
        
        # Convert to BGR for OpenCV text rendering (but don't change dimensions)
        if len(img_with_metrics.shape) == 3 and img_with_metrics.shape[2] == 3:
            # Assume RGB, convert to BGR for cv2
            img_bgr = cv2.cvtColor(img_with_metrics, cv2.COLOR_RGB2BGR)
            
            # Verify dimensions didn't change
            if img_bgr.shape[:2] != (orig_h, orig_w):
                print(f"ERROR: Image dimensions changed during BGR conversion: {orig_h}x{orig_w} -> {img_bgr.shape[:2]}")
            
            # Add metrics text overlay
            metrics_text = [
                f"Rank: {idx}/5",
                f"Mean IoU: {metrics['mean_iou']:.3f}",
                f"Precision: {metrics.get('precision', 0):.3f}",
                f"Recall: {metrics.get('recall', 0):.3f}",
                f"TP: {metrics['num_tp']}, FP: {metrics['num_fp']}, FN: {metrics['num_fn']}"
            ]
            
            y_offset = 30
            for text in metrics_text:
                # White outline, then black text
                cv2.putText(img_bgr, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(img_bgr, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 0, 0), 2, cv2.LINE_AA)
                y_offset += 25
            
            # Convert back to RGB - save_image_result will handle BGR conversion internally
            img_with_metrics = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Verify dimensions still match
            if img_with_metrics.shape[:2] != (orig_h, orig_w):
                print(f"ERROR: Image dimensions changed after RGB conversion: {orig_h}x{orig_w} -> {img_with_metrics.shape[:2]}")
        else:
            # If not 3-channel, use as-is
            img_with_metrics = img_with_metrics
        
        # Debug: Print dimensions before passing to save_image_result
        final_h, final_w = img_with_metrics.shape[:2]
        if pred_boxes is not None and len(pred_boxes) > 0:
            if isinstance(pred_boxes, torch.Tensor):
                pred_sample = pred_boxes[0].cpu().numpy() if len(pred_boxes) > 0 else None
            else:
                pred_sample = pred_boxes[0] if len(pred_boxes) > 0 else None
            if pred_sample is not None:
                print(f"DEBUG save_top5: Image {idx} - img: {final_h}x{final_w}, sample pred box: {pred_sample[:4]}")
        
        # Pass the RGB image to save_image_result (it will convert to BGR for drawing)
        # IMPORTANT: Do not resize the image - boxes are in original image coordinates
        # Use vis_conf_thres for visualization (higher threshold to reduce clutter)
        save_image_result(img_with_metrics, pred_boxes, gt_boxes, 
                         image_path, save_path, names, conf_thres=vis_conf_thres)


def create_comparative_visualization(model_results: List[Dict], save_path: Path, names: Dict):
    """
    Create side-by-side comparison visualization for multiple models on the same image
    
    Args:
        model_results: List of dicts, each with 'model_name', 'image', 'pred_boxes', 'gt_boxes', 'metrics'
        save_path: Path to save comparison image
        names: Class names dictionary
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    num_models = len(model_results)
    fig = plt.figure(figsize=(6 * num_models, 6))
    gs = gridspec.GridSpec(1, num_models + 1, figure=fig)
    
    # Ground truth image
    if model_results[0].get('gt_boxes') is not None:
        ax_gt = fig.add_subplot(gs[0, 0])
        gt_img = model_results[0]['image']
        if isinstance(gt_img, torch.Tensor):
            gt_img = gt_img.cpu().numpy()
        if gt_img.dtype != np.uint8:
            if gt_img.max() <= 1.0:
                gt_img = (gt_img * 255).astype(np.uint8)
        ax_gt.imshow(gt_img)
        ax_gt.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax_gt.axis('off')
    
    # Model predictions
    for idx, result in enumerate(model_results, 1):
        ax = fig.add_subplot(gs[0, idx])
        img = result['image']
        pred_boxes = result.get('pred_boxes')
        metrics = result.get('metrics', {})
        
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
        
        # Draw predictions on image
        img_vis = img.copy()
        if pred_boxes is not None and len(pred_boxes) > 0:
            for box in pred_boxes:
                if len(box) >= 6 and float(box[4]) >= 0.25:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cls = int(box[5])
                    conf = float(box[4])
                    label = names.get(cls, str(cls))
                    # Shorten label
                    if len(label) > 3:
                        if ' ' in label:
                            label = ''.join([w[0].upper() for w in label.split()])
                        else:
                            label = label[:3].upper()
                    short_label = f'{label} {conf:.1f}' if conf >= 0.1 else f'{label}'
                    plot_one_box((x1, y1, x2, y2), img_vis, 
                                label=short_label, 
                                color=(255, 0, 0), line_thickness=2)
        
        ax.imshow(img_vis)
        model_name = result.get('model_name', f'Model {idx}')
        title = f"{model_name}\nmAP@0.5: {metrics.get('map50', 0):.3f}\nIoU: {metrics.get('mean_iou', 0):.3f}"
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

