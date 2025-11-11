# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import cv2

from models.experimental import attempt_load
from utils.datasets import create_dataloader, norm_imgs
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr, \
    target2mask, check_mask, clip_coords
from utils.metrics import ap_per_class, ConfusionMatrix, cluster_recall, sparse_recall, mask_pr, hm_verbose
from utils.plots import plot_images, plot_image, output_to_target, plot_study_txt, LatencyBucket
from utils.torch_utils import select_device, time_synchronized, model_info
from utils.loss import ComputeLoss
from utils.results_saver import (create_results_structure, save_image_result, 
                                compute_image_metrics, save_quantitative_csv, save_top5_results)

lbucket = LatencyBucket()

@torch.no_grad()
def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         is_coco=False,
         use_gt=False,  # ESOD default: Normal inference uses predicted masks, not GT
         sparse_head=False,
         hm_metric=False,
         opt=None):
    # Initialize/load model and set device
    training = model is not None
    
    # Extract dataset and model names for structured saving
    dataset_name = 'unknown'
    model_name = 'unknown'
    structured_results = None
    
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
        # Extract dataset name from data path
        if isinstance(data, str):
            dataset_name = Path(data).stem.replace('.yaml', '').replace('data/', '')
        elif isinstance(data, dict):
            dataset_name = opt.data.split('/')[-1].replace('.yaml', '') if opt and hasattr(opt, 'data') else 'unknown'
        
        # Extract model name from weights or experiment name
        if weights and isinstance(weights, (str, list)):
            weights_str = weights[0] if isinstance(weights, list) else weights
            if 'tinyperson' in weights_str:
                if 'base' in weights_str:
                    model_name = 'tinyperson_yolov5m_base'
                elif 'esod' in weights_str:
                    model_name = 'tinyperson_yolov5m_esod'
                elif 'uncertainty' in weights_str:
                    model_name = 'tinyperson_yolov5m_uncertainty'
                else:
                    model_name = Path(weights_str).parent.parent.name if 'weights' in weights_str else Path(weights_str).stem
            else:
                model_name = Path(weights_str).parent.parent.name if 'weights' in weights_str else Path(weights_str).stem
        elif opt and hasattr(opt, 'name'):
            model_name = opt.name
        
        # Create structured results directory
        if not training:  # Only for direct testing, not during training
            structured_results = create_results_structure(Path('.'), dataset_name, model_name)
            print(f'Structured results will be saved to: {structured_results["root"]}')

        # Load model
        if len(weights) == 1 and weights[0].endswith('.yaml'):
            from models.yolo import Model
            model = Model(weights[0], ch=3, nc=10).to(device).float().fuse()
        else:
            model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size TODO: 32 or 64
        if opt.compute_loss:
            compute_loss = ComputeLoss(model)

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    if sparse_head:
        model.model[-1].set_sparse()
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.safe_load(f)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training and dataloader is None:
        if device.type != 'cpu' and False:
            model(torch.zeros(1, 3, imgsz//2, imgsz//2).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, hyp=None, augment=False, pad=0., rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%11s' * 8) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'BPR', 'Occupy')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(6, device=device)
    statistic_items = torch.zeros(3, device=device, dtype=torch.float32)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    sp_r, m_p, m_r, attr = [], [], [], []
    gflops, infer_times = [], []
    
    # For structured results saving
    image_results_list = []  # Store per-image results for top-5 and CSV
    
    for batch_i, (img, targets, masks, m_weights, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
    # for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img_orig = img.clone()  # Keep original for visualization
        img = img.half() if half else img.float()  # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = norm_imgs(img, model)
        masks = masks.to(device, non_blocking=True)
        masks = masks.half() if half else masks.float()
        m_weights = m_weights.to(device, non_blocking=True)
        m_weights = m_weights.half() if half else m_weights.float()
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        # Debug: Check mask values before model forward
        # if batch_i == 0 and seen == 0 and not training:
        #     print(f"\nDEBUG: Input masks stats:")
        #     print(f"  masks.shape: {masks.shape if masks is not None else 'None'}")
        #     if masks is not None:
        #         print(f"  masks stats: min={masks.min().item():.6f}, max={masks.max().item():.6f}, mean={masks.mean().item():.6f}")
        #         print(f"  masks non-zero pixels: {(masks > 0).sum().item()}/{masks.numel()}")
        #         if masks.dim() == 4:
        #             for bi in range(min(4, masks.shape[0])):
        #                 mask_bi = masks[bi, 0] if masks.shape[1] > 0 else masks[bi]
        #                 print(f"    Image {bi}: min={mask_bi.min().item():.6f}, max={mask_bi.max().item():.6f}, non-zero={(mask_bi > 0).sum().item()}")
        
        # Run model
        if not training and opt.task == 'measure':
            assert img.shape[0] == 1  # bs=1

            try:
                from fvcore.nn import FlopCountAnalysis
                fca = FlopCountAnalysis(model, inputs=(img,))
                fca.unsupported_ops_warnings(False)
                gflop = fca.total() / 1E9 * 2
            except:
                gflop = model_info(model, inputs=(img,))

            gflops.append(gflop)

        t = time_synchronized()
        # ESOD inference: 
        # - During normal inference (use_gt=False): Model uses predicted masks from Segmenter
        # - During evaluation/analysis (use_gt=True): Model uses ground truth masks if valid, otherwise predicted masks
        # According to official ESOD, normal inference should NOT use GT masks
        if use_gt:
            # Use GT masks for evaluation/analysis if they exist and are valid (non-zero)
            # If GT masks are all zeros (missing mask files), fall back to predicted masks
            if masks is not None and masks.numel() > 0:
                # Check if masks have meaningful values (not all zeros)
                if masks.max() > 0.001:
                    model_input = (img, [masks])
                else:
                    # GT masks are all zeros (missing mask files), use predicted masks instead
                    model_input = img
            else:
                # No masks provided, use predicted masks
                model_input = img
        else:
            # Normal inference: Model uses its own predicted masks
            model_input = img
        
        (out, p_det), p_seg = model(model_input, augment=augment)  # inference and training outputs
        infer_times.append(time_synchronized() - t)
        t0 += infer_times[-1]
        train_out = (p_det, p_seg)
        
        # Save visualization of masks if requested
        if not training and opt.visualize_masks and batch_i < 3:
            try:
                import cv2
                vis_dir = save_dir / 'mask_visualizations'
                vis_dir.mkdir(exist_ok=True)
                
                # Visualize ground truth masks
                if masks is not None and masks.shape[1] > 0:
                    for si in range(min(nb, masks.shape[0])):
                        gt_mask = masks[si, 0].cpu().numpy() if masks.dim() == 4 else masks[si].cpu().numpy()
                        gt_mask_norm = (gt_mask * 255).astype(np.uint8)
                        cv2.imwrite(str(vis_dir / f'batch{batch_i}_img{si}_gt_mask.jpg'), gt_mask_norm)
                
                # Visualize predicted masks
                if p_seg is not None and len(p_seg) > 0:
                    pred_mask_list = p_seg[0] if isinstance(p_seg, list) else p_seg
                    if isinstance(pred_mask_list, list) and len(pred_mask_list) > 0:
                        pred_mask = pred_mask_list[0]
                        if isinstance(pred_mask, torch.Tensor):
                            pred_mask_sig = torch.sigmoid(pred_mask).cpu().numpy()
                            for si in range(min(nb, pred_mask_sig.shape[0])):
                                pm = pred_mask_sig[si, 0] if pred_mask_sig.ndim == 4 else pred_mask_sig[si]
                                pm_norm = (pm * 255).astype(np.uint8)
                                cv2.imwrite(str(vis_dir / f'batch{batch_i}_img{si}_pred_mask.jpg'), pm_norm)
            except Exception as e:
                print(f"  Warning: Could not save mask visualizations: {e}")
        
        # Debug: Check model output before NMS (first batch only)
        # if batch_i == 0 and seen == 0:
        #     print(f"\nDEBUG: Model output before NMS:")
        #     if out is not None:
        #         if isinstance(out, torch.Tensor):
        #             print(f"  out is Tensor, shape={out.shape}, dtype={out.dtype}")
        #             # Check predictions per scale
        #             if len(out.shape) > 1:
        #                 print(f"  out[0] shape: {out[0].shape if hasattr(out[0], 'shape') else 'N/A'}")
        #                 # Check if there are any predictions with confidence > 0
        #                 if isinstance(out, list):
        #                     for i, o in enumerate(out):
        #                         if o is not None and len(o) > 0:
        #                             print(f"  out[{i}] shape: {o.shape}, max_conf: {o[:, 4].max().item() if len(o) > 0 and o.shape[1] > 4 else 'N/A'}")
        #                 else:
        #                     # Single tensor
        #                     if out.numel() > 0:
        #                         print(f"  out max confidence: {out[..., 4].max().item() if out.shape[-1] > 4 else 'N/A'}")
        #                         print(f"  out num predictions: {out.shape[0] if len(out.shape) > 0 else 0}")
        #         elif isinstance(out, list):
        #             print(f"  out is list, len={len(out)}")
        #             for i, o in enumerate(out):
        #                 if o is not None:
        #                     print(f"  out[{i}] shape: {o.shape if hasattr(o, 'shape') else type(o)}, dtype: {o.dtype if hasattr(o, 'dtype') else 'N/A'}")
        #                     if hasattr(o, 'shape') and len(o) > 0 and o.shape[-1] > 4:
        #                         print(f"    max conf: {o[..., 4].max().item()}")
        #         else:
        #             print(f"  out type: {type(out)}")
        #     else:
        #         print(f"  out is None!")
        if hm_metric:
            mask_pr(masks, p_seg[0], targets, m_p, m_r)
            sparse_recall(p_seg[0], targets, sp_r)
            attr.append(targets[:, [1,4,5]])  # class, width, height

        # Compute loss
        if compute_loss:
            # loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls
            loss += compute_loss(train_out, targets, imgsz=img.shape, masks=masks, m_weights=m_weights)[1][:6]  # box, obj, cls, pixl, area, dist

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        # targets = targets[targets[:, 4] * targets[:, 5] < 32 * 32]  # TODO: small objects
        
        # Debug: Check targets before processing (first batch)
        # if batch_i == 0 and seen == 0:
        #     print(f"\nDEBUG: Batch 0 - targets.shape={targets.shape}, nb={nb}, targets[:, 0].unique()={targets[:, 0].unique() if len(targets) > 0 else 'empty'}")
        #     if len(targets) > 0:
        #         print(f"DEBUG: targets sample: {targets[:min(3, len(targets))]}")
        
        if isinstance(p_det, tuple):
            clusters = p_det[1][0] if p_det is not None else torch.zeros((0, 5), device=device)
            clusters = [clusters[clusters[:, 0] == bi, 1:] for bi in range(nb)]
            statistic_items += cluster_recall(clusters, targets, imgsz=(width, height), mode='bbox')
            if not training and opt.task == 'measure':
                lbucket.add(len(clusters[0]), gflops[-1], infer_times[-1])
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        if out is not None:
            if isinstance(out, torch.Tensor):
                # Debug before NMS
                # if batch_i == 0 and seen == 0:
                #     print(f"\nDEBUG: Before NMS - out shape: {out.shape}, conf_thres={conf_thres}, iou_thres={iou_thres}")
                #     if out.numel() > 0:
                #         print(f"  Total predictions before NMS: {out.shape[0]}")
                #         print(f"  Confidence range: [{out[..., 4].min().item():.6f}, {out[..., 4].max().item():.6f}]")
                
                t = time_synchronized()
                out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
                t1 += time_synchronized() - t
                
                # Debug after NMS
                # if batch_i == 0 and seen == 0:
                #     print(f"\nDEBUG: After NMS:")
                #     if isinstance(out, list):
                #         total_preds = sum(len(p) for p in out if p is not None)
                #         print(f"  out is list, len={len(out)}, total predictions: {total_preds}")
                #         for i, p in enumerate(out):
                #             if p is not None and len(p) > 0:
                #                 print(f"  out[{i}] shape: {p.shape}, conf range: [{p[:, 4].min().item():.6f}, {p[:, 4].max().item():.6f}]")
                #     else:
                #         print(f"  out type after NMS: {type(out)}")
        else:
            out = [torch.zeros((0, 6), device=device)] * nb
            # if batch_i == 0 and seen == 0:
            #     print(f"\nDEBUG: out was None, created empty predictions list")

        # Statistics per image
        for si, pred in enumerate(out):
            # Debug: Check predictions per image (first batch)
            # if batch_i == 0 and si < 2:
            #     print(f"\nDEBUG: Image {si} predictions - len(pred)={len(pred) if pred is not None else 0}")
            #     if pred is not None and len(pred) > 0:
            #         print(f"  pred shape: {pred.shape}, conf range: [{pred[:, 4].min().item():.6f}, {pred[:, 4].max().item():.6f}]")
            
            # Extract labels for this image - handle both empty targets and mismatched indices
            if len(targets) > 0:
                # Filter targets by batch index
                mask = targets[:, 0] == si
                labels = targets[mask, 1:] if mask.any() else torch.zeros((0, 5), device=targets.device, dtype=targets.dtype)
            else:
                labels = torch.zeros((0, 5), device=device)
            
            # Debug: Check label extraction (first few images)
            # if batch_i == 0 and seen < 3:
            #     print(f"DEBUG: Image {seen} (si={si}) - labels.shape={labels.shape}, nl={len(labels)}, targets.shape={targets.shape}")
            #     if len(targets) > 0:
            #         matching_targets = targets[targets[:, 0] == si]
            #         unique_indices = targets[:, 0].unique() if len(targets) > 0 else torch.tensor([])
            #         print(f"DEBUG: Matching targets for si={si}: {matching_targets.shape}, unique batch indices in targets: {unique_indices.tolist()}")
            
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            
            # Debug: Track tcls for all images (first batch only)
            # if batch_i == 0 and seen < 10:
            #     print(f"DEBUG: Image {seen} - nl={nl}, tcls={tcls[:10] if len(tcls) > 10 else tcls} (showing first 10)")
            
            path = Path(paths[si])
            seen += 1
            
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                elif not training:
                    # Even if no predictions and no labels, we should still record this for proper counting
                    # But only during validation to avoid cluttering stats
                    pass
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    filename = (path.stem + '.txt') if 'uavdt' not in opt.data else \
                        (path.parent.stem + '_' + path.stem + '.txt')
                    with open(save_dir / 'labels' / filename, 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                # Labels from dataset are in pixel xywh format relative to letterboxed image (model input size)
                # Check if labels are normalized (0-1) or pixel coordinates
                # If max value > 1, they're in pixel coordinates; otherwise normalized
                label_max = labels[:, 1:5].max().item() if nl > 0 else 0
                if label_max > 1.0:
                    # Labels are in pixel coordinates - convert xywh to xyxy directly
                    tbox = xywh2xyxy(labels[:, 1:5])  # pixel xyxy in model input space
                else:
                    # Labels are normalized - convert from normalized xywh to pixel xyxy
                    h_model, w_model = img[si].shape[1:]  # model input size (2048)
                    tbox = xywhn2xyxy(labels[:, 1:5], w_model, h_model)  # pixel xyxy in model input space
                # Now scale from model input space to original image space
                # shapes[si] = ((h0, w0), ((h/h0, w/w0), pad))
                # shapes[si][0] = (h0, w0) - original image dimensions
                # shapes[si][1] = ((h/h0, w/w0), pad) - ratio and padding for scale_coords
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            
            # Store per-image results for structured saving (only if not training)
            if not training and structured_results is not None:
                # Get original image - img_orig is before normalization
                orig_img = img_orig[si].cpu().numpy()
                
                # Handle different image formats
                if orig_img.shape[0] == 3:  # CHW -> HWC
                    orig_img = orig_img.transpose(1, 2, 0)
                
                # Ensure uint8 format
                if orig_img.dtype != np.uint8:
                    if orig_img.max() <= 1.0:
                        orig_img = (orig_img * 255).astype(np.uint8)
                    else:
                        orig_img = orig_img.astype(np.uint8)
                
                # Prepare GT boxes
                gt_boxes_for_vis = None
                if nl > 0:
                    # Convert normalized targets to absolute coordinates
                    tbox_abs = tbox.clone()
                    # tbox is already in absolute coordinates after scale_coords
                    # Clip GT boxes to image boundaries
                    img_h, img_w = orig_img.shape[:2]
                    clip_coords(tbox_abs, (img_h, img_w))
                    
                    gt_boxes_for_vis = torch.cat([
                        tbox_abs,
                        torch.ones(len(tbox_abs), 1, device=device),  # conf = 1.0
                        tcls_tensor.float().unsqueeze(1)
                    ], dim=1)
                
                # Prepare prediction boxes
                pred_boxes_for_vis = None
                if len(predn) > 0:
                    pred_boxes_for_vis = predn.clone()
                
                # Compute per-image metrics
                # DEBUG: Print box info for first image only
                if seen <= 1 and pred_boxes_for_vis is not None and gt_boxes_for_vis is not None:
                    print(f"\n{'='*80}")
                    print(f"DEBUG: Image {seen} - Boxes passed to compute_image_metrics")
                    print(f"{'='*80}")
                    print(f"Pred boxes: {len(pred_boxes_for_vis)} boxes")
                    if len(pred_boxes_for_vis) > 0:
                        print(f"  First pred: [{pred_boxes_for_vis[0, 0]:.1f}, {pred_boxes_for_vis[0, 1]:.1f}, "
                              f"{pred_boxes_for_vis[0, 2]:.1f}, {pred_boxes_for_vis[0, 3]:.1f}], "
                              f"conf={pred_boxes_for_vis[0, 4]:.4f}")
                        print(f"  Pred range: X=[{pred_boxes_for_vis[:, 0].min():.1f}, {pred_boxes_for_vis[:, 2].max():.1f}], "
                              f"Y=[{pred_boxes_for_vis[:, 1].min():.1f}, {pred_boxes_for_vis[:, 3].max():.1f}]")
                    print(f"GT boxes: {len(gt_boxes_for_vis)} boxes")
                    if len(gt_boxes_for_vis) > 0:
                        print(f"  First GT: [{gt_boxes_for_vis[0, 0]:.1f}, {gt_boxes_for_vis[0, 1]:.1f}, "
                              f"{gt_boxes_for_vis[0, 2]:.1f}, {gt_boxes_for_vis[0, 3]:.1f}], "
                              f"conf={gt_boxes_for_vis[0, 4]:.1f}")
                        print(f"  GT range: X=[{gt_boxes_for_vis[:, 0].min():.1f}, {gt_boxes_for_vis[:, 2].max():.1f}], "
                              f"Y=[{gt_boxes_for_vis[:, 1].min():.1f}, {gt_boxes_for_vis[:, 3].max():.1f}]")
                    # Compute IoU directly (box_iou already imported at top)
                    ious_direct = box_iou(pred_boxes_for_vis[:, :4], gt_boxes_for_vis[:, :4])
                    max_ious_direct, _ = ious_direct.max(dim=1)
                    print(f"  Direct IoU: best={max_ious_direct.max():.4f}, mean={max_ious_direct.mean():.4f}")
                    print(f"{'='*80}\n")
                
                img_metrics = compute_image_metrics(pred_boxes_for_vis, gt_boxes_for_vis, iouv, conf_thres)
                img_metrics['image_path'] = str(path)
                img_metrics['precision'] = img_metrics['num_tp'] / (img_metrics['num_tp'] + img_metrics['num_fp']) if (img_metrics['num_tp'] + img_metrics['num_fp']) > 0 else 0.0
                img_metrics['recall'] = img_metrics['num_tp'] / (img_metrics['num_tp'] + img_metrics['num_fn']) if (img_metrics['num_tp'] + img_metrics['num_fn']) > 0 else 0.0
                
                image_results_list.append({
                    'image': orig_img,
                    'pred_boxes': pred_boxes_for_vis,
                    'gt_boxes': gt_boxes_for_vis,
                    'image_path': str(path),
                    'metrics': img_metrics
                })

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            pred_targets = output_to_target(out)
            Thread(target=plot_images, args=(img, pred_targets, paths, f, names), daemon=True).start()
            
            # Save individual images
            individual_dir = save_dir / 'individual_images'
            individual_dir.mkdir(exist_ok=True)
            
            # Convert targets and predictions to numpy for easier indexing
            # targets is a CUDA tensor, need to move to CPU first
            if len(targets) > 0:
                targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
            else:
                targets_np = np.array([]).reshape(0, 7)
            
            # pred_targets is already a list from output_to_target, convert to numpy
            if len(pred_targets) > 0:
                pred_targets_np = np.array(pred_targets)
            else:
                pred_targets_np = np.array([]).reshape(0, 7)
            
            # Process each image in the batch
            individual_threads = []
            for i in range(len(paths)):
                img_name = Path(paths[i]).stem
                
                # Get targets for this image
                img_targets = np.array([]).reshape(0, 6)
                if len(targets_np) > 0:
                    img_targets_batch = targets_np[targets_np[:, 0] == i]
                    if len(img_targets_batch) > 0:
                        # Remove batch index column (first column)
                        img_targets = img_targets_batch[:, 1:]
                
                # Get predictions for this image
                img_preds = np.array([]).reshape(0, 7)
                if len(pred_targets_np) > 0:
                    img_preds_batch = pred_targets_np[pred_targets_np[:, 0] == i]
                    if len(img_preds_batch) > 0:
                        # Remove batch index column (first column)
                        img_preds = img_preds_batch[:, 1:]
                
                # Save individual label image
                f_label = individual_dir / f'{img_name}_labels.jpg'
                t_label = Thread(target=plot_image, args=(img[i:i+1], img_targets, paths[i], str(f_label), names), daemon=False)
                t_label.start()
                individual_threads.append(t_label)
                
                # Save individual prediction image
                f_pred = individual_dir / f'{img_name}_pred.jpg'
                t_pred = Thread(target=plot_image, args=(img[i:i+1], img_preds, paths[i], str(f_pred), names), daemon=False)
                t_pred.start()
                individual_threads.append(t_pred)
            
            # Wait for all individual image threads to complete
            for t in individual_threads:
                t.join(timeout=10)  # Wait up to 10 seconds per thread

    # Compute statistics
    # Debug: Check stats before concatenation
    # if len(stats) > 0:
    #     print(f"\nDEBUG: Before concatenation - len(stats)={len(stats)}, number of tuples={len(stats[0]) if len(stats) > 0 and len(stats[0]) > 0 else 0}")
    #     if len(stats) > 0 and len(stats[0]) > 0:
    #         first_tuple = stats[0][0]
    #         print(f"DEBUG: First stats tuple structure: len={len(first_tuple)}, types={[type(x) for x in first_tuple]}")
    #         if len(first_tuple) > 3:
    #             tcls_val = first_tuple[3]
    #             if isinstance(tcls_val, list):
    #                 print(f"DEBUG: First tuple stats[3] (tcls) is list, len={len(tcls_val)}, first 10: {tcls_val[:10]}")
    #             else:
    #                 print(f"DEBUG: First tuple stats[3] (tcls) type: {type(tcls_val)}, value: {tcls_val}")
    
    # Convert to numpy - handle lists and empty tensors properly
    stats_processed = []
    for i, x in enumerate(zip(*stats)):
        # Convert each element to numpy array
        x_np = []
        for item in x:
            if isinstance(item, list):
                # Convert list to numpy array
                if len(item) > 0:
                    x_np.append(np.array(item))
                # Skip empty lists - they'll be handled by concatenating empty arrays
            else:
                # Tensor or numpy array - convert to numpy
                if hasattr(item, 'cpu'):
                    # It's a tensor
                    item_np = item.cpu().numpy()
                    if item_np.size > 0:  # Only add non-empty arrays
                        x_np.append(item_np)
                elif hasattr(item, 'shape'):
                    # It's already a numpy array
                    if item.size > 0:  # Only add non-empty arrays
                        x_np.append(item)
                else:
                    # Try to convert to numpy
                    item_np = np.array(item)
                    if item_np.size > 0:
                        x_np.append(item_np)
        
        # Concatenate - handle empty case
        if len(x_np) > 0:
            stats_processed.append(np.concatenate(x_np, 0))
        else:
            # All elements were empty - create empty array with appropriate dtype
            # For stats[0] (correct): bool, stats[1] (conf): float, stats[2] (pcls): float, stats[3] (tcls): int
            if i == 0:
                stats_processed.append(np.array([], dtype=bool))
            elif i == 3:
                stats_processed.append(np.array([], dtype=int))
            else:
                stats_processed.append(np.array([], dtype=float))
    stats = stats_processed
    
    # Debug: Check stats after concatenation (ALWAYS show for debugging)
    # if len(stats) > 0:
    #     print(f"\nDEBUG: After concatenation - stats lengths: {[len(s) if hasattr(s, '__len__') else 'N/A' for s in stats]}")
    #     if len(stats) > 3:
    #         print(f"DEBUG: stats[3] (tcls) shape: {stats[3].shape if hasattr(stats[3], 'shape') else type(stats[3])}")
    #         print(f"DEBUG: stats[3] (tcls) content (first 20): {stats[3][:20] if len(stats[3]) > 0 else 'EMPTY'}")
    #         print(f"DEBUG: stats[3] unique values: {np.unique(stats[3]) if len(stats[3]) > 0 else 'EMPTY'}")
    #     else:
    #         print(f"DEBUG: stats has less than 4 elements, len(stats)={len(stats)}")
    
    # Calculate number of targets (nt) from stats[3] regardless of predictions
    if len(stats) > 3 and len(stats[3]) > 0:
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        # print(f"DEBUG: After bincount - nt={nt}, nt.sum()={nt.sum()}")
    else:
        nt = np.zeros(nc, dtype=np.int64)
        # print(f"DEBUG: stats[3] is empty or doesn't exist - setting nt to zeros. len(stats)={len(stats)}, stats[3] exists: {len(stats) > 3 if len(stats) > 3 else False}")
    
    # Compute metrics if we have predictions
    if len(stats) and len(stats[0]) > 0 and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        # No predictions but we still have labels count
        p, r, ap50, ap = np.zeros(nc), np.zeros(nc), np.zeros(nc), np.zeros(nc)
        mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
        # print(f"DEBUG: No predictions found - stats[0] length={len(stats[0]) if len(stats) > 0 else 0}, stats[0].any()={stats[0].any() if len(stats) > 0 and len(stats[0]) > 0 else 'N/A'}")
    bpr, occupy = statistic_items[0] / nt.sum(), (statistic_items[1] + 1e-6) / (statistic_items[2] + 1e-6)
    if hm_metric:
        sp_r, m_p, m_r = torch.stack(sp_r), torch.stack(m_p), torch.stack(m_r)
        mp, mr, bpr = m_p.mean().item(), m_r.mean().item(), sp_r.mean().item()
        print('\n'.join([str(res) for res in hm_verbose(sp_r, torch.cat(attr))]))


    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map, bpr, occupy))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats) and False:
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], bpr, occupy))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        if opt.task == 'measure':
            print('GFLOPs: %.1f. FPS: %.1f ' % (np.mean(gflops), 1000. / t[0]), end='')
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        if 'visdrone' in opt.data.lower():
            anno_json = './datasets/VisDrone/annotations/val.json'  # annotations json
        elif 'tinyperson' in opt.data.lower():
            anno_json = './datasets/TinyPerson/test.json'  # annotations json
            format_tinyperson(jdict)
        else:
            raise NotImplementedError(f'Ground-Truth file for {os.path.basename(opt.data)} not found.')

        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        
            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            # map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: \n{e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
        lbucket.save(f'{save_dir}/buckets.json')
        
        # Save structured results (qualitative top-5 and quantitative CSV)
        if structured_results is not None and image_results_list:
            print(f'\nSaving structured results to {structured_results["root"]}...')
            
            # Save quantitative CSV for all images
            save_quantitative_csv(image_results_list, structured_results['quantitative_csv'])
            print(f'  ✓ Quantitative metrics saved: {structured_results["quantitative_csv"]}')
            
            # Save top 5 best results (qualitative)
            # Use higher confidence threshold (0.25) for visualization to reduce clutter
            # Evaluation still uses low threshold (0.001) for comprehensive metrics
            vis_conf_thres = max(0.10, conf_thres)  # Use at least 0.25 for visualization
            if len(image_results_list) >= 5:
                save_top5_results(image_results_list, structured_results['qualitative_top5'], names, vis_conf_thres=vis_conf_thres)
                print(f'  ✓ Top 5 qualitative results saved: {structured_results["qualitative_top5"]}')
            else:
                print(f'  ⚠ Less than 5 images, saving all {len(image_results_list)} as qualitative results')
                save_top5_results(image_results_list, structured_results['qualitative_top5'], names, vis_conf_thres=vis_conf_thres)
            
            print(f'  ✓ Structured results saved successfully!')
    
    # Save overall test metrics (AP, AP50, GFLOPs, FPS) to summary file
    if structured_results is not None:
        import json
        mean_gflops = np.mean(gflops) if len(gflops) > 0 else 0.0
        fps = 1000.0 / t[0] if t[0] > 0 else 0.0
        
        summary_metrics = {
            'AP50': float(map50),
            'AP': float(map),
            'GFLOPs': float(mean_gflops),
            'FPS': float(fps),
            'Precision': float(mp),
            'Recall': float(mr),
        }
        
        summary_file = structured_results['root'] / 'test_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary_metrics, f, indent=2)
        print(f'  ✓ Test summary saved: {summary_file}')
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    
    # Calculate GFLOPs and FPS for comparison
    mean_gflops = np.mean(gflops) if len(gflops) > 0 else 0.0
    fps = 1000.0 / t[0] if t[0] > 0 else 0.0  # FPS = 1000ms / inference_time_ms
    
    # Return results with GFLOPs and FPS appended
    # Format: (mp, mr, map50, map, bpr, occupy, loss_values..., gflops, fps)
    return (mp, mr, map50, map, bpr, occupy, *(loss.cpu() / len(dataloader)).tolist(), mean_gflops, fps), maps, t


def format_tinyperson(jdict):
    with open('datasets/TinyPerson/mini_annotations/tiny_set_test_all.json', 'r') as f:
        anno = json.load(f)
    image_id_dict = {item['file_name'].split('/')[1][:-4]: item['id'] for item in anno['images']}

    for item in jdict:
        item['image_id'] = image_id_dict[item['image_id']]
        item['category_id'] += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py', description='YOLOv5 Testing')
    parser.add_argument('mode', nargs='?', type=str, default='test', choices=['test'], 
                       help='Testing mode (default: test)')
    
    # Default values for VisDrone Uncertainty Model
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/visdrone_yolov5m_uncertainty/weights/best.pt', 
                       help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/visdrone.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=4, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=1536, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='visdrone_esod', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--use-gt', action='store_true', help='use gt masks')
    parser.add_argument('--compute-loss', action='store_true', help='compute loss')
    parser.add_argument('--sparse-head', action='store_true', help='use sparse detection head')
    parser.add_argument('--hm-metric', action='store_true', help='use heatmap-related evaluation metrics')
    parser.add_argument('--visualize-masks', action='store_true', help='save objectness masks for visualization')
    parser.add_argument('--save-debug', action='store_true', help='save debug outputs (masks, patches, clusters)')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.task in ('train', 'val', 'test', 'measure'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             half_precision=opt.half,
             use_gt=opt.use_gt,
             sparse_head=opt.sparse_head,
             hm_metric=opt.hm_metric,
             opt=opt
             )

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        # x = list(range(832, 1920 + 128, 128))  # x axis (image sizes)
        x = list(range(1024, 2560 + 128, 128))  # x axis (image sizes)
        opt.task = 'measure'
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               sparse_head=opt.sparse_head, use_gt=opt.use_gt, half_precision=opt.half, plots=False, opt=opt)
                y.append((r + t).cpu().numpy())  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
