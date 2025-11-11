# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os
import os.path as osp
from os.path import join, exists, isdir, basename, abspath
from glob import glob
from tqdm import tqdm
import random
import cv2
import numpy as np
import json
import warnings
import math

import torch
import torch.nn.functional as F

import sys; sys.path.append('./')
from utils.general import gaussian2D

try:
    from segment_anything import SamPredictor, sam_model_registry

    sam_checkpoint = "./weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device) #.half()  Warning: Precision Drops
    dtype = next(sam.named_parameters())[1].dtype
    predictor = SamPredictor(sam)
except:
    warnings.warn('It is recommended to install segment-anything for better pseudo masks. See instructions in README.md.')
    predictor = None


############ utils ############

def _readlines(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    
    return lines


def check_break(act):
    flag = False
    prev_pos, prev_neg = False, False
    for x in act:
        if x:
            if not prev_pos:
                prev_pos = True
            elif prev_neg:
                flag = True
                break
        elif prev_pos:
            prev_neg = True
    
    return flag


def check_center(crop):
    h, w = crop.shape
    indices = torch.nonzero(crop)
    yc, xc = indices.float().mean(dim=0)
    
    s = 0.15
    return ((yc - h/2.).abs() > h * s) | ((xc - w/2.).abs() > w * s)


def segment_image(image, labels, width, height):
    if len(labels) == 0:
        return torch.zeros(image.shape[:2], dtype=torch.float16).to(device), np.full((0,), False)

    if max(width, height) > 1024:
        mask = torch.zeros((height, width), dtype=torch.float16).to(device)
        invalid = np.full((len(labels),), False)

        # overlap = 20  # pixel
        nx, ny = math.ceil(width / 1024), math.ceil(height / 1024)
        width_, height_ = width // nx, height // ny
        xc, yc, w, h = labels[:, -4:].T
        x1, y1, x2, y2 = xc - w / 2., yc - h / 2., xc + w / 2., yc + h / 2.
        for j in range(ny):
            for i in range(nx):
                grid = np.array([i / nx, j / ny, (i+1) / nx, (j+1) / ny], dtype=labels.dtype)
                indices = (grid[0] < x2) & (x1 < grid[2]) & (grid[1] < y2) & (y1 < grid[3])
                if indices.sum() == 0:
                    continue
                
                x1_, y1_, x2_, y2_ = (x1[indices] - grid[0]).clip(0, 1/nx), \
                                        (y1[indices] - grid[1]).clip(0, 1/ny), \
                                        (x2[indices] - grid[0]).clip(0, 1/nx), \
                                        (y2[indices] - grid[1]).clip(0, 1/ny)
                xc_, yc_, w_, h_ = (x1_ + x2_) / 2., (y1_ + y2_) / 2., (x2_ - x1_), (y2_ - y1_)
                labels_ = np.stack((labels[indices, 0], xc_, yc_, w_, h_), axis=1)

                x1_, y1_, x2_, y2_ = width_*i, height_*j, width_*(i+1), height_*(j+1)
                mask_k, invalid_k = segment_image(image[y1_:y2_, x1_:x2_], labels_, width_, height_)
                mask[y1_:y2_, x1_:x2_] = mask_k
                invalid[indices] |= invalid_k

        return mask, invalid
    
    c, xc, yc, w, h = labels.T
    x1, y1, x2, y2 = (xc - w / 2.) * width, (yc - h / 2.) * height, \
                    (xc + w / 2.) * width, (yc + h / 2.) * height
    input_boxes = np.stack((x1, y1, x2, y2), axis=1)
    input_boxes = torch.from_numpy(input_boxes).long().to(device)

    predictor.set_image(image)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2]).to(dtype)
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
        return_logits=True
    )
    # (batch_size) x (num_predicted_masks_per_input=1) x H x W
    mask = masks.sigmoid().squeeze(1).max(dim=0)[0].half()
    
    invalid = np.full((len(masks),), False)
    for i, (x1, y1, x2, y2) in enumerate(input_boxes.cpu()):
        crop = mask[y1:y2, x1:x2] > 0.5
        invalid[i] = check_center(crop) | check_break(crop.sum(dim=1)) | check_break(crop.sum(dim=0))
    
    return mask, invalid


def gen_mask(label_path, image, cls_ratio=False, thresh=0.5, sam_only=False):
    if cls_ratio:
        cls_ratio = [1.83, 5.35, 13.82, 1.00, 5.80, 11.25, 30.11, 44.63, 24.45, 4.89]  # train set
    stride = 1
    # area_min, area_max = 4 * 4 * stride * stride, 6 * 6 * stride * stride
    area_min, area_max = 4 * 4 * 100, 6 * 6 * 50  # for 1920*1080
    min_size = 1e6

    # Handle both Windows and Unix paths
    label_dir = osp.dirname(label_path)
    label_name = osp.basename(label_path)
    # Replace 'labels' with 'masks' in the directory path (works for both / and \)
    mask_dir = label_dir.replace(os.sep + 'labels' + os.sep, os.sep + 'masks' + os.sep)
    mask_dir = mask_dir.replace('labels' + os.sep, 'masks' + os.sep)
    mask_dir = mask_dir.replace(os.sep + 'labels', os.sep + 'masks')
    # Fallback for edge cases
    if 'labels' in mask_dir:
        mask_dir = mask_dir.replace('labels', 'masks')
    save_path = osp.join(mask_dir, label_name.replace('.txt', '.npy'))
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    
    height, width, _ = image.shape
    nx, ny = width // stride, height // stride

    # Robustly read labels; avoid warnings on empty files
    if (not exists(label_path)) or (osp.getsize(label_path) == 0):
        labels = np.zeros((0, 5), dtype=float)
    else:
        try:
            labels = np.loadtxt(label_path, delimiter=' ')
            if labels.ndim == 1:
                labels = labels.reshape(1, -1)
            labels = labels.reshape(-1, 5)
        except Exception:
            labels = np.zeros((0, 5), dtype=float)

    mask = np.zeros((ny, nx), dtype=np.float16)
    weight = np.ones_like(mask)
    
    if predictor is not None and labels.size > 0:
        sam_res, invalid = segment_image(image, labels, width, height)
        if stride != 1:
            sam_res = F.interpolate(sam_res[None, None, ...].float(), size=(ny, nx), mode='bilinear', align_corners=False)[0, 0]
            # sam_res = F.interpolate(sam_res[None, None, ...].float(), size=(ny, nx), mode='nearest')[0, 0]
        sam_res = (sam_res > 0.5).half().numpy()

    if labels.size == 0:
        np.save(save_path, np.stack((mask, weight), axis=-1))
        return

    c, xc, yc, w, h = labels.T
    x1, y1, x2, y2 = ((xc - w / 2.) * nx).astype(np.int32).clip(0), \
                        ((yc - h / 2.) * ny).astype(np.int32).clip(0), \
                        np.ceil((xc + w / 2.) * nx).astype(np.int32).clip(0, nx), \
                        np.ceil((yc + h / 2.) * ny).astype(np.int32).clip(0, ny)
    input_boxes = np.stack((x1, y1, x2, y2), axis=1)
    
    for i, (x1, y1, x2, y2) in enumerate(input_boxes):
        w, h = x2 - x1, y2 - y1
        # Skip invalid bounding boxes with zero or negative dimensions
        if w <= 0 or h <= 0:
            continue
        gaussian = gaussian2D((h, w), sigma=None, thresh=thresh).astype(mask.dtype)
        
        if predictor is not None:
            sam_mask = sam_res[y1:y2, x1:x2].copy()
            if sam_only:
                gaussian = sam_mask
            else:
                if invalid[i] == 0 and sam_mask.sum() / (w * h) > 0.25:
                    gaussian *= sam_mask
                np.maximum(gaussian, sam_mask * thresh, out=gaussian)
        
        masked_hm = mask[y1:y2, x1:x2]
        np.maximum(masked_hm, gaussian, out=masked_hm)

        area = w * h / (width * height) * (1920 * 1080)
        # min_size = min(min_size, area)
        r_size = max(area_min / area, 1.) ** 2
        # elif area > area_max:
        #     r_size = area / area_max
        if cls_ratio:
            r_cls = cls_ratio[int(c[i])] ** 0.7
        else:
            r_cls = 1.0
        r = max(r_size, r_cls)
        masked_wt = weight[y1:y2, x1:x2]
        curr_wt = np.zeros_like(masked_wt) + math.log(r) + 1.
        curr_wt *= (gaussian > 0).astype(mask.dtype)
        np.maximum(masked_wt, curr_wt, out=masked_wt)
    
    np.save(save_path, np.stack((mask, weight), axis=-1))


############ scripts ############


def prepare_visdrone():
    
    name_dict = {'0': 'ignored regions', '1': 'pedestrian', '2': 'people',
                 '3': 'bicycle', '4': 'car', '5': 'van', '6': 'truck',
                 '7': 'tricycle', '8': 'awning-tricycle', '9': 'bus',
                 '10': 'motor', '11': 'others'}
    split_dict = {'test-dev': 'test-dev.txt', 'val': 'val.txt', 'train': 'train.txt'}
    root = opt.dataset

    os.makedirs(join(root, 'split'), exist_ok=True)
    for sub_dir in glob(join(root, 'VisDrone2019-DET-*')):
        os.makedirs(join(sub_dir, 'labels'), exist_ok=True)
        images = sorted(glob(join(sub_dir, 'images', '*.jpg')))
        if 'test-challenge' in sub_dir:
            with open(join(root, 'split', 'test-challenge.txt'), 'w+') as f:
                f.writelines([line + '\n' for line in images])
            continue

        data_paths = []
        for image_path in tqdm(images):
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            label_path = image_path.replace('images', 'annotations').replace('.jpg', '.txt')
            # if "masked" in label_path:  # avoid repeated processing
            #     continue
            assert exists(label_path)
            label_lines = []
            masked = False
            # <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
            for line in _readlines(label_path):
                if line[-1] == ',':
                    line = line[:-1]
                x1, y1, w, h, score, cls, truncation, occlusion = list(map(int, line.split(',')))
                if cls in [0, 11]:
                    image[y1:y1 + h, x1:x1 + w, :] = 85
                    masked = True
                elif truncation < 2 and (occlusion < 2 or True):
                    xc, yc = x1 + w / 2., y1 + h / 2.
                    label_lines.append(('%d' + ' %.6f' * 4 + '\n') %
                                       (cls - 1, xc / width, yc / height, w / width, h / height))
                else:
                    pass
            if masked:
                image_path = image_path.replace('.jpg', '_masked.jpg')
                cv2.imwrite(image_path, image)
            # for consistency
            label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
            with open(label_path, 'w+') as f:
                f.writelines(label_lines)

            gen_mask(label_path, image, cls_ratio=True)

            data_paths.append(image_path + '\n')
        
        with open(join(root, 'split', split_dict[basename(sub_dir)[17:]]), 'w+') as f:
            f.writelines(data_paths)


def prepare_uavdt():
    root = opt.dataset
    split_dir = join(root, 'split')

    # Branch 1: Official UAVDT structure expected by the original script
    data_dir = join(root, 'UAV-benchmark-M')
    attr_dir = join(root, 'M_attr')
    label_dir = join(root, 'UAV-benchmark-MOTD_v1.0', 'GT')
    has_official_layout = isdir(data_dir) and isdir(attr_dir) and isdir(label_dir)

    # Branch 2: Alt layout (DatasetNinja/Roboflow style): train/{img,ann}, test/{img,ann}
    alt_train_ann = join(root, 'train', 'ann')
    alt_train_img = join(root, 'train', 'img')
    alt_test_ann = join(root, 'test', 'ann')
    alt_test_img = join(root, 'test', 'img')
    has_alt_layout = isdir(alt_train_ann) and isdir(alt_train_img) and isdir(alt_test_ann) and isdir(alt_test_img)

    if has_official_layout:
        # Original pipeline
        # mask images, it takes minutes
        for i, path in enumerate(glob(join(label_dir, '*ignore.txt'))):
            labels = np.loadtxt(path, usecols=(0, 2, 3, 4, 5, 8), dtype=int, delimiter=',')
            vid_name = basename(path).split('_')[0]
            for frameID, x1, y1, w, h, _ in tqdm(labels, desc='%02d/50' % (i + 1)):
                masked_path = join(data_dir, vid_name, 'img%06d_masked.jpg' % frameID)
                input_path = masked_path if exists(masked_path) else masked_path.replace('_masked.jpg', '.jpg')
                image = cv2.imread(input_path)
                image[y1:y1 + h, x1:x1 + w] = (127, 127, 127)
                cv2.imwrite(masked_path, image)

        data_split = {}
        for mode in ['train', 'test']:
            data_split[mode] = []
            for path in glob(join(attr_dir, mode, '*.txt')):
                data_split[mode].append(basename(path)[:5])
        k = 10
        sep = len(data_split['train']) // k if len(data_split['train']) else 0
        random.shuffle(data_split['train'])
        data_split['train'], data_split['valid'] = data_split['train'][sep:], data_split['train'][:sep]

        os.makedirs(split_dir, exist_ok=True)
        for mode in ['train', 'valid', 'test']:
            with open(join(split_dir, '%s_video.txt' % mode), 'w+') as f:
                f.writelines(vid + '\n' for vid in data_split.get(mode, []))
            image_paths = []
            for video_name in tqdm(data_split.get(mode, []), desc=mode):
                ignore_path = join(label_dir, '%s_gt_ignore.txt' % video_name)
                label_path = join(label_dir, '%s_gt_whole.txt' % video_name)

                # the warning caused by empty file doesn't matter
                ignores = np.loadtxt(ignore_path, usecols=(0, 2, 3, 4, 5), dtype=int, delimiter=',')
                ignore_dict = {}
                for frameID, x1, y1, w, h in ignores:
                    xyxy = np.array([[x1, y1, x1 + w, y1 + h]])
                    if frameID in ignore_dict:
                        ignore_dict[frameID] = np.concatenate((ignore_dict[frameID], xyxy), axis=0)
                    else:
                        ignore_dict[frameID] = xyxy

                labels = np.loadtxt(label_path, usecols=(0, 2, 3, 4, 5, 8), dtype=int, delimiter=',')
                label_dict = {}
                for frameID, x1, y1, w, h, cls in labels:
                    xc, yc = x1 + w / 2., y1 + h / 2.
                    if frameID in ignore_dict:
                        ignore_regions = ignore_dict[frameID]
                        if np.logical_and(
                                np.logical_and(ignore_regions[:, 0] < xc, xc < ignore_regions[:, 2]),
                                np.logical_and(ignore_regions[:, 1] < yc, yc < ignore_regions[:, 3])
                        ).sum() > 0:
                            continue

                    box = [cls, xc, yc, w, h]
                    if frameID in label_dict:
                        label_dict[frameID].append(box)
                    else:
                        label_dict[frameID] = [box]

                for frameID, bboxes in label_dict.items():
                    image_path = join(data_dir, video_name, 'img%06d_masked.jpg' % frameID)
                    if not exists(image_path):
                        image_path = image_path.replace('_masked.jpg', '.jpg')

                    image = cv2.imread(image_path)
                    height, width, _ = image.shape
                    label_path = image_path.replace('.jpg', '.txt')
                    with open(label_path, 'w+') as f:
                        for cls, xc, yc, w, h in bboxes:
                            assert 1 <= cls <= 3  # cls - 1
                            # treat all categories to one, 'car'
                            f.write('%d %.6f %.6f %.6f %.6f\n' % (0, xc / width, yc / height, w / width, h / height))

                    gen_mask(label_path, image)
                    image_paths.append(image_path + '\n')

            with open(join(split_dir, '%s.txt' % mode), 'w+') as f:
                f.writelines(image_paths)
            if mode == 'train':
                with open(join(split_dir, '%s_ds.txt' % mode), 'w+') as f:
                    f.writelines(image_paths[::10])

        return

    if has_alt_layout:
        # Alt pipeline: parse per-image JSON annotations and build YOLO labels + splits
        cls_map = {'car': 0, 'truck': 1, 'bus': 2}

        def convert_split(split_name: str):
            img_dir = join(root, split_name, 'img')
            ann_dir = join(root, split_name, 'ann')
            image_paths = []
            for img_name in tqdm(sorted(os.listdir(img_dir)), desc=split_name):
                if not img_name.lower().endswith('.jpg'):
                    continue
                img_path = join(img_dir, img_name)
                ann_path = join(ann_dir, img_name + '.json')
                if not exists(ann_path):
                    # some tools drop extension in json name
                    alt_ann = join(ann_dir, img_name.replace('.jpg', '.json'))
                    if exists(alt_ann):
                        ann_path = alt_ann
                    else:
                        continue

                with open(ann_path, 'r') as f:
                    ann = json.load(f)
                width = ann['size']['width']
                height = ann['size']['height']
                labels = []
                for obj in ann.get('objects', []):
                    cls_title = obj.get('classTitle', '').lower()
                    if cls_title not in cls_map:
                        continue
                    pts = obj['points']['exterior']
                    if len(pts) < 2:
                        continue
                    (x1, y1), (x2, y2) = pts[0], pts[1]
                    w = max(0.0, float(x2) - float(x1))
                    h = max(0.0, float(y2) - float(y1))
                    if w <= 0 or h <= 0:
                        continue
                    xc = float(x1) + w / 2.0
                    yc = float(y1) + h / 2.0
                    labels.append((cls_map[cls_title], xc / width, yc / height, w / width, h / height))

                # write YOLO label next to image
                label_path = img_path.replace('.jpg', '.txt')
                with open(label_path, 'w+') as f:
                    for c, xc, yc, ww, hh in labels:
                        f.write(f"{c} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

                image = cv2.imread(img_path)
                if image is not None:
                    gen_mask(label_path, image)
                image_paths.append(img_path + '\n')

            return image_paths

        os.makedirs(split_dir, exist_ok=True)
        train_list = convert_split('train')
        test_list = convert_split('test')

        # 10% of train for valid
        k = 10
        random.shuffle(train_list)
        sep = len(train_list) // k if len(train_list) else 0
        valid_list = train_list[:sep]
        train_main = train_list[sep:]

        with open(join(split_dir, 'train.txt'), 'w+') as f:
            f.writelines(train_main)
        with open(join(split_dir, 'valid.txt'), 'w+') as f:
            f.writelines(valid_list)
        with open(join(split_dir, 'test.txt'), 'w+') as f:
            f.writelines(test_list)
        with open(join(split_dir, 'train_ds.txt'), 'w+') as f:
            f.writelines(train_main[::10])
        with open(join(split_dir, 'test_ds.txt'), 'w+') as f:
            f.writelines(test_list[::10])

        return

    # Fallback: nothing matched
    warnings.warn('UAVDT layout not recognized. Expected official directories or train/test with img+ann JSONs.')


def prepare_tinyperson():
    root = opt.dataset
    label_file_dict = {'train': join(root, 'mini_annotations', 'tiny_set_train_all_erase.json'),
                       'test': join(root, 'mini_annotations', 'tiny_set_test_all.json')}
    image_dir = join(root, 'erase_with_uncertain_dataset')
    split_dir = join(root, 'split')

    # Check if dataset is already in YOLO format (has train/valid/test directories)
    yolo_train_dir = join(root, 'train', 'images')
    yolo_valid_dir = join(root, 'valid', 'images')
    yolo_test_dir = join(root, 'test', 'images')
    has_yolo_format = isdir(yolo_train_dir) and isdir(yolo_valid_dir) and isdir(yolo_test_dir)
    
    if has_yolo_format:
        # Dataset is already in YOLO format, just generate masks and split files
        os.makedirs(split_dir, exist_ok=True)
        
        for mode in ['train', 'valid', 'test']:
            image_dir_path = join(root, mode, 'images')
            label_dir_path = join(root, mode, 'labels')
            
            if not isdir(image_dir_path):
                continue
                
            image_paths = sorted(glob(join(image_dir_path, '*.jpg')))
            paths = []
            
            for image_path in tqdm(image_paths, desc=f'Processing {mode}'):
                label_path = join(label_dir_path, basename(image_path).replace('.jpg', '.txt'))
                
                # Skip if label doesn't exist
                if not exists(label_path):
                    continue
                
                # Generate mask if it doesn't exist
                # gen_mask will create masks/ directory and .npy file based on label_path
                mask_path = label_path.replace('/labels/', '/masks/').replace('\\labels\\', '\\masks\\').replace('.txt', '.npy')
                os.makedirs(osp.dirname(mask_path), exist_ok=True)
                
                # Only generate mask if it doesn't exist
                if not exists(mask_path):
                    image = cv2.imread(image_path)
                    if image is not None:
                        gen_mask(label_path, image)
                
                paths.append(abspath(image_path) + '\n')
            
            # Write split file
            split_file = join(split_dir, f'{mode}.txt')
            with open(split_file, 'w+') as f:
                f.writelines(paths)
        
        print(f'TinyPerson preparation complete. Split files written to {split_dir}')
        return
    
    # Original pipeline: convert from JSON annotations
    if not exists(label_file_dict['train']) or not exists(label_file_dict['test']):
        warnings.warn('TinyPerson JSON annotation files not found. Expected format:\n'
                     '  - dataset/tinyperson/mini_annotations/tiny_set_train_all_erase.json\n'
                     '  - dataset/tinyperson/mini_annotations/tiny_set_test_all.json\n'
                     '  - dataset/tinyperson/erase_with_uncertain_dataset/\n'
                     'Alternatively, ensure dataset is in YOLO format with train/valid/test directories.')
        return
    
    os.makedirs(split_dir, exist_ok=True)
    for mode in ['train', 'test']:
        with open(label_file_dict[mode], 'r') as f:
            anno = json.load(f)
    
        image_dict = {}
        for item in anno['images']:
            file_name, width, height = item['file_name'], item['width'], item['height']
            file_path = join(image_dir, mode, file_name)
            image_dict[item['id']] = {'shape': [width, height], 'bboxes': [], 'image_path': file_path}
        
        for item in anno['annotations']:
            if item['ignore'] or item['uncertain']:
                continue
            _id, (x1, y1, w, h) = item['image_id'], item['bbox']
            (width, height) = image_dict[_id]['shape']
            xc, yc, w, h = (x1 + w / 2.) / width, (y1 + h / 2.) / height, w / width, h / height
            image_dict[_id]['bboxes'].append('0 %.6f %.6f %.6f %.6f\n' % (xc, yc, w, h))
    
        paths = []
        for item in image_dict.values():
            image_path = item['image_path']
            label_path = item['image_path'][:-4] + '.txt'
            with open(label_path, 'w+') as f:
                f.writelines(item['bboxes'])
            
            image = cv2.imread(image_path)
            gen_mask(label_path, image)
            paths.append(image_path + '\n')
        
        if mode == 'train':
            with open(join(split_dir, 'trainval.txt'), 'w+') as f:
                f.writelines(paths)
            k = 10
            random.shuffle(paths)
            sep = len(paths) // k
            with open(join(split_dir, 'train.txt'), 'w+') as f:
                f.writelines(paths[sep:])
            with open(join(split_dir, 'valid.txt'), 'w+') as f:
                f.writelines(paths[:sep])
        else:
            with open(join(split_dir, 'test.txt'), 'w+') as f:
                f.writelines(paths)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='VisDrone', help='dataset, e.g., VisDrone, UAVDT, and TinyPerson')
    opt = parser.parse_args()

    assert exists(opt.dataset)
    dataset = opt.dataset.lower()
    if 'visdrone' in dataset:
        prepare_visdrone()
    elif 'uavdt' in dataset:
        prepare_uavdt()
    elif 'tinyperson' in dataset:
        prepare_tinyperson()
    else:
        print('%s is coming soon.' % opt.dataset)
