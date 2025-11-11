# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os
from os.path import join, exists, isdir, basename
from glob import glob
from tqdm import tqdm
import random
import cv2
import numpy as np
import json


def _readlines(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    
    return lines


def darknet2visdrone():
    image_dir = join(opt.dataset, 'VisDrone2019-DET-test-dev', 'images')
    save_dir = join(opt.dataset, 'DET_results-test-dev')
    os.mkdir(save_dir) if not exists(save_dir) else None
    for path in tqdm(sorted(glob(join(opt.pred, '*.txt')))):
        image = cv2.imread(join(image_dir, basename(path).replace('.txt', '.jpg')))
        height, width, _ = image.shape
        with open(path, 'r') as f:
            lines = f.read().splitlines()
        preds = []
        for line in lines:
            cls, xc, yc, w, h, conf = list(map(float, line.split()))
            x1, y1 = xc - w / 2., yc - h / 2.
            preds.append('%d,%d,%d,%d,%.8f,%d,-1,-1\n' % (x1 * width, y1 * height, w * width, h * height, conf, int(cls) + 1))
        with open(join(save_dir, basename(path).replace('_masked', '')), 'w+') as f:
            f.writelines(preds)


def darknet2uavdt():
    save_dir = join(opt.dataset, 'UAV-benchmark-MOTD_v1.0', 'RES_DET', 'det_EfficientSOD')
    test_videos = _readlines(join(opt.dataset, 'split', 'test_video.txt'))
    for video_name in tqdm(test_videos):
        detections = []
        img0 = cv2.imread(glob(join(opt.dataset, 'UAV-benchmark-M', video_name, '*.jpg'))[0])
        height, width, _ = img0.shape
        for path in sorted(glob(join(opt.pred, '%s_img*.txt' % video_name))):
            frameID = int(basename(path)[9:15])
            bboxes = np.loadtxt(path, usecols=(0, 1, 2, 3, 4, 5))
            for cls, xc, yc, w, h, conf in bboxes:
                x1, y1 = (xc - w / 2.) * width, (yc - h / 2.) * height
                w, h = w * width, h * height
                detections.append('%d,-1,%.6f,%.6f,%.6f,%.6f,%.6f,1,-1\n' %
                                  (frameID, x1, y1, w, h, conf))
        with open(join(save_dir, '%s.txt' % video_name), 'w+') as f:
            f.writelines(detections)


def darknet2tinyperson():
    with open(join(opt.dataset, 'mini_annotations', 'tiny_set_test_all.json'), 'r') as f:
        anno = json.load(f)
    image_id_dict = {item['file_name']: item['id'] for item in anno['images']}
    image_shape_dict = {item['id']: [item['width'], item['height']] for item in anno['images']}

    detections = []
    for path in sorted(glob(join(opt.pred, '*.txt'))):
        image_name = basename(path)[:-4] + '.jpg'
        k = 'labeled_images/%s' % image_name
        v = image_id_dict[k] if k in image_id_dict else image_id_dict['pure_bg_images/%s' % image_name]
        width, height = image_shape_dict[v]
        
        bboxes = np.loadtxt(path, usecols=(0, 1, 2, 3, 4, 5))
        for cls, xc, yc, w, h, conf in bboxes:
            x1, y1 = (xc - w / 2.) * width, (yc - h / 2.) * height
            w, h = w * width, h * height
            if w * h > 30 ** 2:  # for better performance on AP_tiny (~0.00x)
                continue
            detections.append({
                'bbox': [x1, y1, w, h], 'score': conf,
                'image_id': v, 'category_id': 1
            })
    with open(join('evaluation', 'tiny_benchmark', 'MyPackage', 'tools', 'evaluate', 'pred.json'), 'w+') as f:
        json.dump(detections, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='VisDrone', help='dataset, e.g., VisDrone, UAVDT, and TinyPerson')
    parser.add_argument('--pred', type=str, default='runs/test/exp/labels', help='directory for prediction results (*.txt)')
    opt = parser.parse_args()

    assert exists(opt.dataset)
    dataset = opt.dataset.lower()
    if 'visdrone' in dataset:
        darknet2visdrone()
    elif 'uavdt' in dataset:
        darknet2uavdt()
    elif 'tinyperson' in dataset:
        darknet2tinyperson()
    else:
        print('%s is coming soon.' % opt.dataset)
