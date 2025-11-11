# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import logging
import os.path as osp
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from models.yolo import Model


def export_onnx(model: torch.nn.Module, path: str):
    import onnx
    import onnxsim

    device = next(model.parameters()).device
    img = torch.zeros(1, 3, 640, 640).to(device)

    torch.onnx.export(model, img, path, 
                      verbose=False, opset_version=12,
                      training=torch.onnx.TrainingMode.EVAL,
                      # do_constant_folding=True,
                      input_names=['images'],
                      output_names=['output'],
                      dynamic_axes=None)

    model_onnx = onnx.load(path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    model_onnx, check = onnxsim.simplify(
                        model_onnx,
                        # dynamic_input_shape=opt.dynamic,
                        input_shapes=None)
    assert check, 'assert check failed'
    onnx.save(model_onnx, path)


def convert_retinanet():
    import torchvision.transforms as T
    import torchvision.models.detection as detection

    device = torch.device('cuda:0')

    model = detection.retinanet_resnet50_fpn_v2(pretrained=True).to(device)
    model.eval()

    # export_onnx(model, 'weights/retinanet_resnet50_fpn_v2.onnx')

    model_yolo = Model('models/cfg/retina.yaml', ch=3, nc=80).to(device)  # create
    model_yolo.eval()
    # model_yolo.info()
    # print(model_yolo)

    # return

    model_state_dict = model.state_dict()
    yolo_state_dict = model_yolo.state_dict()
    # for k, v in yolo_state_dict.items():
    #     print(k, v.shape)
    # return
    state_dict = {}
    for k, v in model_state_dict.items():
        # print(k, v.shape)
        # continue
        if '.body.' in k:
            if '.layer' not in k:
                k = k.replace('backbone.body.', 'model.0.')
                k = k.replace('1.', '.')
            else:
                li = int(k.split('.layer')[1].split('.')[0])
                k = k.replace(f'backbone.body.layer{li}.', f'model.{li+1}.m.')
        elif '.fpn.' in k:
            if 'blocks.p' in k:
                continue
            li = int(k.split('_blocks.')[1].split('.')[0])
            if '.inner_' in k:
                k = k.replace(f'backbone.fpn.inner_blocks.{li}.0.', f'model.{13-li*4}.')
                k = k.replace('model.5.', 'model.6.')
            elif '.layer_' in k:
                k = k.replace(f'backbone.fpn.layer_blocks.{li}.0.', f'model.{15-li*4}.')
            else:
                raise ValueError(k)
        else:
            continue
        
        state_dict[k] = v
        # print(k, v.shape)
        # continue
    for k, v in yolo_state_dict.items():
        if k not in state_dict:
            state_dict[k] = v
    
    model_yolo.load_state_dict(state_dict, strict=True)
    model_yolo.eval()

    torch.save({'model': model_yolo}, 'weights/retinanet.pt')
    return

    img = torch.rand((1, 3, 320, 320)).to(device)
    features_retina = model.backbone(img)
    # for k, v in features_retina.items():
    #     print(k, type(k), type(v), v.shape)
    features_retina = [features_retina[str(i)] for i in [2, 1, 0]]  # large, medium, small

    features_yolo = model_yolo(img)
    # print([x.shape for x in features_yolo])

    for fi, (f1, f2) in enumerate(zip(features_retina, features_yolo)):
        assert (f1 == f2).all(), f'{f1 - f2}\n{fi}'

    # e1, e2 = None, None
    # f1, f2 = None, None
    # try:
    #     f1 = model.backbone.body['conv1'](img)
    #     f1 = model.backbone.body['bn1'](f1)
    #     f1 = model.backbone.body['relu'](f1)
    #     f1 = model.backbone.body['maxpool'](f1)
    #     f1 = model.backbone.body['layer1'](f1)
    # except Exception as e:
    #     print(e)
    #     e1 = str(e)

    # try:
    #     f2 = model_yolo(img)
    # except Exception as e:
    #     print(e)
    #     e2 = str(e)

    # # print(e1 == e2)
    # print(f1.shape, f2.shape, (f1 == f2).all().item())
    # # print(f1 - f2)
    # print((model.backbone.body['conv1'].weight == model_yolo.model[0].conv.weight).all().item(), end=" ")
    # print((model.backbone.body['bn1'].weight == model_yolo.model[0].bn.weight).all().item(), end=" ")
    # print((model.backbone.body['bn1'].bias == model_yolo.model[0].bn.bias).all().item(), end=" ")
    # print((model.backbone.body['bn1'].running_mean == model_yolo.model[0].bn.running_mean).all().item(), end=" ")
    # print((model.backbone.body['bn1'].running_var == model_yolo.model[0].bn.running_var).all().item(), end=" ")
    # print((model.backbone.body['bn1'].num_batches_tracked == model_yolo.model[0].bn.num_batches_tracked).all().item(), end="\n")
    # print(model.backbone.body['bn1'].training, model_yolo.model[0].bn.training)
    # print(model.backbone.body['bn1'])
    # print(model_yolo.model[0].bn)

    # print(model.backbone.body['layer1'][0].bn1 == model_yolo.model[2].m[0].bn1)
    # print(model_yolo.model[2].m[0].bn1)

    return
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img_org.convert("RGB")).unsqueeze(0).to(device)

    predictions = model(img)
    pred_boxes = predictions[0]['boxes'].cpu().detach().numpy()  # 边界框
    pred_scores = predictions[0]['scores'].cpu().detach().numpy()  # 得分
    pred_labels = predictions[0]['labels'].cpu().detach().numpy()  # 标签（通常是COCO数据集的类别索引）
    # print(pred_scores)

    score_threshold = 0.1

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img_org)

    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if score > score_threshold:
            x1, y1, x2, y2 = box
            
            patch = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            
            ax.add_patch(patch)

            plt.text(x1, y1, f"{label}: {score:.2f}", fontsize=12, color='white', 
                    bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.savefig('tmp.png', dpi=400)


def convert_rtmdet():
    import mmcv
    from mmdet.apis import inference_detector, init_detector
    from mmengine.config import Config, ConfigDict
    from mmengine.logging import print_log
    from mmengine.utils import ProgressBar, path

    from mmyolo.registry import VISUALIZERS
    from mmyolo.utils import switch_to_deploy
    from mmyolo.utils.labelme_utils import LabelmeFormat
    from mmyolo.utils.misc import get_file_list, show_data_classes

    rtmdet_root = '../mmyolo'
    cfg = f'{rtmdet_root}/configs/rtmdet/rtmdet_m_syncbn_fast_8xb32-300e_coco.py'
    checkpoint = f'{rtmdet_root}/weights/rtmdet_m_syncbn_fast_8xb32-300e_coco_20230102_135952-40af4fe8.pth'
    file = f'{rtmdet_root}/demo/demo.jpg'

    device = torch.device('cuda:0')
    config = Config.fromfile(cfg)
    if 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    
    model = init_detector(config, checkpoint, device=device, cfg_options={})

    # export_onnx(model, f'weights/{osp.basename(checkpoint).split(".")[0]}.onnx')
    # return

    # result = inference_detector(model, file)
    # print(type(result.pred_instances))

    model_yolo = Model('models/cfg/rtmdet_m.yaml', ch=3, nc=80).to(device)  # create
    model_yolo.eval()

    # # return

    model_state_dict = model.state_dict()
    yolo_state_dict = model_yolo.state_dict()
    # for k, v in yolo_state_dict.items():
    #     print(k, v.shape)
    # return
    state_dict = {}
    csplayer_cnt, csplayer_idx = 0, ()
    for k, v in model_state_dict.items():
        # print(k, v.shape)
        # continue
        if 'backbone.stem.' in k:
            k = k.replace('backbone.stem.', 'model.')
        elif 'backbone.stage' in k:
            m, n = list(map(int, k.split('backbone.stage')[1].split('.')[:2]))
            if (m, n) != csplayer_idx:
                csplayer_idx = (m, n)
                csplayer_cnt += 1
            k = k.replace(f'backbone.stage{m}.{n}.', f'model.{2+csplayer_cnt}.')
            if m == 4 and n == 1:
                k = k.replace(f'model.{2+csplayer_cnt}.conv', f'model.{2+csplayer_cnt}.cv')
        elif 'neck.reduce_layers.' in k:
            k = k.replace('neck.reduce_layers.2.', f'model.{2+csplayer_cnt+1}.')
        elif 'neck.top_down_layers.' in k:
            try:
                m, n = list(map(int, k.split('neck.top_down_layers.')[1].split('.')[:2]))
            except:
                m = int(k.split('.')[2])
                n = -1
            if (m, n) != csplayer_idx:
                csplayer_idx = (m, n)
                csplayer_cnt += 1
            if n != -1:
                k = k.replace(f'neck.top_down_layers.{m}.{n}.', f'model.{2+3+csplayer_cnt}.')
            else:
                k = k.replace(f'neck.top_down_layers.{m}.', f'model.{2+3+2+csplayer_cnt}.')
        elif 'neck.downsample_layers.' in k:
            li = int(k.split('neck.downsample_layers.')[1].split('.')[0])
            k = k.replace(f'neck.downsample_layers.{li}.', f'model.{2+3+2+1+csplayer_cnt+3*li}.')
        elif 'neck.bottom_up_layers.' in k:
            li = int(k.split('neck.bottom_up_layers.')[1].split('.')[0])
            k = k.replace(f'neck.bottom_up_layers.{li}.', f'model.{2+3+2+1+2+csplayer_cnt+3*li}.')
        else:
            continue
        
        state_dict[k] = v
        # print(k, v.shape)
    # return

    for k, v in yolo_state_dict.items():
        if k not in state_dict:
            state_dict[k] = v
    
    model_yolo.load_state_dict(state_dict, strict=True)
    model_yolo.eval()
    
    torch.save({'model': model_yolo}, 'weights/rtmdet_m.pt')
    return

    def _forword_neck(self, inputs):
        assert len(inputs) == len(self.in_channels)
        # self.upsample = torch.nn.Upsample(scale_factor=2)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[idx](feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_heigh)

            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        # for idx, conv in enumerate(self.out_layers):
        #     outs[idx] = conv(outs[idx])

        return tuple(outs)

    img = torch.rand((1, 3, 160, 160)).to(device)
    features_backbone = model.backbone(img)
    # features_rtmdet = model.neck(features_backbone)  # small, mediam, large
    # print([x.shape for x in features_rtmdet])
    features_rtmdet2 = _forword_neck(model.neck, features_backbone)
    # print([x.shape for x in features_rtmdet2])
    # for x1, x2, in zip(features_rtmdet, features_rtmdet2):
    #     assert (x1 == x2).all()

    features_yolo = model_yolo(img)
    # print([x.shape for x in features_yolo])

    for fi, (f1, f2) in enumerate(zip(features_rtmdet2, features_yolo)):
        assert (f1 == f2).all(), f'{f1 - f2}\n{fi}'
    print('Done.')
    return

    e1, e2 = None, None
    f1, f2 = None, None
    try:
        f1 = model.backbone.stem[0](img)
        # f1 = model.backbone.stem[0].conv(img)
        # f1 = model.backbone.stem[0].bn(f1)
        # f1 = model.backbone.body['relu'](f1)
        # f1 = model.backbone.body['maxpool'](f1)
        # f1 = model.backbone.body['layer1'](f1)
    except Exception as e:
        print(e)
        e1 = str(e)

    try:
        f2 = model_yolo(img)
    except Exception as e:
        print(e)
        e2 = str(e)

    # print(e1 == e2)
    print(f1.shape, f2.shape, (f1 == f2).all().item())
    # print(f1 - f2)
    # print(model.backbone.stem[0].bn)
    # print(model_yolo.model[0].bn)

    # print(model.backbone.body['layer1'][0].bn1 == model_yolo.model[2].m[0].bn1)
    # print(model_yolo.model[2].m[0].bn1)


def convert_sync_batchnorm_to_batchnorm(module):
    """
    递归地将给定模块及其子模块中的所有 SyncBatchNorm 实例替换为 BatchNorm。
    """
    module_output = module
    if isinstance(module, nn.SyncBatchNorm):
        # 创建一个新的 BatchNorm 实例
        module_output = nn.BatchNorm2d(module.num_features, 
                                       module.eps, 
                                       module.momentum,
                                       module.affine,
                                       module.track_running_stats)
        # 复制状态（weight, bias, running_mean, running_var等）
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()

        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked

    for name, child in module.named_children():
        new_child = convert_sync_batchnorm_to_batchnorm(child)
        module_output.add_module(name, new_child)

    return module_output


def convert_gpvit():
    from mmcv import Config, DictAction
    from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                            wrap_fp16_model)
    from mmdet.models import build_detector

    device = torch.device('cuda:0')

    config_path = 'models/gpvit/configs/gpvit/retinanet/gpvit_l2_retinanet_1x.py'
    ckpt_path = 'weights/gpvit_l2_retinanet_1x.pth'
    cfg = Config.fromfile(config_path)
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None
    cfg.gpu_ids = ['0']
    cfg.model.train_cfg = None
    model_mmcv = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')).to(device)
    load_checkpoint(model_mmcv, ckpt_path, map_location='cpu')
    model_mmcv.eval()
    # model_mmcv = convert_sync_batchnorm_to_batchnorm(model)

    # from models.gpvit import GPViTAdapterSingleStage
    # backbone = GPViTAdapterSingleStage().to(device)
    # backbone.eval()
    # # model_dict = backbone.state_dict()
    # # for i, (k, v) in enumerate(model_dict.items()):
    # #     print(i, k, v.shape)
    state_dict = torch.load(ckpt_path)['state_dict']
    # for i, (k, v) in enumerate(state_dict.items()):
    #     print(i, k, v.shape)
    # backbone.load_state_dict(
    #     {k.replace('backbone.', ''): v for k, v in state_dict.items() if 'backbone.' in k}
    #     , strict=False)
    # exit()
    
    # print((backbone.ad_norm2.weight == model_mmcv.backbone.ad_norm2.weight).all())
    img = torch.rand((1, 3, 640, 640), device=device)
    feats_mmcv = model_mmcv.backbone(img)
    # feats_yolo = backbone(img)

    model_yolo = Model('models/cfg/gpvit_l2.yaml', ch=3, nc=10).to(device)  # create
    # model_yolo = convert_sync_batchnorm_to_batchnorm(model_yolo)
    model_yolo.eval()
    model_dict = model_yolo.state_dict()
    # for i, (k, v) in enumerate(model_dict.items()):
    #     print(i, k, v.shape)
    # exit()
    updated_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            if 'backbone.level_embed' in k:
                k = 'model.0.level_embed'
            elif 'backbone.spm' in  k:
                k = k.replace('backbone.spm.', 'model.0.')
            else:
                k = k.replace('backbone.', 'model.1.')  # 1 or 6
        elif k.startswith('neck.'):
            li = int(k.split('.')[2])
            if li > 2:
                continue
            if 'lateral_convs' in k:
                if li == 2:
                    k = k.replace(f'neck.lateral_convs.{li}.', 'model.5.')
                else:
                    k = k.replace(f'neck.lateral_convs.{li}.', f'model.{11-li*4}.')
            elif 'fpn_convs' in k:
                k = k.replace(f'neck.fpn_convs.{li}.', f'model.{14-li*4}.')
            else:
                raise ValueError(k)
        else:
            continue
        updated_state_dict[k] = v
    # for i, (k, v) in enumerate(updated_state_dict.items()):
    #     print(i, k, v.shape)
    updated_state_dict.update({k: v for k, v in model_dict.items() if k not in updated_state_dict})

    model_yolo.load_state_dict(updated_state_dict, strict=True)
    # print(model_yolo.model[5].bn.weight)
    # print(model_mmcv.neck.lateral_convs[2].bn.weight)
    # exit()

    torch.save({'model': model_yolo}, 'weights/gpvit_l2.pt')
    exit()


    e1, e2 = None, None
    try:
        feats_mmcv = model_mmcv.neck(feats_mmcv)[:3]
    except Exception as e:
        e1 = str(e)

    feats_yolo = None
    try:
        feats_yolo = model_yolo(img)[0]
    except Exception as e:
        e2 = str(e)
    
    assert e1 == e2, f'\n{e1}\n{e2}'

    # print(type(feats_mmcv), type(feats_yolo))
    for i, (f1, f2) in enumerate(zip(feats_mmcv, feats_yolo)):
        # print(f1.shape, f2.shape)
        assert (f1 == f2).all(), f'\nlayer{i}\n{f1}\n{f2}'

    print('Done.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='retinanet', choices=['retinanet', 'rtmdet', 'gpvit'])
    parser.add_argument('--img_path', type=str, default='VisDrone/VisDrone2019-DET-train/images/0000002_00005_d_0000014_masked.jpg', help='directory for prediction results (*.txt)')
    opt = parser.parse_args()

    img_org = Image.open(opt.img_path)

    with torch.no_grad():
        eval(f'convert_{opt.model}')
