# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import logging
import math
import os
import random
import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.load.*weights_only.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*mmdet.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.meshgrid.*indexing.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*distutils.*')
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
import torch.amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader, norm_imgs
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr, target2mask, check_mask
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution, plot_cluster
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel, time_synchronized
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from utils.csv_logger import CSVLogger
from utils.adaptive_hyperparameters import AdaptiveHyperparameterScheduler

logger = logging.getLogger(__name__)

def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank
    half_precision = not opt.disable_half

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'
    training_log_file = save_dir / 'training.log'  # Complete training log
    
    # Auto-resume: Automatically resume from latest checkpoint if it exists and --resume is not explicitly False
    if not opt.resume and last.exists() and last.stat().st_size > 0:
        # Check if checkpoint is valid and contains training state
        try:
            test_ckpt = torch.load(last, map_location='cpu', weights_only=False)
            if 'epoch' in test_ckpt and 'optimizer' in test_ckpt and test_ckpt.get('optimizer') is not None:
                opt.resume = True
                opt.weights = str(last)
                weights = str(last)  # Update local weights variable
                if rank in [-1, 0]:
                    logger.info(f'Auto-resuming training from {last} (epoch {test_ckpt["epoch"] + 1})')
        except Exception as e:
            if rank in [-1, 0]:
                logger.warning(f'Could not auto-resume from {last}: {e}. Starting fresh training.')
    
    # Initialize CSV logger
    csv_logger = CSVLogger(save_dir, filename='metrics.csv')
    
    # Setup file logging to save all training output to the experiment folder
    if rank in [-1, 0]:
        import sys
        from datetime import datetime
        
        # Create file handler for detailed training log
        file_handler = logging.FileHandler(training_log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                                     datefmt='%Y-%m-%d %H:%M:%S'))
        
        # Add handler to root logger to capture all logs
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        # Also create a handler that captures print statements
        class TeeOutput:
            """Tee output to both console and file"""
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
            def isatty(self):
                # Check if any of the underlying files is a TTY
                for f in self.files:
                    if hasattr(f, 'isatty'):
                        result = f.isatty()
                        if result:
                            return True
                return False
        
        # Open log file for stdout/stderr capture
        log_file_handle = open(training_log_file, 'a', encoding='utf-8')
        log_file_handle.write(f'\n\n{"="*80}\n')
        log_file_handle.write(f'Training started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        log_file_handle.write(f'{"="*80}\n\n')
        log_file_handle.flush()
        
        # Save original stdout/stderr and log file handle for cleanup
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Store log file handle for cleanup later (use train function's namespace)
        train._log_file_handle = log_file_handle
        train._original_stdout = original_stdout
        train._original_stderr = original_stderr
        
        # Redirect stdout and stderr to both console and file
        sys.stdout = TeeOutput(original_stdout, log_file_handle)
        sys.stderr = TeeOutput(original_stderr, log_file_handle)

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    if rank in [-1, 0] and not opt.resume:
        os.mkdir(save_dir / 'scripts')
        files_to_copy = ['train.py', 'test.py', 'utils/loss.py', 'utils/general.py', 'utils/datasets.py',
                         'models/yolo.py', 'models/common.py', opt.cfg]
        for file in files_to_copy:
            if file and os.path.isfile(file):
                shutil.copyfile(file, save_dir / 'scripts' / os.path.basename(file))

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights, weights_only=False).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check
    is_coco = opt.data.endswith('coco.yaml') and nc == 80  # COCO dataset

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device, weights_only=False)  # load checkpoint
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume and not opt.freeze else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude, cfg_path=opt.cfg)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    unfreeze = []
    if opt.freeze:
        raise NotImplementedError('Navigate to here to manually set the layers to freeze and uncomment this error.')
        hyp['weight_decay'] = 0.
        # freeze = ['model.{}.'.format(ii) for ii in range(8)]
        # print('freeze', freeze)
        unfreeze = ['model.5', 'model.6'] # DWConv and Segmenter
        print('unfreeze', unfreeze)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze) or (len(unfreeze) and all(x not in k for x in unfreeze)):
            # print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter) and (v.bias.requires_grad or True):
            pg2.append(v.bias)  # biases
        if (isinstance(v, nn.BatchNorm2d) or isinstance(v, nn.LayerNorm)) and (v.weight.requires_grad or True):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter) and (v.weight.requires_grad or True):
            pg1.append(v.weight)  # apply decay

    if opt.adam:
        # optimizer = optim.AdamW(pg0, lr=hyp['lr0'] * 0.02)  # adjust beta1 to momentum
        optimizer = optim.AdamW(pg0, lr=hyp['lr0'] * 0.1, betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:  # TODO: freeze
        # Optimizer
        if opt.resume and ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
            if rank in [-1, 0]:
                print('loaded optimizer, best_fitness = %.2f' % best_fitness)

        # EMA
        if opt.resume and ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']
            if rank in [-1, 0]:
                print('loaded ema, updates = %d' % ckpt['updates'])

        # Results
        if opt.resume and ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        if opt.resume:
            start_epoch = ckpt['epoch'] + 1
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs > start_epoch > 0:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs - start_epoch))
            # epochs += start_epoch  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 64)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, batch_size, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0., prefix=colorstr('val: '))[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # nc = np.array([len(c[c == ii]) for ii in range(int(c.max() + 1))])
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                # plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        find_unused_parameters = opt.hm_only or any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules())
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=find_unused_parameters)

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    # number of warmup iterations, max(3 epochs, 1k iterations)
    nw = round(hyp['warmup_epochs'] * nb) if hyp['warmup_epochs'] > 0 else -1
    # nw = max(round(hyp['warmup_epochs'] * nb), 1000) if hyp['warmup_epochs'] > 0 else -1
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.amp.GradScaler('cuda', enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class
    # Store model reference for uncertainty loss computation
    if hasattr(model, 'pred_mask_mean') or hasattr(compute_loss.hyp, 'get'):
        compute_loss.hyp['_model_module'] = model.module if hasattr(model, 'module') else model
    
    # Initialize Adaptive Hyperparameter Scheduler
    adaptive_scheduler = AdaptiveHyperparameterScheduler(
        hyp=hyp,
        save_dir=save_dir,
        enabled=not opt.disable_adaptive_hyp,  # Enable by default, disable with --disable-adaptive-hyp
        check_interval=1,  # Check every epoch
        min_epochs_before_adapt=10,  # Start adapting after 10 epochs
    )
    
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Training logs saved to: {save_dir / "training.log"}\n'
                f'Metrics CSV saved to: {save_dir / "metrics.csv"}\n'
                f'Results saved to: {save_dir / "results.txt"}\n'
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        use_gt = epoch < epochs * 0.6
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(7, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 11) %
                    ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'pixl', 'area', 'dist', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        warmup_flag = False
        for i, (imgs, targets, masks, m_weights, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            warmup_flag = ni <= nw
            imgs = imgs.to(device, non_blocking=True).float()
            # imgs /= 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            imgs = norm_imgs(imgs, model)
            masks = masks.to(device, non_blocking=True).float()
            m_weights = m_weights.to(device, non_blocking=True).float()

            # Warmup
            if warmup_flag:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                raise NotImplementedError('Interplotation will destroy the heatmap labels')
                sz = random.randrange(imgsz * 0.75, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    mask_stride = imgs.size(2) // masks.size(2)
                    assert mask_stride == 8
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                    masks = F.interpolate(masks, size=[_s // mask_stride for _s in ns], mode='nearest')

            # Forward
            # ESOD training strategy:
            # - During warmup (first nw iterations): Use GT masks to help Segmenter learn
            # - After warmup: Use predicted masks to train the full pipeline
            # - GT masks are always provided to compute_loss for supervision
            with torch.amp.autocast('cuda', enabled=cuda and half_precision):
                targets = targets.to(device, non_blocking=True)
                # Use GT masks during warmup to bootstrap Segmenter learning
                # After warmup, model uses its own predicted masks (proper ESOD training)
                use_gt_masks_for_forward = use_gt and warmup_flag
                pred = model((imgs, [masks]) if use_gt_masks_for_forward else imgs, hm_only=opt.hm_only)  # forward
                # Always provide GT masks to loss computation for proper supervision
                loss, loss_items = compute_loss(pred, targets, imgsz=imgs.shape, masks=masks, m_weights=m_weights)  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                
                # Periodic memory cleanup to reduce memory overhead
                if ni % (accumulate * 100) == 0 and cuda:  # Every 100 optimization steps
                    torch.cuda.empty_cache()

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 9) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot (only for first epoch and less frequently to save memory/time)
                if plots and ni < 3 and epoch == 0:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    if tb_writer and ni == 0 and not opt.sync_bn:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')  # suppress jit trace warning
                            tb_writer.add_graph(torch.jit.trace(de_parallel(model), imgs, strict=False), [])  # graph
                elif plots and ni == 10 and epoch == 0 and wandb_logger.wandb:
                    wandb_logger.log({'Mosaics': [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                # ESOD validation strategy:
                # - During early epochs (first 60%): Use GT masks for validation (helps with training)
                # - During later epochs: Use predicted masks (normal inference mode)
                # - This ensures model learns to use its own predictions
                validation_use_gt = use_gt and (epoch < epochs * 0.6)
                results, maps, times = test.test(data_dict,
                                                 use_gt=validation_use_gt,
                                                 conf_thres=0.001,  # Always use low threshold for validation to see all predictions
                                                 batch_size=batch_size * 2 if '_tr' not in opt.cfg else 1,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 save_json=is_coco and final_epoch,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 half_precision=half_precision,
                                                 hm_metric=opt.hm_metric,
                                                 is_coco=is_coco)

            # Write
            with open(results_file, 'a') as f:
                # results now includes: (mp, mr, map50, map, bpr, occupy, val_box, val_obj, val_cls, val_pixl, val_area, val_dist, gflops, fps)
                num_metrics = len(results)
                if num_metrics >= 14:  # Has GFLOPs and FPS
                    f.write(s + '%10.4g' * num_metrics % results + '\n')  # append all metrics including GFLOPs and FPS
                else:
                    f.write(s + '%10.4g' * 12 % results[:12] + '\n')  # backward compatibility

            # Update best mAP
            fi = fitness(np.array(results[:4]).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            
            # Adaptive Hyperparameter Update
            # Prepare metrics for adaptive scheduler
            adaptive_metrics = {
                'recall': results[1],  # mr
                'map50_95': results[3],  # mAP@0.5:0.95
                'box_loss': float(mloss[0].cpu()),
                'obj_loss': float(mloss[1].cpu()),
                'pixl_loss': float(mloss[3].cpu()),
                'val_total_loss': float(mloss[-1].cpu()),  # Total training loss as proxy
            }
            
            # Update adaptive hyperparameters
            applied_adjustments = adaptive_scheduler.update(epoch, adaptive_metrics)
            
            # If hyperparameters were updated, update model and loss computation
            if applied_adjustments:
                # Update model.hyp (used by ComputeLoss)
                # Note: hyp dict is modified in-place by adaptive_scheduler, so model.hyp 
                # (which is a reference to hyp) is automatically updated
                # However, ComputeLoss stores self.hyp = model.hyp at init, so we need to 
                # ensure compute_loss.hyp points to the updated dict
                model.hyp = hyp
                if hasattr(compute_loss, 'hyp'):
                    # Update the reference to point to the updated hyp dict
                    compute_loss.hyp = hyp
                compute_loss.gr = model.gr
                
                # Log adaptive changes
                logger.info(f"Adaptive hyperparameter adjustments at epoch {epoch}:")
                for param, info in applied_adjustments.items():
                    logger.info(f"  {param}: {info['new_value']:.6f} (reason: {info['reason']})")
                    # Log to TensorBoard/W&B
                    if tb_writer:
                        tb_writer.add_scalar(f'adaptive_hyp/{param}', info['new_value'], epoch)
                    if wandb_logger.wandb:
                        wandb_logger.log({f'adaptive_hyp/{param}': info['new_value']})
            
            # Log to CSV: epoch, train_losses, val_metrics, val_losses, learning_rates, fitness, gflops, fps
            # results = (mp, mr, map50, map, bpr, occupy, val_box, val_obj, val_cls, val_pixl, val_area, val_dist, gflops, fps)
            # mloss = [box, obj, cls, pixl, area, dist, total] (7 values, on GPU)
            val_metrics = list(results[:6])  # precision, recall, map50, map50_95, bpr, occupy
            val_losses = list(results[6:12]) if len(results) >= 12 else list(results[6:]) + [0.0] * (6 - len(results[6:]))  # validation losses (6 values)
            gflops = float(results[12]) if len(results) > 12 else 0.0  # GFLOPs
            fps = float(results[13]) if len(results) > 13 else 0.0  # FPS
            csv_logger.log(
                epoch=epoch,
                train_losses=[float(x) for x in mloss.cpu().tolist()],  # All 7 training losses
                val_metrics=val_metrics,  # Validation metrics
                val_losses=val_losses,  # Validation losses (ensure exactly 6)
                learning_rates=[float(x) for x in lr],  # Learning rates
                fitness=float(fi.item() if hasattr(fi, 'item') else fi),  # Fitness score
                gflops=gflops,  # GFLOPs
                fps=fps  # FPS
            )

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'train/pixl_loss', 'train/area_loss', 'train/dist_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'metrics/bestpr', 'metrics/occupy',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'val/pixl_loss', 'val/area_loss', 'val/dist_loss',  # train loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                # Use eval mode for model copy to save memory (faster deepcopy)
                was_training = model.training
                model.eval()
                if ema:
                    ema_was_training = ema.ema.training
                    ema.ema.eval()
                
                # Clear temporary attributes that can't be deepcopied (from UncertaintySegmenter)
                model_to_save = de_parallel(model)
                if hasattr(model_to_save, 'pred_mask_mean'):
                    delattr(model_to_save, 'pred_mask_mean')
                if hasattr(model_to_save, 'pred_mask_uncertainty'):
                    delattr(model_to_save, 'pred_mask_uncertainty')
                
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model_to_save).half(),
                        'ema': deepcopy(ema.ema).half() if ema else None,
                        'updates': ema.updates if ema else 0,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}
                model.train(was_training)  # Restore original training mode
                if ema:
                    ema.ema.train(ema_was_training)

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
        # Clear cache at end of each epoch to reduce memory overhead
        if cuda and epoch % 5 == 0:  # Every 5 epochs to balance between speed and memory
            torch.cuda.empty_cache()
    
    # end training
    if rank in [-1, 0]:
        logger.info(f'{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n')
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})

        if not opt.evolve:
            if is_coco:  # COCO dataset
                for m in [last, best] if best.exists() else [last]:  # speed, mAP tests
                    results, _, _ = test.test(opt.data,
                                              batch_size=batch_size * 2 if '_tr' not in opt.cfg else 1,
                                              imgsz=imgsz_test,
                                              conf_thres=0.001,
                                              iou_thres=0.7,
                                              model=attempt_load(m, device).half(),
                                              single_cls=opt.single_cls,
                                              dataloader=testloader,
                                              save_dir=save_dir,
                                              save_json=True,
                                              plots=False,
                                              half_precision=half_precision,
                                              is_coco=is_coco)

            # Strip optimizers
            for f in last, best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
            if wandb_logger.wandb:  # Log the stripped model
                try:
                    wandb_logger.wandb.log_artifact(str(best if best.exists() else last), type='model',
                                                    name='run_' + wandb_logger.wandb_run.id + '_model',
                                                    aliases=['latest', 'best', 'stripped'])
                except Exception as e:
                    logger.warning(f'Failed to log model artifact to wandb: {e}')
        wandb_logger.finish_run()
        
        # Generate adaptive hyperparameter summary and plot
        if rank in [-1, 0] and adaptive_scheduler.enabled:
            summary = adaptive_scheduler.get_summary()
            if summary['total_adjustments'] > 0:
                logger.info("\n" + "="*80)
                logger.info("ADAPTIVE HYPERPARAMETER SUMMARY")
                logger.info("="*80)
                logger.info(f"Total adjustments made: {summary['total_adjustments']}")
                logger.info("\nInitial hyperparameters:")
                for param, value in summary['initial_params'].items():
                    logger.info(f"  {param}: {value:.6f}")
                logger.info("\nFinal hyperparameters:")
                for param, value in summary['current_params'].items():
                    logger.info(f"  {param}: {value:.6f}")
                logger.info("\nRecent adjustments:")
                for adj in summary['recent_adjustments']:
                    logger.info(f"  Epoch {adj['epoch']}: {adj['parameter']} = {adj['old_value']:.6f} → {adj['new_value']:.6f} ({adj['reason']})")
                logger.info("="*80 + "\n")
                
                # Generate evolution plot
                try:
                    plot_path = save_dir / 'adaptive_hyp_evolution.png'
                    adaptive_scheduler.plot_evolution(save_path=plot_path)
                except Exception as e:
                    logger.warning(f"Failed to generate adaptive hyperparameter plot: {e}")
        
        # Close log file and restore stdout/stderr
        if rank in [-1, 0]:
            from datetime import datetime
            import sys
            log_file_handle = getattr(train, '_log_file_handle', None)
            if log_file_handle:
                log_file_handle.write(f'\n\n{"="*80}\n')
                log_file_handle.write(f'Training completed at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                log_file_handle.write(f'{"="*80}\n\n')
                log_file_handle.flush()
                log_file_handle.close()
                # Restore stdout/stderr
                original_stdout = getattr(train, '_original_stdout', None)
                original_stderr = getattr(train, '_original_stderr', None)
                if original_stdout:
                    sys.stdout = original_stdout
                if original_stderr:
                    sys.stderr = original_stderr
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv5 Training with Adaptive Hyperparameters')
    parser.add_argument('mode', nargs='?', type=str, default='train', choices=['train'], 
                       help='Training mode (default: train)')
    
    # Default values for VisDrone Uncertainty Model
    parser.add_argument('--weights', type=str, default='yolov5m.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/cfg/esod/visdrone_yolov5m_uncertainty.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/visdrone.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.visdrone.finetune.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[1536, 1536], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='visdrone_yolov5m_uncertainty', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--addition', action='store_true', help='additional training')
    parser.add_argument('--freeze', action='store_true', help='freeze shadow backbone for fine-tuning')
    parser.add_argument('--hm-only', action='store_true', help='training on heatmap prediction only')
    parser.add_argument('--hm-metric', action='store_true', help='use heatmap-related evaluation metrics')
    parser.add_argument('--disable-half', action='store_true', help='disable FP16 half-precision training')
    parser.add_argument('--disable-adaptive-hyp', action='store_true', help='disable adaptive hyperparameter tuning')
    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0] and False:
        check_git_status()
        check_requirements(exclude=('pycocotools', 'thop'))

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        addition = opt.addition
        # Preserve exist_ok from command line to ensure same directory is used when resuming
        exist_ok_resume = opt.exist_ok
        if addition:
            epochs, hyp = opt.epochs, opt.hyp
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt_dict = yaml.safe_load(f)
            opt = argparse.Namespace(**opt_dict)  # replace
        # Use the original save_dir from checkpoint when resuming to ensure same directory
        original_save_dir = opt_dict.get('save_dir', None)
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank, opt.exist_ok = \
            '', ckpt, True, opt.total_batch_size, *apriori, exist_ok_resume  # reinstate (preserve exist_ok from command line)
        if addition:
            opt.epochs, opt.hyp = opt.epochs * 0 + epochs, hyp  # TODO: epochs
        logger.info('Resuming training from %s' % ckpt)
        # Use original save_dir if available, otherwise use increment_path with exist_ok
        if original_save_dir and Path(original_save_dir).exists():
            opt.save_dir = str(original_save_dir)
        else:
            opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve))
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        
        # Check if directory exists and has a valid checkpoint for auto-resume
        # If so, use existing directory (set exist_ok=True) to prevent creating new numbered directory
        potential_save_dir = Path(opt.project) / opt.name
        potential_last = potential_save_dir / 'weights' / 'last.pt'
        if potential_last.exists() and potential_last.stat().st_size > 0:
            try:
                test_ckpt = torch.load(potential_last, map_location='cpu', weights_only=False)
                if 'epoch' in test_ckpt and 'optimizer' in test_ckpt and test_ckpt.get('optimizer') is not None:
                    # Valid checkpoint exists, use existing directory to allow auto-resume
                    opt.save_dir = str(potential_save_dir)
                    logger.info(f'Found existing checkpoint at {potential_last}, will auto-resume from epoch {test_ckpt["epoch"] + 1}')
                else:
                    # No valid checkpoint, create new directory
                    opt.save_dir = str(increment_path(potential_save_dir, exist_ok=opt.exist_ok | opt.evolve))
            except Exception:
                # Checkpoint load failed, create new directory
                opt.save_dir = str(increment_path(potential_save_dir, exist_ok=opt.exist_ok | opt.evolve))
        else:
            # No checkpoint exists, create directory (may be new or existing based on exist_ok)
            opt.save_dir = str(increment_path(potential_save_dir, exist_ok=opt.exist_ok | opt.evolve))

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0] and False:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        
        # with torch.autograd.set_detect_anomaly(True):
        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
