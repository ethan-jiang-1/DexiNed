
from __future__ import print_function

import argparse
import os
import time
import platform
import numpy as np

import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import DATASET_NAMES, BipedDataset, TestDataset, dataset_info
#from losses import *
from losses import bdcn_loss2
from model import DexiNed
from utils import (image_normalization, save_image_batch_to_disk,
                   visualize_result)

from main import parse_args, train_one_epoch, validate_one_epoch, testPich, test

def main_train(args):
    """Main function."""

    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")

    # Tensorboard summary writer

    tb_writer = None
    training_dir = os.path.join(args.output_dir,args.train_data)
    os.makedirs(training_dir,exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, args.train_data, args.checkpoint_data)
    if args.tensorboard and not args.is_testing:
        # from tensorboardX import SummaryWriter  # previous torch version
        from torch.utils.tensorboard import SummaryWriter # for torch 1.4 or greather
        tb_writer = SummaryWriter(log_dir=training_dir)

    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')

    # Instantiate model and move it to the computing device
    model = DexiNed().to(device)
    # model = nn.DataParallel(model)
    ini_epoch =0
    if not args.is_testing:
        if args.resume:
            ini_epoch=17
            model.load_state_dict(torch.load(checkpoint_path,
                                         map_location=device))
        dataset_train = BipedDataset(args.input_dir,
                                     img_width=args.img_width,
                                     img_height=args.img_height,
                                     mean_bgr=args.mean_pixel_values[0:3] if len(
                                         args.mean_pixel_values) == 4 else args.mean_pixel_values,
                                     train_mode='train',
                                     arg=args
                                     )
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)

    dataset_val = TestDataset(args.input_val_dir,
                              test_data=args.test_data,
                              img_width=args.test_img_width,
                              img_height=args.test_img_height,
                              mean_bgr=args.mean_pixel_values[0:3] if len(
                                  args.mean_pixel_values) == 4 else args.mean_pixel_values,
                              test_list=args.test_list, arg=args
                              )
    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.workers)
    # Testing
    if args.is_testing:

        output_dir = os.path.join(args.res_dir, args.train_data+"2"+ args.test_data)
        print(f"output_dir: {output_dir}")
        if args.double_img:
            # predict twice an image changing channels, then mix those results
            testPich(checkpoint_path, dataloader_val, model, device, output_dir, args)
        else:
            test(checkpoint_path, dataloader_val, model, device, output_dir, args)

        return

    criterion = bdcn_loss2
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.wd)
    # lr_schd = lr_scheduler.StepLR(optimizer, step_size=args.lr_stepsize,
    #                               gamma=args.lr_gamma)

    # Main training loop
    seed=1021
    for epoch in range(ini_epoch,args.epochs):
        if epoch % 7==0:

            seed = seed+1000
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print("------ Random seed applied-------------")
        # Create output directories

        output_dir_epoch = os.path.join(args.output_dir,args.train_data, str(epoch))
        img_test_dir = os.path.join(output_dir_epoch, args.test_data + '_res')
        os.makedirs(output_dir_epoch,exist_ok=True)
        os.makedirs(img_test_dir,exist_ok=True)

        train_one_epoch(epoch,
                        dataloader_train,
                        model,
                        criterion,
                        optimizer,
                        device,
                        args.log_interval_vis,
                        tb_writer,
                        args=args)
        validate_one_epoch(epoch,
                           dataloader_val,
                           model,
                           device,
                           img_test_dir,
                           arg=args)

        # Save model after end of every epoch
        torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                   os.path.join(output_dir_epoch, '{0}_model.pth'.format(epoch)))


if __name__ == '__main__':
    args = parse_args()

    args.is_testing = True
    args.input_dir = args.input_dir.replace("../../dataset/", "./data/")
    main_train(args)
