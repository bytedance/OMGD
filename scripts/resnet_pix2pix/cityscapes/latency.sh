#!/usr/bin/env bash
python latency.py --dataroot database/cityscapes \
  --gpu_ids -1 \
  --results_dir results/resnet_pix2pix/cityscapes/S16 \
  --ngf 16 --netG mobile_resnet_9blocks \
  --restore_G_path checkpoints/resnet_pix2pix/cityscapes/best_net_G16.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/table.txt \
  --direction BtoA --need_profile