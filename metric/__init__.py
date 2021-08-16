import torch
from torch import nn

from metric.cityscapes_mIoU import DRNSeg
from metric.fid_score import _compute_statistics_of_ims, calculate_frechet_distance
from metric.inception import InceptionV3
from utils import util


def get_fid(fakes, model, npz, device, batch_size=1, tqdm_position=None):
    m1, s1 = npz['mu'], npz['sigma']
    fakes = torch.cat(fakes, dim=0)
    fakes = util.tensor2im(fakes).astype(float)
    m2, s2 = _compute_statistics_of_ims(fakes, model, batch_size, 2048,
                                        device, tqdm_position=tqdm_position)
    return float(calculate_frechet_distance(m1, s1, m2, s2))


def get_cityscapes_mIoU(fakes, names, model, device,
                        table_path='datasets/table.txt',
                        data_dir='database/cityscapes',
                        batch_size=1, num_workers=8, num_classes=19,
                        tqdm_position=None):
    from .cityscapes_mIoU import test
    fakes = torch.cat(fakes, dim=0)
    fakes = util.tensor2im(fakes)
    mIoU = test(fakes, names, model, device, table_path=table_path, data_dir=data_dir,
                batch_size=batch_size, num_workers=num_workers, num_classes=num_classes, tqdm_position=tqdm_position)
    return float(mIoU)

def create_metric_models(opt, device):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx])
    inception_model.to(device)
    inception_model.eval()
    if 'cityscapes' in opt.dataroot and opt.direction == 'BtoA':
        drn_model = DRNSeg('drn_d_105', 19, pretrained=False)
        util.load_network(drn_model, opt.drn_path, verbose=False)
        if len(opt.gpu_ids) > 0:
            drn_model = nn.DataParallel(drn_model, opt.gpu_ids)
        drn_model.to(device)
        drn_model.eval()
    else:
        drn_model = None
    return inception_model, drn_model
