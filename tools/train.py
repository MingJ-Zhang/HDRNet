from __future__ import annotations

import argparse
import os

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmseg.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Train HDRNet with MMSegmentation')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='directory to save logs and models')
    parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint in work_dir')
    parser.add_argument('--amp', action='store_true', help='force AMP optimizer wrapper')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config options')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif not hasattr(cfg, 'work_dir') or cfg.work_dir is None:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])

    cfg.launcher = args.launcher
    if args.resume:
        cfg.resume = True

    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
