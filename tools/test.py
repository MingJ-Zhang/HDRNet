from __future__ import annotations

import argparse
import os

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmseg.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Test HDRNet with MMSegmentation')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='directory to save evaluation metrics')
    parser.add_argument('--out', help='pickle output path for predictions')
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
    cfg.load_from = args.checkpoint
    cfg.launcher = args.launcher
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()
