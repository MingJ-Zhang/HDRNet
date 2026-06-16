from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mmengine.analysis import get_model_complexity_info
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Print FLOPs and parameter count for an MMSegmentation model')
    parser.add_argument('config', help='model config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=None,
        help='input image size as H W, or one value for square input')
    parser.add_argument(
        '--show-table',
        action='store_true',
        help='print per-module complexity table')
    parser.add_argument(
        '--show-arch',
        action='store_true',
        help='print model architecture with complexity annotations')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options, e.g. model.decode_head.num_classes=10')
    return parser.parse_args()


def get_input_shape(args, cfg):
    if args.shape is None:
        crop_size = cfg.get('crop_size', None)
        if crop_size is None:
            crop_size = cfg.model.get('data_preprocessor', {}).get('size', None)
        if crop_size is None:
            raise ValueError('No input shape provided and no crop_size found in config.')
        height, width = crop_size
    elif len(args.shape) == 1:
        height = width = args.shape[0]
    elif len(args.shape) == 2:
        height, width = args.shape
    else:
        raise ValueError('--shape must be one value or two values: H W.')

    return 3, int(height), int(width)


def main():
    args = parse_args()

    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmseg'))

    input_shape = get_input_shape(args, cfg)
    model = MODELS.build(cfg.model)
    model.eval()

    complexity = get_model_complexity_info(
        model,
        input_shape,
        show_table=args.show_table,
        show_arch=args.show_arch)

    print('Input shape: {}'.format(input_shape))
    print('FLOPs: {}'.format(complexity['flops_str']))
    print('Params: {}'.format(complexity['params_str']))

    if args.show_table:
        print(complexity['out_table'])
    if args.show_arch:
        print(complexity['out_arch'])

    print(
        '\nNote: FLOPs are computed for one forward pass with the given input '
        'shape. For slide inference or multi-scale testing, total cost scales '
        'with the number of crops/scales.')


if __name__ == '__main__':
    main()
