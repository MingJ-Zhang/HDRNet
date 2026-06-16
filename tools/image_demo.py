from __future__ import annotations

import argparse
from mmseg.apis import MMSegInferencer


def main():
    parser = argparse.ArgumentParser(description='Single image inference')
    parser.add_argument('img')
    parser.add_argument('config')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-file', default=None)
    args = parser.parse_args()

    inferencer = MMSegInferencer(model=args.config, weights=args.checkpoint)
    inferencer(args.img, out_dir=None if args.out_file is None else args.out_file.rsplit('/', 1)[0], show=False)


if __name__ == '__main__':
    main()
