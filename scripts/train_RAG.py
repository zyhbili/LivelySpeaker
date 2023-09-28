# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from mdm_utils.fixseed import fixseed
from mdm_utils.parser_util import train_args
from mdm_utils import dist_util
from train_utils.train_loop import TrainLoop

from mdm_utils.model_util import create_model_and_diffusion
from train_utils.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from train_utils.ted_loader import build_dataloader
def main():
    args = train_args()
    save_dir = f"{args.save_dir}/{args.exp}"
    args.save_dir = save_dir
    print("save_dir:", save_dir)
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = build_dataloader('train', args, shuffle = True)
    print("creating model and diffusion...")
    lang_model = data.dataset.lang_model
    args.lang_model = lang_model
    model, diffusion = create_model_and_diffusion(args, '')

    model.to(dist_util.dev())

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()

    train_platform.close()

if __name__ == "__main__":
    main()
