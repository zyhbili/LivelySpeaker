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
# from data_loaders.get_data import get_dataset_loader
from mdm_utils.model_util import create_model_and_diffusion
from train_utils.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
import torch
from dataloaders.build_vocab import Vocab


def main():
    args = train_args()
    save_dir = f"./beat_output/{args.exp}"
    args.save_dir = save_dir
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    # import pdb;pdb.set_trace()
    # if args.save_dir is None:
    #     raise FileNotFoundError('save_dir was not specified.')
    # elif os.path.exists(args.save_dir) and not args.overwrite:
    #     raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    # elif not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")

    train_data = __import__(f"dataloaders.beat", fromlist=["something"]).CustomDataset(args, "train")  
    print('train length:', len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=args.batch_size,  
        shuffle=False,  
        num_workers=16,
        drop_last=False,
    )
    data = train_loader
    # data = None
    # import pdb;pdb.set_trace()
    print("creating model and diffusion...")
    lang_model = data.dataset.lang_model
    args.lang_model = lang_model
    # if 'inpainting' in args.exp:
    # model, diffusion = create_model_and_diffusion(args)
    model, diffusion = create_model_and_diffusion(args)

    model.to(dist_util.dev())
    # model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
