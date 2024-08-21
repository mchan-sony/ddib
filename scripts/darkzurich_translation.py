"""
Synthetic domain translation from a source 2D domain to a target.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch.distributed as dist

from common import read_model_and_diffusion
from torchvision.utils import save_image, make_grid
from guided_diffusion import dist_util, logger
from guided_diffusion.synthetic_datasets import scatter, heatmap, load_2d_data, Synthetic2DType
from guided_diffusion.image_datasets import load_source_data_for_domain_translation, load_data
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
import torch as th
from PIL import Image


def main():
    args = create_argparser().parse_args()
    logger.log(f"args: {args}")


    dist_util.setup_dist()
    logger.configure()

    logger.log("loading data...")
    # data = load_source_data_for_domain_translation(
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     data_dir="/data/matthew/dark_zurich/val/rgb_anon/val_ref/day/GOPR0356_ref"
    # )
    data = load_data(
        data_dir="/data/matthew/dark_zurich/val/rgb_anon/val_ref/day/GOPR0356_ref",
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=False,
        deterministic=True
    )

    # source_dir = "/data/matthew/P2-weighting/ckpts/day"
    # source_model, diffusion = read_model_and_diffusion(args, source_dir)

    # target_dir = "/data/matthew/P2-weighting/ckpts/night"
    # target_model, _ = read_model_and_diffusion(args, target_dir)

    logger.log("creating source model and diffusion...")
    source_model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    source_model.load_state_dict(
        dist_util.load_state_dict("/data/matthew/P2-weighting/logs/ema_0.9999_080000.pt", map_location="cpu")
    )
    source_model.to(dist_util.dev())
    if args.use_fp16:
        source_model.convert_to_fp16()
    source_model.eval()

    logger.log("creating target model and diffusion...")
    target_model, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    target_model.load_state_dict(
        dist_util.load_state_dict("/data/matthew/P2-weighting/night_logs/ema_0.9999_110000.pt", map_location="cpu")
    )
    target_model.to(dist_util.dev())
    if args.use_fp16:
        target_model.convert_to_fp16()
    target_model.eval()

    out_dir = "exp/daytonight"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sources = []
    latents = []
    targets = []

    for i, (source, _) in enumerate(data):
        logger.log(f"translating batch {i}, shape {source.shape}.")
        logger.log(f"device: {dist_util.dev()}")

        source = source.to(dist_util.dev())

        noise = diffusion.ddim_reverse_sample_loop(
            source_model, source,
            clip_denoised=False,
            device=dist_util.dev(),
        )
        logger.log(f"obtained latent representation for {source.shape[0]} samples...")
        logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")

        target = diffusion.ddim_sample_loop(
            target_model, 
            (args.batch_size, 3, args.image_size, args.image_size),
            noise=noise,
            clip_denoised=False,
            device=dist_util.dev(),
            eta=args.eta
        )
        target = (target + 1) / 2
        source = (source + 1) / 2
        # target = ((target + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # target = target.permute(0, 2, 3, 1)
        # target = target.contiguous()

        # source = ((source + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # source = source.permute(0, 2, 3, 1)
        # source = source.contiguous()

        logger.log(f"finished translation {target.shape}")

        # sources.append(source.cpu().numpy())
        # latents.append(noise.cpu().numpy())
        # targets.append(target.cpu().numpy())
        # images = []
        # gathered_samples = [th.zeros_like(target) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, target)  # gather not supported with NCCL
        # images.extend([sample.cpu().numpy() for sample in gathered_samples])
        # logger.log(f"created {len(images) * args.batch_size} samples")

        # images = np.concatenate(images, axis=0)
        logger.log("saving translated images.")
        output = th.vstack([source, target])
        save_image(make_grid(output, nrow=len(target)), f"{out_dir}/grid_{i}.png")

        # for index in range(images.shape[0]):
        #     image = Image.fromarray(images[index])
        #     image.save(f"{out_dir}/img_{index}.png")
        break

    dist.barrier()
    # logger.log(f"synthetic data translation complete: {i}->{j}\n\n")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=8,
        eta=0.0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    # defaults = dict(
    #     num_samples=90000,
    #     batch_size=30000,
    #     model_path=""
    # )
    # defaults.update(model_and_diffusion_defaults_2d())
    # parser = argparse.ArgumentParser()
    # add_dict_to_argparser(parser, defaults)
    # return parser


if __name__ == "__main__":
    main()
