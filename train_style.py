#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, StyleOptimizationParams
from utils.nnfm_loss import NNFMLoss, match_colors_for_image_set, color_histgram_match
from utils.image_utils import load_and_preprocess_style_image
import imageio.v2 as imageio
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def set_geometry_grad(gaussian_model,freeze):
    if freeze:
        gaussian_model._xyz.requires_grad = False
        gaussian_model._scaling.requires_grad = False
        gaussian_model._rotation.requires_grad = False
        gaussian_model._opacity.requires_grad = False
    else:
        gaussian_model._xyz.requires_grad = True
        gaussian_model._scaling.requires_grad = True
        gaussian_model._rotation.requires_grad = True
        gaussian_model._opacity.requires_grad = True


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    nnfm_loss_fn = NNFMLoss(device='cuda')
    if args.point_cloud:
        xyz, o, s = gaussians.load_ply(args.point_cloud, reset_basis_dim=args.reset_basis_dim)
        original_xyz, original_opacity, original_scale = torch.tensor(xyz).cuda(), torch.tensor(o).cuda(), torch.tensor(s).cuda()
        first_iter = 30_000

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # Load style images
    style_img = load_and_preprocess_style_image(args.style, (scene.img_width, scene.img_height))
    # choose other hues:
    if args.second_style:
        style_img2 = imageio.imread(args.second_style, pilmode="RGB").astype(np.float32) / 255.0
        style_img2 = torch.from_numpy(style_img2).cuda()
    # precolor step
    if not args.preserve_color:
        gt_img_list = []
        for view in scene.getTrainCameras():
            gt_img_list.append(view.original_image.permute(1,2,0))
        gt_imgs = torch.stack(gt_img_list)

        if args.histgram_match:
            gt_imgs, color_ct = color_histgram_match(gt_imgs, style_img if not args.second_style else style_img2) #.repeat(gt_imgs.shape[0],1,1,1))
        else:
            gt_imgs, color_ct = match_colors_for_image_set(gt_imgs, style_img if not args.second_style else style_img2)
        gaussians.apply_ct(color_ct.detach().cpu().numpy())
        gt_img_list = [item.permute(2,0,1) for item in gt_imgs]
        imageio.imwrite(
            os.path.join(args.model_path, "gt_image_recolor.png"),
            np.clip(gt_img_list[0].permute(1,2,0).detach().cpu().numpy() * 255.0, 0.0, 255.0).astype(np.uint8),
        )

    # prepare depth image
    depth_img_list = []
    with torch.no_grad():
        for view in tqdm(scene.getTrainCameras(), desc="Rendering progress"):
            depth_render = render(view, gaussians, pipe, background)["depth"]
            depth_img_list.append(depth_render)

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if not args.preserve_color:
                gt_stack = gt_img_list.copy()
            depth_stack = depth_img_list.copy()
        view_idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(view_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, depth_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg['depth'], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        if not args.preserve_color:
            gt_image = gt_stack.pop(view_idx).cuda()
        else:
            gt_image = viewpoint_cam.original_image.cuda()
        depth_gt = depth_stack.pop(view_idx)

        gt_image = gt_image.unsqueeze(0)
        image = image.unsqueeze(0)

        if iteration > first_iter + 200: # stylization
            set_geometry_grad(gaussians,False) # True -> Turn off the geo change
            loss_dict = nnfm_loss_fn(
                F.interpolate(
                    image,
                    size=None,
                    scale_factor=0.5,
                    mode="bilinear",
                ),
                style_img.permute(2,0,1).unsqueeze(0),
                blocks=args.vgg_block,
                loss_names=["nnfm_loss", "content_loss"] if not args.preserve_color else ['lum_nnfm_loss','content_loss'],
                contents=F.interpolate(
                    gt_image,
                    size=None,
                    scale_factor=0.5,
                    mode="bilinear",
                ),
            )
            image.requires_grad_(True)
            w_variance = torch.mean(torch.pow(image[:, :, :, :-1] - image[:, :, :, 1:], 2))
            h_variance = torch.mean(torch.pow(image[:, :, :-1, :] - image[:, :, 1:, :], 2))

            loss_dict['nnfm_loss' if not args.preserve_color else 'lum_nnfm_loss'] *= args.style_weight
            loss_dict["content_loss"] *= args.content_weight
            loss_dict["img_tv_loss"] = args.img_tv_weight * (h_variance + w_variance) / 2.0
            loss_dict['depth_loss'] = l2_loss(depth_image, depth_gt)
            
        else: 
            set_geometry_grad(gaussians,True)
            loss_dict = {}
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            loss_dict['ddsm_loss'] = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # opacity & scale regulariers
        loss_dict['opacity_regu'] = l1_loss(gaussians._opacity, original_opacity)
        loss_dict['scale_regu'] = l1_loss(gaussians._scaling, original_scale)

        loss = sum(list(loss_dict.values()))

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = StyleOptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    # style params
    parser.add_argument("--point_cloud", type=str, help='trained real 3DGS ply', default = None)
    parser.add_argument("--style", type=str, help="path to style image")
    parser.add_argument("--second_style", type=str, default="", help="path to second style image")
    parser.add_argument("--histgram_match", action="store_true", default=True)
    parser.add_argument("--style_weight", type=float, default=5, help="style loss weight")
    parser.add_argument("--content_weight", type=float, default=5e-3, help="content loss weight")
    parser.add_argument("--img_tv_weight", type=float, default=1, help="image tv loss weight")
    parser.add_argument(
        "--vgg_block",
        type=list,
        default=[2,3],
        help="vgg block for nnfm extracting feature maps",
    )
    parser.add_argument(
        "--reset_basis_dim",
        type=int,
        default=1,
        help="whether to reset the number of spherical harmonics basis to this specified number",
    )
    parser.add_argument("--preserve_color", action="store_true", default=False)
   
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
