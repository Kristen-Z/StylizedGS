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

import torch
import imageio.v2 as imageio
import os, cv2
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_and_preprocess_style_image(style_path, target_size, output_dir=None, device='cuda', save_img=False):
    """
    Load and preprocess the reference style image for neural style transfer.

    Args:
        style_path (str): Path to the style image.
        target_size (tuple): (width, height) of the content image for alignment.
        output_dir (str): Directory to save the processed style image.
        device (str): Device to move the tensor to ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Preprocessed style image tensor on the specified device.
    """
    # Load and normalize the image
    style_img = imageio.imread(style_path, pilmode="RGB").astype(np.float32) / 255.0
    style_h, style_w = style_img.shape[:2]
    content_w, content_h = target_size
    content_long_side = max(content_w, content_h)

    # Resize while preserving aspect ratio
    if style_h > style_w:
        new_w = int(content_long_side / style_h * style_w)
        new_h = content_long_side
    else:
        new_w = content_long_side
        new_h = int(content_long_side / style_w * style_h)

    style_img = cv2.resize(style_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    style_img = cv2.resize(
        style_img,
        (style_img.shape[1] // 2, style_img.shape[0] // 2),
        interpolation=cv2.INTER_AREA,
    )

    if output_dir and save_img:
        save_path = os.path.join(output_dir, "style_image.jpg")
        imageio.imwrite(save_path, np.clip(style_img * 255.0, 0, 255).astype(np.uint8))

    style_tensor = torch.from_numpy(style_img).permute(2, 0, 1).unsqueeze(0).to(device)

    print("[INFO] Style image loaded: {}, processed size: {}".format(style_path, style_tensor.shape))
    return style_tensor