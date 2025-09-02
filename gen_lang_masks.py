import os
import argparse
import numpy as np
import cv2
from PIL import Image
from lang_sam import LangSAM


def main(image_dir, output_dir, text_prompt, scale, device):
    """
    Run LangSAM to predict masks from text prompt and save masks and images.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = LangSAM(device=device)

    image_files = sorted(os.listdir(image_dir))
    for idx, img_name in enumerate(image_files):
        image_path = os.path.join(image_dir, img_name)
        image = Image.open(image_path).convert('RGB')

        results = model.predict([image], [text_prompt])[0]

        # Save the selected scale mask
        mask = results['masks'][scale].astype(np.uint8)
        np.save(os.path.join(output_dir, f'{img_name[:-4]}.npy'), mask)
        cv2.imwrite(os.path.join(output_dir, f'{img_name[:-4]}.jpg'), mask * 255)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate masks using LangSAM")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save masks")
    parser.add_argument("--text_prompt", type=str, required=True, help="Text prompt for mask generation")
    parser.add_argument("--scale", type=int, default=1, help="Mask scale level (0,1)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference")

    args = parser.parse_args()
    main(
        args.image_dir,
        args.output_dir,
        args.text_prompt,
        args.scale,
        args.device
    )
