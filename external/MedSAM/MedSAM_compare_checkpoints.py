import os
import numpy as np
import torch
from skimage import io, transform
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import cv2

# Load config from YAML
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'infer.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

VAL_IMG = config.get('VAL_IMG', 'data/segmentation/val_data/images')
VAL_MASK = config.get('VAL_MASK', 'data/segmentation/val_data/masks')
CHECKPOINT_FINETUNED = config.get('FINETUNE_CHECKPOINT')
CHECKPOINT_ORIGINAL = config.get('ORIGINAL_CHECKPOINT', 'external/MedSAM/work_dir/MedSAM/medsam_vit_b.pth')
RESULTS_DIR = config.get('RESULTS_DIR_COMPARE', 'external/MedSAM/results/compare_checkpoints')
VIS_DIR = os.path.join(RESULTS_DIR, 'vis')
os.makedirs(VIS_DIR, exist_ok=True)

from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling.prompt_encoder import PromptEncoder
from utils.lora_medsam import apply_lora_to_vit_encoder
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

def load_medsam_model(checkpoint_path, device, use_lora):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    medsam_model = sam_model_registry["vit_b"](checkpoint="work_dir/MedSAM/medsam_vit_b.pth")
    prompt_embed_dim = 256
    medsam_model.prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16,
    )
    if use_lora:
        medsam_model.image_encoder = apply_lora_to_vit_encoder(medsam_model.image_encoder, r=8, alpha=16)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    medsam_model.load_state_dict(state_dict, strict=False)
    medsam_model.eval()
    medsam_model.to(device)
    return medsam_model

def overlay_mask(img, mask, color, alpha=0.5):
    img = img.copy()
    mask = mask.astype(bool)
    img[mask] = (1 - alpha) * img[mask] + alpha * np.array(color)
    return img

def main():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device selected: {device}")
    print(f"Loading original MedSAM model from: {CHECKPOINT_ORIGINAL}")
    print(f"Loading finetuned MedSAM model from: {CHECKPOINT_FINETUNED}")
    model_original = load_medsam_model(CHECKPOINT_ORIGINAL, device, use_lora=False)
    model_finetuned = load_medsam_model(CHECKPOINT_FINETUNED, device, use_lora=True)
    mask_generator_original = SamAutomaticMaskGenerator(
        model_original,
        points_per_side=16,
        pred_iou_thresh=0.3,
        stability_score_thresh=0.3,
        min_mask_region_area=0,
    )
    mask_generator_finetuned = SamAutomaticMaskGenerator(
        model_finetuned,
        points_per_side=16,
        pred_iou_thresh=0.3,
        stability_score_thresh=0.3,
        min_mask_region_area=0,
    )
    img_names = sorted(os.listdir(VAL_IMG))
    print(f"Found {len(img_names)} images in {VAL_IMG}")
    for img_name in tqdm(img_names):
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(VAL_IMG, img_name)
        img_np = io.imread(img_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        img_3c = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.float32)
        # Generate masks
        masks_orig = mask_generator_original.generate(img_3c)
        masks_finetuned = mask_generator_finetuned.generate(img_3c)
        main_mask_orig = max(masks_orig, key=lambda m: m['area'])['segmentation'] if len(masks_orig) > 0 else np.zeros((1024,1024), dtype=np.uint8)
        main_mask_finetuned = max(masks_finetuned, key=lambda m: m['area'])['segmentation'] if len(masks_finetuned) > 0 else np.zeros((1024,1024), dtype=np.uint8)
        # Overlay masks
        vis_orig = overlay_mask(img_3c/255.0, main_mask_orig, [1,0,0], 0.5)
        vis_finetuned = overlay_mask(img_3c/255.0, main_mask_finetuned, [0,0,1], 0.5)
        vis_combined = np.hstack((vis_orig*255, vis_finetuned*255)).astype(np.uint8)
        out_path = os.path.join(VIS_DIR, f'{base}_compare.png')
        io.imsave(out_path, vis_combined)
        print(f"Saved comparison for {img_name} to {out_path}")

if __name__ == "__main__":
    main()
