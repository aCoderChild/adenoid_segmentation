import os
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io, transform
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import importlib.util
import cv2

# Paths
VAL_IMG = '../../data/segmentation/val_data/images'
VAL_BBOX = '../../external/YOLOv26/results' # This is where your input .txt files live
VAL_MASK = '../../data/segmentation/val_data/masks'
CHECKPOINT = 'work_dir/MedSAM/medsam_vit_b.pth'
RESULTS_DIR = 'results/yolov26'

# --- Define a specific output folder for the converted CSVs ---
CSV_OUT_DIR = os.path.join(RESULTS_DIR, 'predicted_bbox')

VIS_DIR = os.path.join(RESULTS_DIR, 'visualisations')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
PER_SAMPLE_CSV = os.path.join(METRICS_DIR, 'metrics_per_single_sample.csv')
AVG_CSV = os.path.join(METRICS_DIR, 'metrics_average.csv')

# --- Create the CSV output directory if it doesn't exist ---
os.makedirs(CSV_OUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def convert_txt_to_csv(txt_path, csv_path, img_width, img_height, class_mapping=None):
    """
    Converts a YOLO format .txt bounding box file to a .csv format.
    """
    if class_mapping is None:
        class_mapping = {0: 'polyp'}

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_name', 'xmin', 'ymin', 'xmax', 'ymax'])

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5: 
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                xmin = int((x_center - width / 2) * img_width)
                ymin = int((y_center - height / 2) * img_height)
                xmax = int((x_center + width / 2) * img_width)
                ymax = int((y_center + height / 2) * img_height)

                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img_width, xmax)
                ymax = min(img_height, ymax)

                class_name = class_mapping.get(class_id, str(class_id))
                writer.writerow([class_name, xmin, ymin, xmax, ymax])

# Load MedSAM model
def load_medsam_model(checkpoint_path, device):
    from segment_anything.build_sam import sam_model_registry
    medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    medsam_model.eval()
    medsam_model.to(device)
    return medsam_model

# Inference function
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None, boxes=box_torch, masks=None)
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred, size=(H, W), mode="bilinear", align_corners=False)
    low_res_pred = low_res_pred.squeeze().detach().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

# Visualisation helpers
def overlay_mask(img, mask, color, alpha=0.5):
    img = img.copy()
    mask = mask.astype(bool)
    img[mask] = (1 - alpha) * img[mask] + alpha * np.array(color)
    return img

def draw_bbox(img, bbox, color, lw=2):
    img = img.copy()
    x0, y0, x1, y1 = [int(v) for v in bbox]
    img = cv2.rectangle(img, (x0, y0), (x1, y1), color, lw)
    return img

# Load SurfaceDice
SURFACE_DICE_PATH = 'utils/SurfaceDice.py'
spec = importlib.util.spec_from_file_location('SurfaceDice', SURFACE_DICE_PATH)
SurfaceDice = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SurfaceDice)

# Main batch inference and evaluation
def main():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device selected: {device}")
    print(f"Loading MedSAM model from: {CHECKPOINT}")
    medsam_model = load_medsam_model(CHECKPOINT, device)
    
    results = []
    img_names = sorted(os.listdir(VAL_IMG))
    print(f"Found {len(img_names)} images in {VAL_IMG}")
    processed_count = 0
    
    for img_name in tqdm(img_names):
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(VAL_IMG, img_name)
        
        # --- NEW: Read from VAL_BBOX, but save to CSV_OUT_DIR ---
        bbox_txt_path = os.path.join(VAL_BBOX, base + '.txt')
        bbox_csv_path = os.path.join(CSV_OUT_DIR, base + '.csv') 
        mask_path = os.path.join(VAL_MASK, base + '.jpg')
        
        if not os.path.exists(mask_path):
            print(f"Skipping {img_name}: Mask not found at {mask_path}")
            continue

        # Read image
        print(f"\nProcessing image: {img_path}")
        img_np = io.imread(img_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        H, W, _ = img_3c.shape

        # Check for CSV. If missing, convert TXT to CSV and save to CSV_OUT_DIR
        if not os.path.exists(bbox_csv_path):
            if os.path.exists(bbox_txt_path):
                print(f"Converting {bbox_txt_path} -> {bbox_csv_path}")
                convert_txt_to_csv(bbox_txt_path, bbox_csv_path, W, H)
            else:
                print(f"Skipping {img_name}: .txt bbox not found at {bbox_txt_path}.")
                continue

        # Preprocess
        img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
        img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Read bbox from the newly created CSV
        # Read bbox from the newly created CSV
        box = None
        
        with open(bbox_csv_path, 'r') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
                
                # Check if the header is actually data. If it's text, grab the next row.
                if not header[1].replace('.', '', 1).isdigit():
                    row = next(reader)
                else:
                    row = header
                    
                # Parse the bounding box if the row has the correct format
                if len(row) == 5:
                    box = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]
                else:
                    raise ValueError(f"Unexpected bbox format in {bbox_csv_path}: {row}")
                    
            except StopIteration:
                # This catches the error if the file only contains a header or is completely empty
                pass

        # If no valid box was extracted, use the whole image dimensions
        if box is None:
            print(f"No bounding box found in {bbox_csv_path}. Falling back to full image.")
            box = [0.0, 0.0, float(W), float(H)]
                
        print(f"Loaded bbox from {bbox_csv_path}: {box}")
        box_np = np.array([box])
        box_1024 = box_np / np.array([W, H, W, H]) * 1024
        
        # Inference
        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor)
        pred_mask = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
        
        # Load GT mask
        gt_mask = io.imread(mask_path)
        if gt_mask.max() > 1:
            gt_mask = (gt_mask > 0).astype(np.uint8)
            
        # Create visualizations
        vis1 = overlay_mask(img_3c/255.0, gt_mask, [0,1,0], 0.5) # GT green
        vis1 = draw_bbox((vis1*255).astype(np.uint8), box, (0,255,0))
        vis2 = overlay_mask(img_3c/255.0, pred_mask, [1,0,0], 0.5) # Pred red
        vis2 = draw_bbox((vis2*255).astype(np.uint8), box, (0,0,255))
        vis3 = overlay_mask(img_3c/255.0, gt_mask, [0,1,0], 0.3)
        vis3 = overlay_mask(vis3, pred_mask, [1,0,0], 0.3)
        vis3 = (vis3*255).astype(np.uint8)

        # Combine the three images horizontally with a separator
        separator = np.full((vis1.shape[0], 5, 3), 255, dtype=np.uint8)
        combined = np.hstack((vis1, separator, vis2, separator, vis3))

        # Save the combined image
        combined_path = os.path.join(VIS_DIR, f'{base}_combined.jpg')
        io.imsave(combined_path, combined)
        
        # Evaluate
        gt_mask_3d = np.expand_dims(gt_mask, axis=0)
        pred_mask_3d = np.expand_dims(pred_mask, axis=0)
        
        surface_distances = SurfaceDice.compute_surface_distances(gt_mask_3d, pred_mask_3d, spacing_mm=(1.0, 1.0, 1.0))
        snd_score = SurfaceDice.compute_surface_dice_at_tolerance(surface_distances, 1)
        dice_score = SurfaceDice.compute_dice_coefficient(gt_mask_3d.astype(bool), pred_mask_3d.astype(bool))
        
        results.append({'image': img_name, 'surface_dice': snd_score, 'dice': dice_score})
        processed_count += 1

    # Save metrics
    print(f"\nProcessed {processed_count} images.")
    with open(PER_SAMPLE_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['image', 'surface_dice', 'dice'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    if results:
        avg_snd = np.mean([r['surface_dice'] for r in results])
        avg_dice = np.mean([r['dice'] for r in results])
    else:
        avg_snd = 0
        avg_dice = 0
        
    with open(AVG_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['average_surface_dice', 'average_dice'])
        writer.writerow([avg_snd, avg_dice])
    print('Done!')

if __name__ == '__main__':
    main()