"""
Example: Root cell image dataset writer.

Processes JPG images using FastSAM to extract bounding boxes, then crops and resizes
images before writing to TFRecord format.

Usage:
    python examples/root_writer.py <input_dir> <output_dir> [debug_dir]

Example:
    python examples/root_writer.py ~/Downloads/root_images ~/Downloads/root_tfrecords ~/Downloads/debug
"""
from __future__ import annotations
import sys
import os
import cv2
import numpy as np
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from ultralytics import FastSAM
except ImportError:
    print("Error: ultralytics package required. Install with: pip install ultralytics")
    sys.exit(1)

from beagle.dataset import (
    Datum,
    write_dataset,
    write_parser_dict,
    serialize_image,
    load_tfr_dict,
)


def find_jpg_files(directory: Path) -> list[str]:
    """Recursively find all JPG files in directory and subdirectories (has side effects: file I/O)."""
    jpg_files: list[str] = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                jpg_files.append(os.path.join(root, file))
    return jpg_files


def image_decoder(dat: Datum) -> Datum:
    """Decode and crop image from file path and bounding box (has side effects: file I/O)."""
    img = cv2.imread(dat.value[0])
    if img is None:
        raise ValueError(f"Could not read image: {dat.value[0]}")
    
    x, y, w_box, h_box = dat.value[1]
    img = img[y:y+h_box, x:x+w_box, :]

    # Resize all to same size for consistency
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

    return Datum(
        name=dat.name,
        value=img,
        serialize_fn=dat.serialize_fn,
        decompress_fn=dat.decompress_fn
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python examples/root_writer.py <input_dir> <output_dir> [debug_dir]")
        sys.exit(1)

    dat_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(exist_ok=True, parents=True)

    if not dat_dir.exists():
        print(f"Error: Input directory {dat_dir} does not exist")
        sys.exit(1)

    debug = False
    debug_dir: Path | None = None
    if len(sys.argv) > 3:
        debug_dir = Path(sys.argv[3])
        debug_dir.mkdir(exist_ok=True, parents=True)
        debug = True

    jpg_files_list = find_jpg_files(dat_dir)
    
    if not jpg_files_list:
        print(f"Error: No JPG files found in {dat_dir}")
        sys.exit(1)
    
    print(f"Found {len(jpg_files_list)} jpg files!")

    try:
        model = FastSAM("FastSAM-s.pt")
    except Exception as e:
        print(f"Error loading FastSAM model: {e}")
        print("Make sure FastSAM-s.pt is in the current directory or model path is correct")
        sys.exit(1)
    
    parseables: list[list[Datum]] = []
    resize_res = 128
    for jpg_path in tqdm(jpg_files_list):
        if 'scale' in jpg_path:
            continue

        print(jpg_path)
        img = cv2.imread(jpg_path)
        if img is None:
            print(f"Warning: Could not read {jpg_path}, skipping")
            continue
            
        h_orig, w_orig = img.shape[:2]
        img_resized = cv2.resize(img, (resize_res, resize_res), interpolation=cv2.INTER_AREA)
        res = model(img_resized, verbose=False)
        
        mask = None
        for r in res:
            masks = r.masks.data.cpu().numpy() 
            if len(masks) > 0:
                mask = masks[0] * 255
                mask = mask.astype(np.uint8)
                break
        
        if mask is None:
            print(f"Warning: No mask found for {jpg_path}, skipping")
            continue
        
        # Get tight bounding box from mask
        h_mask, w_mask = mask.shape[:2]
        coords = cv2.findNonZero(mask)
        if coords is None:
            print(f"Warning: No non-zero coordinates in mask for {jpg_path}, skipping")
            continue
            
        x_mask, y_mask, w_box_mask, h_box_mask = cv2.boundingRect(coords)

        scale_x = w_orig / resize_res
        scale_y = h_orig / resize_res

        # Scale coordinates from mask resolution to original image resolution
        x_low = x_mask * (resize_res / w_mask)
        y_low = y_mask * (resize_res / h_mask)
        w_box_low = w_box_mask * (resize_res / w_mask)
        h_box_low = h_box_mask * (resize_res / h_mask)

        x = int(x_low * scale_x)
        y = int(y_low * scale_y)
        w_box = int(w_box_low * scale_x)
        h_box = int(h_box_low * scale_y)
        
        if debug and debug_dir:
            img_orig = cv2.imread(jpg_path)
            if img_orig is not None:
                cropped = img_orig[y:y+h_box, x:x+w_box]
                cv2.imwrite(str(debug_dir / f"{os.path.basename(jpg_path)}_cropped.png"), cropped)

        parseables.append([
            Datum(
                name="image",
                value=(str(jpg_path), (x, y, w_box, h_box)),
                decompress_fn=image_decoder,
                serialize_fn=serialize_image
            )
        ])

    if not parseables:
        print("Error: No valid images processed")
        sys.exit(1)

    print(f"Writing parser dict to {out_dir / 'root_cell.json'} -- {len(parseables)} parseables ...")
    write_dataset(parseables, str(out_dir.absolute()), num_shards=10)
    write_parser_dict(parseables[0], str(out_dir.absolute()), "root_cell.json")
    print("Done.")

    feature_dict, shape_dict = load_tfr_dict(str(out_dir / 'root_cell.json'))
    
    print(f"Feature dict: {feature_dict}")
    print(f"Shape dict: {shape_dict}")
    
    print("Done.")

