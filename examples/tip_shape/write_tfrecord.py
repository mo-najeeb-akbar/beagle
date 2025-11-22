"""
Example: Root tip mask and image dataset writer.

Converts root tip masks (PNG) and corresponding TIFF images into TFRecord format.
Resizes all images to 512x512.

Usage:
    python examples/tip_shape/write_tfrecord.py <output_dir>

Example:
    make run CMD='python examples/tip_shape/write_tfrecord.py /data/output'
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
from beagle.dataset import (
    Datum,
    write_dataset,
    write_parser_dict,
    serialize_image,
    load_tfr_dict,
)


def load_and_resize_mask(dat: Datum) -> Datum:

    img = cv2.imread(dat.value, 0)
    resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
    resized = resized > 0.0
    resized = resized[:, :, np.newaxis]

    return Datum(name=dat.name, value=resized.astype(np.uint8), serialize_fn=dat.serialize_fn, decompress_fn=dat.decompress_fn)

def load_and_resize_image( dat: Datum) -> Datum:

    img = cv2.imread(dat.value, 0)
    resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    resized = resized
    resized = resized[:, :, np.newaxis]

    return Datum(name=dat.name, value=resized.astype(np.uint8), serialize_fn=dat.serialize_fn, decompress_fn=dat.decompress_fn)


def find_matching_image(mask_path: Path, image_dir: Path) -> Path | None:
    """Find corresponding TIFF image for a mask file (pure function with side effects: file I/O)."""
    mask_stem = mask_path.stem.split('_mask')[0]
    
    for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
        image_path = image_dir / f"{mask_stem}{ext}"
        if image_path.exists():
            return image_path
    
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/tip_shape/write_tfrecord.py <output_dir>")
        sys.exit(1)

    mask_dir = Path("/data/RootTipMasks")
    image_dir = Path("/data/Wheat_Exome_JSS_2025/Exome_Panel_Rep_2")
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(exist_ok=True, parents=True)

    if not mask_dir.exists():
        print(f"Error: Mask directory {mask_dir} does not exist")
        sys.exit(1)
    
    if not image_dir.exists():
        print(f"Error: Image directory {image_dir} does not exist")
        sys.exit(1)

    mask_files = sorted(mask_dir.glob("*.png"))
    
    if not mask_files:
        print(f"Error: No PNG mask files found in {mask_dir}")
        sys.exit(1)
    
    print(f"Found {len(mask_files)} mask files")
    
    parseables: list[list[Datum]] = []
    skipped = 0
    
    for mask_path in mask_files:
        image_path = find_matching_image(mask_path, image_dir)
        
        if image_path is None:
            print(f"Warning: No matching image found for {mask_path.name}")
            skipped += 1
            continue
        
        parseables.append([
            Datum(
                name="image",
                value=str(image_path.absolute()),
                decompress_fn=load_and_resize_image,
                serialize_fn=serialize_image
            ),
            Datum(
                name="mask",
                value=str(mask_path.absolute()),
                decompress_fn=load_and_resize_mask,
                serialize_fn=serialize_image
            )
        ])
    
    if not parseables:
        print(f"Error: No valid image-mask pairs found")
        sys.exit(1)
    
    print(f"Matched {len(parseables)} image-mask pairs (skipped {skipped})")
    print(f"Writing TFRecords to {out_dir} ...")
    
    write_dataset(parseables, str(out_dir.absolute()), num_shards=10)
    write_parser_dict(parseables[0], str(out_dir.absolute()), "root_tip.json")
    
    print(f"Wrote {len(parseables)} samples to TFRecords")
    
    feature_dict, shape_dict = load_tfr_dict(str(out_dir / 'root_tip.json'))
    
    print(f"Feature dict: {feature_dict}")
    print(f"Shape dict: {shape_dict}")
    
    print("Done.")


