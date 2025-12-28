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


def load_and_resize_image( dat: Datum) -> Datum:

    img = cv2.imread(dat.value, 0)
    resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    resized = resized
    resized = resized[:, :, np.newaxis]

    return Datum(name=dat.name, value=resized.astype(np.uint8), serialize_fn=dat.serialize_fn, decompress_fn=dat.decompress_fn)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/nema_vae/write_tfrecord.py <output_dir>")
        sys.exit(1)

    image_dir = Path("/data/NemaImageDatasetStandardized/")
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(exist_ok=True, parents=True)

    if not image_dir.exists():
        print(f"Error: Image directory {image_dir} does not exist")
        sys.exit(1)

    img_names = sorted(image_dir.glob("*.jpg"))
    
    
    parseables: list[list[Datum]] = []
    skipped = 0
    
    for image_path in img_names:
        image_path = Path(image_path)

        parseables.append([
            Datum(
                name="image",
                value=str(image_path.absolute()),
                decompress_fn=load_and_resize_image,
                serialize_fn=serialize_image
            ),
        ])
    
    if not parseables:
        print(f"Error: No valid image-mask pairs found")
        sys.exit(1)
    
    print(f"Matched {len(parseables)} images (skipped {skipped})")
    print(f"Writing TFRecords to {out_dir} ...")
    
    write_dataset(parseables, str(out_dir.absolute()), num_shards=50)
    write_parser_dict(parseables[0], str(out_dir.absolute()), "nema_vae.json")
    
    print(f"Wrote {len(parseables)} samples to TFRecords")
    
    feature_dict, shape_dict = load_tfr_dict(str(out_dir / 'nema_vae.json'))
    
    print(f"Feature dict: {feature_dict}")
    print(f"Shape dict: {shape_dict}")
    
    print("Done.")


