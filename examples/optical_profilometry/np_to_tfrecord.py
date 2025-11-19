"""
Example: Polymer depth map dataset writer.

Converts polymer depth map data (numpy arrays + metadata) into TFRecord format.
Handles no-data values by converting them to NaN.

Usage:
    python examples/polymer_np_to_tfrecord.py <input_dir> <output_dir>

Example:
    python examples/polymer_np_to_tfrecord.py ~/Downloads/polymer_data ~/Downloads/polymer_tfrecords
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
from beagle.dataset import (
    Datum,
    write_dataset,
    write_parser_dict,
    serialize_float_array,
    load_tfr_dict,
)


def image_decoder(dat: Datum) -> Datum:
    """Decode depth map from file paths and apply no-data mask (pure function with side effects: file I/O)."""
    depth_map = np.load(dat.value[0])
    dataset_metadata = None
    with open(dat.value[1], 'r') as f:
        dataset_metadata = json.load(f)

    no_data_mask = (depth_map == dataset_metadata['surface']['no_data_value'])
    no_data_count = np.sum(no_data_mask)
    
    if no_data_count > 0:
        depth_map = depth_map.copy()
        depth_map[no_data_mask] = np.nan

    return Datum(
        name=dat.name,
        value=depth_map,
        serialize_fn=dat.serialize_fn,
        decompress_fn=dat.decompress_fn
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python examples/polymer_np_to_tfrecord.py <input_dir> <output_dir>")
        sys.exit(1)

    dat_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(exist_ok=True, parents=True)

    if not dat_dir.exists():
        print(f"Error: Input directory {dat_dir} does not exist")
        sys.exit(1)

    depth_maps = [item for item in dat_dir.iterdir() if item.is_dir()]
    
    parseables: list[list[Datum]] = []
    for depth_map_path in depth_maps:
        path_np = (depth_map_path / "surface.npy").absolute()
        md = (depth_map_path / "dataset_metadata.json").absolute()
        
        if not path_np.exists() or not md.exists():
            print(f"Warning: Skipping {depth_map_path} - missing surface.npy or dataset_metadata.json")
            continue
        
        parseables.append([
            Datum(
                name="surface",
                value=(str(path_np), str(md)),
                decompress_fn=image_decoder,
                serialize_fn=serialize_float_array
            )
        ])
    
    if not parseables:
        print(f"Error: No valid depth map directories found in {dat_dir}")
        sys.exit(1)
    
    print(f"Writing parser dict to {out_dir / 'polymer.json'} -- {len(parseables)} parseables ...")
    write_dataset(parseables, str(out_dir.absolute()), num_shards=10)
    write_parser_dict(parseables[0], str(out_dir.absolute()), "polymer.json")
    print("Done.")

    feature_dict, shape_dict = load_tfr_dict(str(out_dir / 'polymer.json'))
    
    print(f"Feature dict: {feature_dict}")
    print(f"Shape dict: {shape_dict}")
    
    print("Done.")

