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
import matplotlib.pyplot as plt



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
        depth_map[no_data_mask] = 0.0
    depth_map[depth_map < 0] = 0
    # depth_map = np.nan_to_num(depth_map, nan=0.0)
    depth_map = np.sqrt(depth_map)
    depth_map = np.clip(depth_map, 0.0, 255.0)
    # print(np.min(depth_map), np.max(depth_map))
    # depth_map = np.sqrt(depth_map)
    # depth_map = depth_map - np.mean(depth_map)

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
        
        # print(f"Decoding {path_np} and {md}")
        # image_depth = image_decoder(
        #     Datum(name="surface",
        #     value=(str(path_np), str(md)),
        #     decompress_fn=image_decoder,
        #     serialize_fn=serialize_float_array)
        # ).value
        # # Replace NaN values with 0
        
        
        # image_depth = np.nan_to_num(image_depth, nan=0.0)
        # if np.min(image_depth) > -1 : continue
        # print(np.min(image_depth), np.max(image_depth), np.mean(image_depth), np.std(image_depth))
        # print(np.where(image_depth < 0)[0].sum())
        # image_depth[image_depth < 0] = 0
        # plt.imshow(image_depth)
        # plt.savefig(f'./debug/negative.png')
        # plt.close()
        # import pdb; pdb.set_trace()
        # log_image_depth = np.sqrt(image_depth)
        # log_image_depth = log_image_depth - np.mean(log_image_depth)

        # print(np.mean(image_depth), np.std(image_depth))
        # print(f"Image depth range: {np.min(image_depth):.3f} to {np.max(image_depth):.3f}")
        # fig, axs = plt.subplots(1, 4, figsize=(10, 5))
        # axs[0].imshow(image_depth)
        # # axs[0].colorbar()
        # # axs[0].title(f"Surface: {path_np.parent.name}")
        # axs[1].imshow(log_image_depth)
        # axs[2].hist(log_image_depth.flatten(), bins=100)
        # axs[3].hist(image_depth.flatten(), bins=100)

        # plt.savefig(f'./debug/debug_surface.png')
        # plt.close()
        # import pdb; pdb.set_trace()
        
        parseables.append([
            Datum(
                name="depth",
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

