#!/usr/bin/env python3
"""
Extract overlapping square tiles from a GeoTIFF
Saves all tiles as JPG images to a specified folder
"""

import cv2
import sys
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path

def extract_tiles(input_tif, output_dir, tile_size=512, overlap=0, quality=95, resize_factor=1):
    """
    Extract overlapping square tiles from a GeoTIFF
    
    Args:
        input_tif: Path to input GeoTIFF file
        output_dir: Directory to save tiles
        tile_size: Size of square tiles to extract (pixels)
        overlap: Overlap between tiles (pixels)
        quality: JPEG quality (0-100)
        resize_factor: Factor to resize tiles before saving (1=no resize, 2=half size, 4=quarter size)
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Opening {input_tif}...")
    
    with rasterio.open(input_tif) as src:
        img_width = src.width
        img_height = src.height
        num_bands = src.count
        
        output_size = tile_size // resize_factor
        
        print(f"Image size: {img_width} x {img_height} pixels")
        print(f"Bands: {num_bands}")
        print(f"Tile size: {tile_size}x{tile_size} pixels")
        print(f"Output size: {output_size}x{output_size} pixels (resize factor: {resize_factor})")
        print(f"Overlap: {overlap} pixels")
        
        # Calculate step size (tile_size - overlap)
        step = tile_size - overlap
        
        # Calculate number of tiles
        tiles_x = (img_width + step - 1) // step
        tiles_y = (img_height + step - 1) // step
        
        total_tiles = tiles_x * tiles_y
        print(f"Will extract {tiles_x} x {tiles_y} = {total_tiles} tiles")
        print(f"Output directory: {output_dir}")
        print()
        
        tile_count = 0
        
        # Extract tiles
        for row in range(tiles_y):
            y = row * step
            
            for col in range(tiles_x):
                x = col * step
                
                # Determine actual tile dimensions (handle edge cases)
                actual_width = min(tile_size, img_width - x)
                actual_height = min(tile_size, img_height - y)
                
                # Skip tiles that are too small
                if actual_width < tile_size // 2 or actual_height < tile_size // 2:
                    continue
                
                # Read the tile
                window = Window(x, y, actual_width, actual_height)
                tile_data = src.read(window=window)
                
                # Convert from (bands, height, width) to (height, width, bands)
                if num_bands == 1:
                    tile = tile_data[0]
                else:
                    tile = np.moveaxis(tile_data, 0, -1)
                    # Convert RGB to BGR for OpenCV
                    if num_bands == 3:
                        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                    elif num_bands == 4:
                        tile = cv2.cvtColor(tile, cv2.COLOR_RGBA2BGRA)
                
                # Handle different data types
                if tile.dtype == np.uint16:
                    tile = (tile / 256).astype(np.uint8)
                elif tile.dtype == np.float32 or tile.dtype == np.float64:
                    tile = ((tile - tile.min()) / (tile.max() - tile.min()) * 255).astype(np.uint8)
                
                # Pad if necessary to make square
                if actual_width != tile_size or actual_height != tile_size:
                    if num_bands > 1:
                        padded = np.zeros((tile_size, tile_size, num_bands), dtype=np.uint8)
                        padded[:actual_height, :actual_width] = tile
                    else:
                        padded = np.zeros((tile_size, tile_size), dtype=np.uint8)
                        padded[:actual_height, :actual_width] = tile
                    tile = padded
                
                # Resize if requested
                if resize_factor > 1:
                    output_size = tile_size // resize_factor
                    tile = cv2.resize(tile, (output_size, output_size), interpolation=cv2.INTER_AREA)
                
                # Save as JPEG
                output_path = os.path.join(output_dir, f"tile_r{row:04d}_c{col:04d}_x{x}_y{y}.jpg")
                cv2.imwrite(output_path, tile, [cv2.IMWRITE_JPEG_QUALITY, quality])
                
                tile_count += 1
                if tile_count % 100 == 0:
                    print(f"Processed {tile_count}/{total_tiles} tiles...")
        
        print(f"\nDone! Extracted {tile_count} tiles to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_tiles.py <input.tif> <output_dir> [tile_size] [overlap] [quality] [resize_factor]")
        print("\nExamples:")
        print("  python extract_tiles.py input.tif ./tiles")
        print("  python extract_tiles.py input.tif ./tiles 512 64")
        print("  python extract_tiles.py input.tif ./tiles 4000 0 95 4")
        print("  python extract_tiles.py input.tif ./tiles 2048 256 90 2")
        print("\nParameters:")
        print("  tile_size: Size of square tiles to extract in pixels (default: 512)")
        print("  overlap: Overlap between tiles in pixels (default: 0)")
        print("  quality: JPEG quality 0-100 (default: 95)")
        print("  resize_factor: Downscale factor (default: 1, no resize)")
        print("                 2 = half size, 4 = quarter size, etc.")
        print("\nExample with resize:")
        print("  Extract 4000x4000 tiles and save as 1000x1000:")
        print("  python extract_tiles.py input.tif ./tiles 4000 0 95 4")
        sys.exit(1)
    
    input_tif = sys.argv[1]
    output_dir = sys.argv[2]
    tile_size = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    overlap = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    quality = int(sys.argv[5]) if len(sys.argv) > 5 else 95
    resize_factor = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    
    extract_tiles(input_tif, output_dir, tile_size, overlap, quality, resize_factor)