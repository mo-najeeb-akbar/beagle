#!/usr/bin/env python3
"""
High-performance Z-stack to 3D Point Cloud Converter
Optimized for: 32-core CPU, 64GB RAM, RTX 4090
Output: Three.js compatible point cloud formats
"""

import numpy as np
from pathlib import Path
import multiprocessing as mp
from functools import partial
import json
import struct
import re
from typing import Tuple, Optional, List
import argparse
from tqdm import tqdm
import time

# Try to import GPU libraries (CuPy for CUDA acceleration)
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter
    GPU_AVAILABLE = True
    print("✓ GPU acceleration enabled (CuPy)")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠ CuPy not found - running CPU only. Install: pip install cupy-cuda12x")

# Standard scientific computing
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
from skimage import io, filters, morphology, measure
from skimage.filters import threshold_otsu
import warnings
warnings.filterwarnings('ignore')


class PointCloudGenerator:
    """High-performance point cloud generator with GPU acceleration"""
    
    def __init__(self, 
                 n_cores: int = 32,
                 downsample_factor: float = 0.25,
                 morph_radius: int = 2):
        """
        Args:
            n_cores: Number of CPU cores to use
            downsample_factor: Spatial downsampling (0.25 = quarter resolution)
            morph_radius: Radius for morphological operations
        """
        self.n_cores = n_cores
        self.downsample_factor = downsample_factor
        self.morph_radius = morph_radius
        
        # Set up multiprocessing
        mp.set_start_method('spawn', force=True)
        
    def load_and_preprocess_streaming(self, image_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
        """
        Load and preprocess images one at a time to avoid memory explosion
        Returns points, colors, and original shape
        """
        print(f"Processing {len(image_paths)} high-resolution images...")
        
        # First pass: get dimensions and estimate threshold
        print("Pass 1: Analyzing images for thresholding...")
        first_img = io.imread(str(image_paths[0]))
        is_color = first_img.ndim == 3
        
        # Convert to grayscale for thresholding only
        if is_color:
            first_img_gray = np.dot(first_img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        else:
            first_img_gray = first_img
        
        h, w = first_img_gray.shape
        
        # Downsample dimensions
        new_h = int(h * self.downsample_factor)
        new_w = int(w * self.downsample_factor)
        n_slices = len(image_paths)
        
        print(f"Original slice size: {h}×{w}")
        print(f"Downsampled slice size: {new_h}×{new_w}")
        print(f"Processing {n_slices} slices")
        print(f"Color mode: {'RGB' if is_color else 'Grayscale'}")
        
        # Sample images to estimate threshold (use grayscale for thresholding)
        sample_indices = np.linspace(0, len(image_paths)-1, min(20, len(image_paths)), dtype=int)
        sample_values = []
        
        for idx in sample_indices:
            img = io.imread(str(image_paths[idx]))
            # Convert to grayscale for threshold estimation
            if img.ndim == 3:
                img_gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            else:
                img_gray = img
            # Downsample for threshold estimation
            if self.downsample_factor < 1.0:
                from skimage.transform import resize
                img_gray = resize(img_gray, (new_h, new_w), preserve_range=True, anti_aliasing=True)
            sample_values.append(img_gray[img_gray > 10].ravel())  # Ignore very dark pixels
        
        sample_values = np.concatenate(sample_values)
        threshold = threshold_otsu(sample_values)
        print(f"Otsu threshold: {threshold:.1f}")
        
        # Second pass: process each slice and build binary mask
        print("\nPass 2: Processing slices and extracting points with colors...")
        
        # We'll collect points directly instead of building full 3D array
        all_points = []
        all_colors = []
        
        # Use multiprocessing for parallel slice processing
        process_func = partial(
            self._process_single_slice,
            threshold=threshold,
            target_size=(new_h, new_w),
            downsample_factor=self.downsample_factor,
            morph_radius=self.morph_radius,
            preserve_color=is_color
        )
        
        with mp.Pool(self.n_cores) as pool:
            slice_data = []
            for z_idx, (coords, colors) in enumerate(tqdm(
                pool.imap(process_func, [(z, path) for z, path in enumerate(image_paths)]),
                total=len(image_paths),
                desc="Processing slices"
            )):
                if len(coords) > 0:
                    # Add z coordinate
                    z_coords = np.full((len(coords), 1), z_idx)
                    full_coords = np.hstack([z_coords, coords])
                    all_points.append(full_coords)
                    all_colors.append(colors)
        
        # Combine all points
        if not all_points:
            raise ValueError("No points extracted! Try adjusting threshold or downsample factor")
        
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        
        print(f"\n✓ Extracted {len(all_points):,} total points with {'RGB' if is_color else 'grayscale'} colors")
        
        return all_points, all_colors, (n_slices, new_h, new_w)
    
    @staticmethod
    def _load_and_convert_grayscale(path: Path) -> np.ndarray:
        """Load image and convert to grayscale"""
        img = io.imread(str(path))
        if img.ndim == 3:
            # RGB to grayscale
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        return img
    
    @staticmethod
    def _process_single_slice(args, threshold, target_size, downsample_factor, morph_radius, preserve_color=True):
        """Process a single slice: load, threshold, extract points with colors"""
        z_idx, path = args
        
        # Load original image (RGB or grayscale)
        img = io.imread(str(path))
        is_color = img.ndim == 3
        
        # Create grayscale version for thresholding
        if is_color:
            img_gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        else:
            img_gray = img
        
        # Downsample both color and grayscale
        if downsample_factor < 1.0:
            from skimage.transform import resize
            img_gray = resize(img_gray, target_size, preserve_range=True, anti_aliasing=True).astype(np.uint8)
            if is_color and preserve_color:
                img = resize(img, (target_size[0], target_size[1], 3), preserve_range=True, anti_aliasing=True).astype(np.uint8)
        
        # Threshold using grayscale
        binary = img_gray > threshold
        
        # Morphological cleanup
        if morph_radius > 0:
            from skimage.morphology import disk, binary_opening, binary_closing
            footprint = disk(morph_radius)
            binary = binary_opening(binary, footprint)
            binary = binary_closing(binary, footprint)
        
        # Extract point coordinates
        coords = np.argwhere(binary)  # Returns (y, x) coordinates
        
        # Extract colors from original image
        if is_color and preserve_color:
            # Get RGB values at the point locations
            colors = img[binary].astype(np.float32) / 255.0  # Shape: (n_points, 3)
        else:
            # Use grayscale as RGB (all channels same)
            gray_values = img_gray[binary].astype(np.float32) / 255.0
            colors = np.stack([gray_values, gray_values, gray_values], axis=1)
        
        return coords, colors
    
    def threshold_and_clean(self, stack: np.ndarray) -> np.ndarray:
        """Threshold and clean the stack with morphological operations"""
        print("Thresholding...")
        
        # Use Otsu's method on a sample for speed
        sample = stack[::max(1, len(stack)//10)].ravel()
        sample = sample[sample > 0.01]  # Ignore very dark pixels
        threshold = threshold_otsu(sample)
        
        print(f"Otsu threshold: {threshold:.4f}")
        binary = stack > threshold
        
        print("Cleaning with morphological operations...")
        if self.morph_radius > 0:
            # Remove small noise
            footprint = morphology.ball(self.morph_radius)
            binary = binary_opening(binary, footprint)
            # Fill small holes
            binary = binary_closing(binary, footprint)
        
        print(f"✓ Binary mask created: {binary.sum():,} voxels above threshold")
        return binary
    
    def smooth_for_surface(self, stack: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing for smoother surface extraction"""
        if self.smoothing_sigma <= 0:
            return stack
        
        print(f"Smoothing with sigma={self.smoothing_sigma}...")
        
        if self.use_gpu:
            try:
                # Use GPU for smoothing
                stack_gpu = cp.asarray(stack)
                smoothed_gpu = gpu_gaussian_filter(stack_gpu, sigma=self.smoothing_sigma)
                smoothed = cp.asnumpy(smoothed_gpu)
                print("✓ GPU smoothing complete")
                return smoothed
            except Exception as e:
                print(f"GPU smoothing failed ({e}), falling back to CPU")
        
        # CPU fallback
        return gaussian_filter(stack, sigma=self.smoothing_sigma)
    
    def extract_surface_points_marching_cubes(self, 
                                              volume: np.ndarray,
                                              level: float = 0.5,
                                              step_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract high-quality surface using marching cubes
        Returns vertices and faces for mesh, or vertices only for point cloud
        """
        print("Extracting surface with marching cubes...")
        
        try:
            verts, faces, normals, values = measure.marching_cubes(
                volume,
                level=level,
                step_size=step_size,
                allow_degenerate=False
            )
            
            print(f"✓ Extracted {len(verts):,} vertices, {len(faces):,} faces")
            return verts, normals
            
        except Exception as e:
            print(f"Error in marching cubes: {e}")
            return None, None
    
    def extract_volumetric_points(self, 
                                  binary_mask: np.ndarray,
                                  intensity_stack: np.ndarray,
                                  sample_rate: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract points from entire volume with intensity-based sampling
        Alternative to surface-only extraction
        """
        print(f"Extracting volumetric points (sample rate: {sample_rate})...")
        
        # Get coordinates of all non-zero voxels
        coords = np.argwhere(binary_mask)
        intensities = intensity_stack[binary_mask]
        
        print(f"Found {len(coords):,} non-zero voxels")
        
        if sample_rate < 1.0:
            # Intensity-weighted random sampling
            probs = intensities / intensities.sum()
            n_samples = int(len(coords) * sample_rate)
            indices = np.random.choice(len(coords), size=n_samples, replace=False, p=probs)
            coords = coords[indices]
            intensities = intensities[indices]
            print(f"Sampled down to {len(coords):,} points")
        
        return coords, intensities
    
    def save_for_threejs(self,
                        points: np.ndarray,
                        colors: Optional[np.ndarray] = None,
                        output_path: str = "pointcloud",
                        format: str = "json"):
        """
        Save point cloud in Three.js compatible formats
        
        Formats:
        - 'json': JSON format (easy but large)
        - 'binary': Binary format (compact and fast)
        - 'ply': PLY format (standard 3D format)
        """
        output_path = Path(output_path)
        
        if colors is None:
            # Default to white
            colors = np.ones((len(points), 3), dtype=np.float32)
        
        # Normalize coordinates to reasonable range for Three.js
        points = points.astype(np.float32)
        center = points.mean(axis=0)
        points -= center
        scale = np.abs(points).max()
        points /= scale
        
        if format == 'json':
            self._save_json(points, colors, output_path.with_suffix('.json'))
        elif format == 'binary':
            self._save_binary(points, colors, output_path.with_suffix('.bin'))
        elif format == 'ply':
            self._save_ply(points, colors, output_path.with_suffix('.ply'))
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _save_json(self, points: np.ndarray, colors: np.ndarray, path: Path):
        """Save as JSON (simple but large)"""
        print(f"Saving to JSON: {path}")
        
        data = {
            'metadata': {
                'version': 1,
                'type': 'PointCloud',
                'points': len(points)
            },
            'positions': points.flatten().tolist(),
            'colors': colors.flatten().tolist()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
        
        print(f"✓ Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    
    def _save_binary(self, points: np.ndarray, colors: np.ndarray, path: Path):
        """Save as binary format (compact)"""
        print(f"Saving to binary: {path}")
        
        with open(path, 'wb') as f:
            # Header: number of points (uint32)
            f.write(struct.pack('I', len(points)))
            
            # Interleaved: x, y, z, r, g, b for each point
            for i in range(len(points)):
                f.write(struct.pack('fff', *points[i]))
                f.write(struct.pack('fff', *colors[i]))
        
        print(f"✓ Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    
    def _save_ply(self, points: np.ndarray, colors: np.ndarray, path: Path):
        """Save as PLY format (standard 3D format)"""
        print(f"Saving to PLY: {path}")
        
        # Convert colors to 0-255 range
        colors_255 = (colors * 255).astype(np.uint8)
        
        with open(path, 'wb') as f:
            # PLY header
            header = f"""ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
            f.write(header.encode('ascii'))
            
            # Binary data
            for i in range(len(points)):
                f.write(struct.pack('fff', *points[i]))
                f.write(struct.pack('BBB', *colors_255[i]))
        
        print(f"✓ Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='Convert Z-stack to 3D point cloud')
    parser.add_argument('input_dir', type=str, help='Directory containing z-stack images')
    parser.add_argument('--output', type=str, default='pointcloud', help='Output file prefix')
    parser.add_argument('--format', type=str, default='ply', choices=['json', 'binary', 'ply'],
                       help='Output format (default: ply)')
    parser.add_argument('--cores', type=int, default=32, help='Number of CPU cores')
    parser.add_argument('--downsample', type=float, default=0.25, 
                       help='Spatial downsample factor (0.25 = quarter res, recommended for huge images)')
    parser.add_argument('--sample-rate', type=float, default=1.0,
                       help='Point sampling rate after extraction (1.0 = keep all points)')
    parser.add_argument('--morph-radius', type=int, default=2, 
                       help='Morphological operation radius for noise removal')
    
    args = parser.parse_args()
    
    # Find all images in directory
    input_dir = Path(args.input_dir)
    image_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
    
    # Sort by numerical value in filename (e.g., DSC36189.JPG -> 36189)
    import re
    def extract_number(path):
        """Extract first number from filename for sorting"""
        match = re.search(r'\d+', path.stem)
        return int(match.group()) if match else 0
    
    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions],
        key=extract_number
    )
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    print(f"Using {args.cores} CPU cores")
    
    # Initialize generator
    generator = PointCloudGenerator(
        n_cores=args.cores,
        downsample_factor=args.downsample,
        morph_radius=args.morph_radius
    )
    
    # Load and process in streaming fashion
    start_time = time.time()
    
    points, colors, shape = generator.load_and_preprocess_streaming(image_paths)
    
    # Optional: subsample if still too many points
    if args.sample_rate < 1.0:
        print(f"\nSubsampling to {args.sample_rate*100}% of points...")
        n_samples = int(len(points) * args.sample_rate)
        # Uniform random sampling (or could use intensity-weighted)
        indices = np.random.choice(len(points), size=n_samples, replace=False)
        points = points[indices]
        colors = colors[indices]
        print(f"Kept {len(points):,} points")
    
    # Colors are already in RGB format from the images
    
    # Save
    generator.save_for_threejs(points, colors, args.output, args.format)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Total time: {elapsed:.1f} seconds")
    print(f"✓ Output: {args.output}.{args.format}")


if __name__ == '__main__':
    main()