"""Run batch inference on PNG images using trained mask model - GPU OPTIMIZED VERSION.

This version maximizes GPU utilization by:
1. Moving preprocessing to GPU
2. Pipelining batch loading with inference
3. Using larger batch sizes
4. Minimizing Python overhead

Run: python batch_inference_gpu_optimized.py /data/experiments/hrnet_masks_XXX /data/png_input /data/output --batch_size 128
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import image as jax_image
from PIL import Image
from scipy.ndimage import binary_opening

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from beagle.training import TrainState, load_checkpoint, load_config
from beagle.dataset import load_field_stats
from beagle.network.hrnet import MoNet


class ImageLoader:
    """Asynchronous image loader that prefetches batches."""
    
    def __init__(self, image_paths: list[Path], batch_size: int, target_size: int, num_workers: int = 8):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_workers = num_workers
        self.queue = Queue(maxsize=3)  # Prefetch up to 3 batches
        self.num_batches = (len(image_paths) + batch_size - 1) // batch_size
        
    def load_single_image(self, path: Path):
        """Load single image - fast version."""
        try:
            img = Image.open(path)
            if img.mode != 'L':
                img = img.convert('L')
            
            original_array = np.array(img, dtype=np.uint8)
            original_size = (img.height, img.width)
            
            # Resize
            img_resized = img.resize((self.target_size, self.target_size), Image.BILINEAR)
            img_array = np.array(img_resized, dtype=np.float32)
            
            # Normalize to [-1, 1] - do this on GPU later
            # img_array = (img_array - 127.5) / 127.5
            
            return img_array, original_size, original_array, path
        except Exception as e:
            return None
    
    def load_batch(self, batch_paths: list[Path]):
        """Load a batch of images in parallel."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self.load_single_image, batch_paths))
        
        # Filter valid results
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            return None
        
        images = np.stack([r[0] for r in valid_results], axis=0)
        original_sizes = [r[1] for r in valid_results]
        original_images = [r[2] for r in valid_results]
        paths = [r[3] for r in valid_results]
        
        return {
            'images': images,
            'original_sizes': original_sizes,
            'original_images': original_images,
            'paths': paths
        }
    
    def producer(self):
        """Producer thread that loads batches asynchronously."""
        for batch_idx in range(self.num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.image_paths))
            batch_paths = self.image_paths[start_idx:end_idx]
            
            batch_data = self.load_batch(batch_paths)
            self.queue.put((batch_idx, batch_data))
        
        # Signal completion
        self.queue.put(None)
    
    def start(self):
        """Start the producer thread."""
        self.thread = Thread(target=self.producer, daemon=True)
        self.thread.start()
    
    def __iter__(self):
        """Iterate over batches."""
        while True:
            item = self.queue.get()
            if item is None:
                break
            yield item


@jax.jit
def preprocess_batch_gpu(images: jnp.ndarray) -> jnp.ndarray:
    """Normalize images on GPU - much faster than CPU."""
    # Normalize to [-1, 1]
    normalized = (images - 127.5) / 127.5
    # Add channel dimension [B, H, W, 1]
    return normalized[:, :, :, jnp.newaxis]


def create_inference_fn(model):
    """Create a JIT-compiled inference function."""
    @jax.jit
    def inference_batch(params, batch_stats, images: jnp.ndarray):
        """JIT-compiled inference function."""
        predictions = model.apply(
            {"params": params, "batch_stats": batch_stats},
            images,
            train=False
        )
        
        pred_mask_full = predictions[0]
        pred_mask_full = jax.nn.softmax(pred_mask_full, axis=-1)
        pred_mask_full = jnp.argmax(pred_mask_full, axis=-1)
        
        return pred_mask_full
    
    return inference_batch


def resize_masks_gpu(masks_jax: jnp.ndarray, original_sizes: list[tuple[int, int]]) -> list[np.ndarray]:
    """Resize masks on GPU, grouped by target size."""
    # Group by unique sizes
    size_to_indices = {}
    for idx, size in enumerate(original_sizes):
        if size not in size_to_indices:
            size_to_indices[size] = []
        size_to_indices[size].append(idx)
    
    # Process each group on GPU
    resized_masks = [None] * len(masks_jax)
    
    for target_size, indices in size_to_indices.items():
        # Extract masks for this size (on GPU)
        batch_masks = masks_jax[jnp.array(indices)]
        
        # Resize on GPU
        batch_4d = batch_masks[:, :, :, jnp.newaxis]
        resized = jax_image.resize(
            batch_4d,
            shape=(len(indices), target_size[0], target_size[1], 1),
            method="nearest"
        )
        
        # Transfer to CPU only once per group
        resized_np = np.array(resized[:, :, :, 0])
        
        for i, idx in enumerate(indices):
            resized_masks[idx] = resized_np[i]
    
    return resized_masks


def crop_and_mask_batch(
    original_images: list[np.ndarray],
    masks: list[np.ndarray],
    padding: int = 10
) -> list[tuple[np.ndarray | None, Path]]:
    """Fast batch cropping and masking - keep only largest connected component."""
    import cv2
    results = []
    
    for img, mask in zip(original_images, masks):
        # Threshold mask
        mask_binary = (mask == 1).astype(np.uint8)
        
        # Find connected components (OpenCV is MUCH faster than scipy)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
        
        if num_labels <= 1:  # Only background
            results.append(None)
            continue
        
        # Find largest component (skip label 0 which is background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_component = areas.argmax() + 1  # +1 because we skipped background
        
        # Keep only the largest component
        binary_mask = (labels == largest_component).astype(np.uint8)
        
        # Find bounding box - vectorized
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        
        if not rows.any() or not cols.any():
            results.append(None)
            continue
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Add padding
        h, w = mask.shape
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding)
        y_max = min(h - 1, y_max + padding)
        x_max = min(w - 1, x_max + padding)
        
        # Crop
        cropped_img = img[y_min:y_max+1, x_min:x_max+1]
        cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
        
        # Apply mask
        if cropped_img.ndim == 3:
            cropped_mask = cropped_mask[..., np.newaxis]
        
        masked_img = cropped_img * (cropped_mask == 1)
        results.append(masked_img.astype(np.uint8))
    
    return results


class AsyncImageSaver:
    """Asynchronous image saver."""
    
    def __init__(self, num_workers: int = 8):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.futures = []
    
    def save(self, img_array: np.ndarray, path: Path):
        """Queue an image to save."""
        future = self.executor.submit(self._save_single, img_array, path)
        self.futures.append(future)
    
    @staticmethod
    def _save_single(img_array: np.ndarray, path: Path):
        try:
            Image.fromarray(img_array).save(path)
            return True
        except Exception as e:
            print(f"Error saving {path}: {e}")
            return False
    
    def wait_all(self):
        """Wait for all saves to complete and return count."""
        results = [f.result() for f in self.futures]
        self.futures.clear()
        return sum(results)
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


def process_with_pipeline(
    image_loader: ImageLoader,
    inference_fn,
    params,
    batch_stats,
    output_dir: Path,
    padding: int,
    saver: AsyncImageSaver
):
    """Process batches with pipelined loading and inference."""
    total_processed = 0
    total_skipped = 0
    
    batch_iter = image_loader
    if HAS_TQDM:
        batch_iter = tqdm(image_loader, total=image_loader.num_batches, desc="Processing")
    
    for batch_idx, batch_data in batch_iter:
        if batch_data is None:
            continue
        
        # STEP 1: Move to GPU and preprocess (GPU operation)
        images_gpu = jnp.array(batch_data['images'])
        images_normalized = preprocess_batch_gpu(images_gpu)
        
        # STEP 2: Run inference (GPU operation)
        masks_pred = inference_fn(params, batch_stats, images_normalized)
        
        # STEP 3: Resize masks (GPU operation, minimal CPU transfer)
        resized_masks = resize_masks_gpu(masks_pred, batch_data['original_sizes'])
        
        # STEP 4: Crop and mask (CPU operation, but fast)
        cropped_images = crop_and_mask_batch(
            batch_data['original_images'],
            resized_masks,
            padding
        )
        
        # STEP 5: Queue saves asynchronously (doesn't block)
        for path, cropped_img in zip(batch_data['paths'], cropped_images):
            if cropped_img is not None:
                output_path = output_dir / f"{path.stem}_cropped.jpg"
                saver.save(cropped_img, output_path)
                total_processed += 1
            else:
                total_skipped += 1
        
        if HAS_TQDM:
            batch_iter.set_postfix({
                'processed': total_processed,
                'skipped': total_skipped
            })
    
    # Wait for all saves to complete
    saver.wait_all()
    
    return total_processed, total_skipped


def run_batch_inference(
    checkpoint_dir: Path,
    input_dir: Path,
    output_dir: Path,
    batch_size: int = 128,
    padding: int = 10,
    num_io_workers: int = 16,
    prefetch_batches: int = 3
) -> None:
    """Run GPU-optimized batch inference.
    
    Args:
        checkpoint_dir: Path to experiment directory with checkpoints
        input_dir: Directory containing image files
        output_dir: Directory to save results
        batch_size: Number of images to process in parallel (can be much larger now!)
        padding: Padding around tight crop bbox
        num_io_workers: Number of parallel I/O workers
        prefetch_batches: Number of batches to prefetch
    """
    # Load config
    config = load_config(str(checkpoint_dir))
    print(f"Loaded config from {checkpoint_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Batch size: {batch_size}")
    print(f"I/O workers: {num_io_workers}")
    print(f"Prefetch batches: {prefetch_batches}")

    if config["upsample_to_full_res"]:
        outputs = [(3, False, 2)]
    else:
        outputs = [(3, False)]
    
    # Initialize model
    model = MoNet(
        num_stages=config["num_stages"],
        features=config["features"],
        target_res=config["target_res"],
        train_bb=config["train_backbone"],
        outputs=outputs,
    )
    
    # Initialize model variables
    key = random.key(42)
    dummy = jnp.ones((1, config["input_size"], config["input_size"], 1))
    variables = model.init(key, dummy, train=False)
    
    # Find and load checkpoint
    checkpoint_path = checkpoint_dir / "checkpoint_final"
    if not checkpoint_path.exists():
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        checkpoint_path = checkpoints[-1]
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model_data = load_checkpoint(str(checkpoint_path))
    params = model_data['params']
    batch_stats = model_data['batch_stats']
    print(f"Checkpoint loaded successfully")
    
    # Create JIT-compiled inference function
    print("Creating inference function...")
    inference_fn = create_inference_fn(model)
    
    # Warm up JIT compilation with actual batch size
    print("Warming up JIT compilation...")
    warmup_batch = jnp.ones((batch_size, config["input_size"], config["input_size"], 1))
    _ = inference_fn(params, batch_stats, warmup_batch)
    _ = preprocess_batch_gpu(warmup_batch[:, :, :, 0])  # Remove channel for warmup
    print("JIT compilation complete")
    
    # Find all image files
    image_patterns = ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]
    image_files = []
    for pattern in image_patterns:
        image_files.extend(input_dir.glob(pattern))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Create image loader with prefetching
    loader = ImageLoader(
        image_files,
        batch_size=batch_size,
        target_size=config["input_size"],
        num_workers=num_io_workers
    )
    
    # Create async saver
    saver = AsyncImageSaver(num_workers=num_io_workers)
    
    # Start prefetching
    print("Starting pipelined processing...")
    loader.start()
    
    # Process with pipeline
    total_processed, total_skipped = process_with_pipeline(
        loader,
        inference_fn,
        params,
        batch_stats,
        output_dir,
        padding,
        saver
    )
    
    # Cleanup
    saver.shutdown()
    
    print(f"\nBatch inference complete!")
    print(f"Processed {total_processed} images successfully")
    if total_skipped > 0:
        print(f"Skipped {total_skipped} invalid/unreadable images or images with no mask")
    print(f"Cropped and masked images saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Run GPU-optimized batch inference on images'
    )
    parser.add_argument('checkpoint_dir', type=str, help='Directory containing checkpoint and config')
    parser.add_argument('input_dir', type=str, help='Directory with image files to process')
    parser.add_argument('output_dir', type=str, help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='Batch size for inference (increase this for better GPU util!)')
    parser.add_argument('--padding', type=int, default=10, help='Padding around tight crop bbox')
    parser.add_argument('--num_io_workers', type=int, default=16, 
                       help='Number of parallel I/O workers')
    parser.add_argument('--prefetch_batches', type=int, default=3,
                       help='Number of batches to prefetch')
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory does not exist: {checkpoint_dir}")
        sys.exit(1)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    run_batch_inference(
        checkpoint_dir,
        input_dir,
        output_dir,
        batch_size=args.batch_size,
        padding=args.padding,
        num_io_workers=args.num_io_workers,
        prefetch_batches=args.prefetch_batches
    )


if __name__ == "__main__":
    main()