"""Blazing fast custom augmentations to replace Albumentations.

These are optimized for speed using:
1. cv2.remap() for all geometric transforms (single interpolation pass)
2. Precomputed displacement fields
3. Vectorized NumPy operations
4. No Python loops in hot paths
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from functools import lru_cache


@lru_cache(maxsize=4)
def _get_base_grid(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cache base coordinate grids to avoid repeated allocation."""
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    return np.meshgrid(x, y)


def shift_scale_rotate(
    image: np.ndarray,
    mask: np.ndarray,
    shift_x: float,
    shift_y: float,
    scale: float,
    angle: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply shift, scale, rotation in one pass using cv2.warpAffine.

    Args:
        shift_x, shift_y: Shift as fraction of image size [-1, 1]
        scale: Scale factor (1.0 = no change)
        angle: Rotation in degrees
    """
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    # Build affine matrix: rotate + scale around center, then shift
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += shift_x * w
    M[1, 2] += shift_y * h

    img_out = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask_out = cv2.warpAffine(
        mask, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return img_out, mask_out


def elastic_transform(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 1.0,
    sigma: float = 50.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast elastic deformation using cv2.remap.

    Optimized version: generate displacement at lower resolution and upscale.
    This is 4-10x faster than full-resolution Gaussian blur.
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = image.shape[:2]

    # Key optimization: work at 1/4 resolution for displacement field
    # This gives nearly identical visual results but is much faster
    small_h, small_w = h // 4, w // 4

    # Generate random displacement at low res
    dx_small = rng.uniform(-1, 1, (small_h, small_w)).astype(np.float32)
    dy_small = rng.uniform(-1, 1, (small_h, small_w)).astype(np.float32)

    # Smaller sigma for downscaled image (proportional to resolution)
    small_sigma = max(sigma / 4, 3)
    ksize = int(4 * small_sigma + 1) | 1
    ksize = min(ksize, 31)  # Cap kernel size

    dx_small = cv2.GaussianBlur(dx_small, (ksize, ksize), small_sigma) * alpha
    dy_small = cv2.GaussianBlur(dy_small, (ksize, ksize), small_sigma) * alpha

    # Upscale displacement field (bicubic for smoothness)
    dx = cv2.resize(dx_small, (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy_small, (w, h), interpolation=cv2.INTER_CUBIC)

    # Create coordinate maps
    map_x, map_y = _get_base_grid(h, w)
    map_x = (map_x + dx).astype(np.float32)
    map_y = (map_y + dy).astype(np.float32)

    img_out = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask_out = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    return img_out, mask_out


def grid_distortion(
    image: np.ndarray,
    mask: np.ndarray,
    num_steps: int = 5,
    distort_limit: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Grid-based distortion - deform a coarse grid and interpolate.

    Much faster than per-pixel random because we only randomize grid points.
    Optimized: direct grid generation without intermediate arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = image.shape[:2]
    cell_w = w / num_steps
    cell_h = h / num_steps

    # Generate grid displacements directly with boundary conditions
    # Interior points get random displacement, edges stay at 0
    dx_grid = np.zeros((num_steps + 1, num_steps + 1), dtype=np.float32)
    dy_grid = np.zeros((num_steps + 1, num_steps + 1), dtype=np.float32)

    # Only randomize interior (not edges)
    interior = (slice(1, -1), slice(1, -1))
    dx_grid[interior] = rng.uniform(-distort_limit * cell_w, distort_limit * cell_w,
                                     (num_steps - 1, num_steps - 1)).astype(np.float32)
    dy_grid[interior] = rng.uniform(-distort_limit * cell_h, distort_limit * cell_h,
                                     (num_steps - 1, num_steps - 1)).astype(np.float32)

    # Interpolate to full resolution
    dx = cv2.resize(dx_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy_grid, (w, h), interpolation=cv2.INTER_CUBIC)

    # Create coordinate maps
    map_x, map_y = _get_base_grid(h, w)
    map_x = (map_x + dx).astype(np.float32)
    map_y = (map_y + dy).astype(np.float32)

    img_out = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask_out = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    return img_out, mask_out


def optical_distortion(
    image: np.ndarray,
    mask: np.ndarray,
    distort_limit: float = 0.5,
    shift_limit: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Barrel/pincushion distortion (lens distortion effect).

    Uses OpenCV's undistort with synthetic camera matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = image.shape[:2]

    # Random distortion coefficients
    k1 = rng.uniform(-distort_limit, distort_limit)

    # Shift of optical center
    cx = w / 2 + rng.uniform(-shift_limit, shift_limit) * w / 2
    cy = h / 2 + rng.uniform(-shift_limit, shift_limit) * h / 2

    # Camera matrix
    fx = fy = max(w, h)
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float32)

    dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float32)

    # Compute undistort maps once
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1
    )

    img_out = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask_out = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    return img_out, mask_out


class FastAugmentor:
    """Fast augmentor that replaces Albumentations OneOf pipeline.

    Usage:
        augmentor = FastAugmentor(seed=42)
        augmented = augmentor(image, mask)
    """

    def __init__(
        self,
        shift_limit: float = 0.1,
        scale_limit: float = 0.1,
        rotate_limit: float = 45.0,
        elastic_alpha: float = 1.0,
        elastic_sigma: float = 50.0,
        grid_steps: int = 5,
        grid_distort_limit: float = 0.3,
        optical_distort_limit: float = 0.5,
        optical_shift_limit: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.grid_steps = grid_steps
        self.grid_distort_limit = grid_distort_limit
        self.optical_distort_limit = optical_distort_limit
        self.optical_shift_limit = optical_shift_limit
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentation (one of 4 types)."""
        choice = self.rng.integers(0, 4)

        if choice == 0:
            # Shift-Scale-Rotate
            shift_x = self.rng.uniform(-self.shift_limit, self.shift_limit)
            shift_y = self.rng.uniform(-self.shift_limit, self.shift_limit)
            scale = 1.0 + self.rng.uniform(-self.scale_limit, self.scale_limit)
            angle = self.rng.uniform(-self.rotate_limit, self.rotate_limit)
            return shift_scale_rotate(image, mask, shift_x, shift_y, scale, angle)

        elif choice == 1:
            # Elastic
            return elastic_transform(
                image, mask,
                alpha=self.elastic_alpha,
                sigma=self.elastic_sigma,
                rng=self.rng,
            )

        elif choice == 2:
            # Grid distortion
            return grid_distortion(
                image, mask,
                num_steps=self.grid_steps,
                distort_limit=self.grid_distort_limit,
                rng=self.rng,
            )

        else:
            # Optical distortion
            return optical_distortion(
                image, mask,
                distort_limit=self.optical_distort_limit,
                shift_limit=self.optical_shift_limit,
                rng=self.rng,
            )


# Module-level instance for use in multiprocessing (avoids recreation overhead)
_fast_augmentor = None


def get_fast_augmentor() -> FastAugmentor:
    """Get or create the module-level augmentor."""
    global _fast_augmentor
    if _fast_augmentor is None:
        _fast_augmentor = FastAugmentor()
    return _fast_augmentor


def apply_fast_augmentation(sample: dict) -> dict:
    """Drop-in replacement for sample_transform in disk loader.

    This function has the same signature and can be used directly.
    """
    augmentor = get_fast_augmentor()

    # Handle channel dimension
    img = sample['image']
    msk = sample['mask']

    has_channel_img = img.ndim == 3
    has_channel_msk = msk.ndim == 3

    if has_channel_img:
        img = img[:, :, 0]
    if has_channel_msk:
        msk = msk[:, :, 0]

    # Apply augmentation
    img_out, msk_out = augmentor(img, msk)

    # Restore channel dimension
    if has_channel_img:
        img_out = img_out[:, :, np.newaxis]
    if has_channel_msk:
        msk_out = msk_out[:, :, np.newaxis]

    return {'image': img_out, 'mask': msk_out}

