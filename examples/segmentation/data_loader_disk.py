from beagle.dataset import (
    load_fields_from_disk,
    create_disk_iterator,
)
from beagle.augmentations import (
    compose,
    random_flip_left_right,
    random_flip_up_down,
    random_rotate_90,
    random_brightness,
    random_contrast,
)
from pathlib import Path
import numpy as np
import cv2


# ============================================================================
# Example 1: Segmentation (image + mask)
# ============================================================================

def load_image(path: str) -> np.ndarray:
    img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)[:, :, np.newaxis], (512, 512), interpolation=cv2.INTER_LINEAR)
    return img[:, :, np.newaxis]

def load_mask(path: str) -> np.ndarray:
    mask = cv2.resize((cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)[:, :, np.newaxis], (512, 512), interpolation=cv2.INTER_NEAREST)
    return mask[:, :, np.newaxis]

def segmentation_example(dataset_path):
    """Load images and masks for semantic segmentation."""

    image_dir = Path(dataset_path)
    mask_files = sorted(image_dir.glob("*_mask.png"))
    
    file_list = []
    for mask_path in mask_files:
        image_path = Path(str(mask_path).split('_mask.png')[0] + '.jpg')
        file_list.append({'image': image_path, 'mask': mask_path})
    # Load data
    data = load_fields_from_disk(
        file_list=file_list,
        field_loaders={
            'image': load_image,
            'mask': load_mask,
        },
        field_transforms={
            'image': lambda x: (x - 127.5)/ 127.5,  # [-1, 1]
            'mask': lambda x: x,  # Keep masks as-is (class indices)
        }
    )

    # Define augmentation (applied to both image and mask)
    aug_fn = compose(
        random_flip_left_right(fields=['image', 'mask']),
        random_flip_up_down(fields=['image', 'mask']),
        random_rotate_90(fields=['image', 'mask']),
        # Only augment image photometrically
        random_brightness(max_delta=0.1, field='image'),
        random_contrast(lower=0.9, upper=1.1, field='image'),
    )

    # Create iterator with heavy augmentation
    iterator = create_disk_iterator(
        data=data,
        batch_size=128,
        shuffle=True,
        repeat=True,
        augment_fn=aug_fn,
    )

    # repeats infinitely, so no need to compute batches_per_epoch

    # Train
    for epoch in range(100):
        for _ in range(100):
            batch = next(iterator)
            print(batch['image'].shape)
            print(batch['mask'].shape)
            pass


if __name__ == '__main__':
    dataset_path = "/data/nema_imgs_masks"
    segmentation_example(dataset_path)
