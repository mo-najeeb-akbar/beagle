"""
Example: Reproducible training with deterministic data pipeline.

Demonstrates how to set seeds for fully reproducible results across:
- Train/val splits
- Dataset shuffling
- Data augmentations
"""
from beagle.dataset import create_iterator, set_global_seed
from beagle.augmentations import compose, random_flip_left_right, random_brightness


def create_augment_fn():
    """Create TensorFlow augmentation function."""
    def augment(data_dict):
        img = data_dict['image']
        # Random flips
        img = random_flip_left_right(fields=['image'])(data_dict)['image']
        # Random brightness
        img = random_brightness(0.2)(data_dict)['image']
        data_dict['image'] = img
        return data_dict
    return augment


def main():
    # IMPORTANT: Set seed BEFORE creating iterators
    set_global_seed(42)
    
    # Now all randomness is deterministic and reproducible:
    # 1. Train/val file split uses Python's random (seeded)
    # 2. Dataset shuffling uses TensorFlow's random (seeded)
    # 3. Augmentations use TensorFlow's random (seeded)
    
    (train_iter, train_batches), (val_iter, val_batches) = create_iterator(
        "data/*.tfrecord",
        batch_size=32,
        val_split=0.2,           # Deterministic split
        shuffle=True,             # Deterministic shuffling
        augment_fn=create_augment_fn(),  # Deterministic augmentations
        repeat=True,
    )
    
    print(f"Train batches per epoch: {train_batches}")
    print(f"Val batches per epoch: {val_batches}")
    
    # Get first batch - will be identical every run
    batch = next(train_iter)
    print(f"First batch shape: {batch['image'].shape}")
    print(f"First batch mean: {batch['image'].mean():.6f}")
    
    # Running this script multiple times will give identical results!
    
    # For even stricter determinism (at cost of performance):
    # from beagle.dataset import set_tf_deterministic
    # set_tf_deterministic(True)


if __name__ == "__main__":
    main()

