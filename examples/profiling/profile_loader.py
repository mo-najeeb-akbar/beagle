"""
Profile dataloader performance and identify bottlenecks.

Uses TensorFlow's profiling tools to time each stage of the pipeline.

Usage:
    python examples/profile_loader.py ~/data/polymer_tfrecords
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from functools import partial

import tensorflow as tf
import numpy as np

from beagle.dataset import load_tfr_dict, create_iterator
from beagle.augmentations import create_transform, MODERATE_AUGMENT, apply_transform


def make_polymer_parser(
    feature_dict: dict[str, tf.io.FixedLenFeature],
    shape_dict: dict[str, list[int]],
):
    """Create parser for polymer TFRecords (pure)."""
    def parse(example_proto: tf.Tensor) -> dict[str, tf.Tensor]:
        parsed = tf.io.parse_single_example(example_proto, feature_dict)
        surface = tf.reshape(parsed['surface'], shape_dict['surface'] + [1])
        surface = tf.where(tf.math.is_nan(surface), 0.0, surface)
        return {'image': surface}
    
    return parse


def benchmark_pipeline(
    dataset: tf.data.Dataset,
    num_batches: int = 50,
    warmup: int = 5,
) -> dict[str, float]:
    """Benchmark pipeline throughput."""
    # Warmup
    iterator = iter(dataset)
    for _ in range(warmup):
        next(iterator)
    
    # Time
    start = time.perf_counter()
    for _ in range(num_batches):
        batch = next(iterator)
        # Force computation
        if isinstance(batch, dict):
            # Parsed data - materialize first field
            field_name = list(batch.keys())[0]
            _ = batch[field_name].numpy()
        else:
            # Raw TFRecords - just materialize the tensor
            _ = batch.numpy()
    
    elapsed = time.perf_counter() - start
    
    return {
        'total_time': elapsed,
        'time_per_batch': elapsed / num_batches,
        'batches_per_sec': num_batches / elapsed,
    }


def profile_stages(
    tfrecord_pattern: str,
    parser,
    img_height: int,
    img_width: int,
    crop_size: int = 256,
    stride: int = 192,
    batch_size: int = 32,
) -> None:
    """Profile each stage of the pipeline."""
    import glob
    files = sorted(glob.glob(tfrecord_pattern))
    
    print("=" * 60)
    print("PROFILING DATALOADER PIPELINE")
    print("=" * 60)
    
    # Stage 1: Raw TFRecord reading
    print("\n[Stage 1] Raw TFRecord Reading")
    ds1 = tf.data.TFRecordDataset(files)
    stats1 = benchmark_pipeline(ds1.batch(batch_size).prefetch(2), num_batches=20)
    print(f"  Throughput: {stats1['batches_per_sec']:.2f} batches/sec")
    print(f"  Time/batch: {stats1['time_per_batch']*1000:.2f} ms")
    
    # Stage 2: + Parsing
    print("\n[Stage 2] + Parsing")
    ds2 = tf.data.TFRecordDataset(files).map(
        parser, num_parallel_calls=tf.data.AUTOTUNE
    )
    stats2 = benchmark_pipeline(ds2.batch(batch_size).prefetch(2), num_batches=20)
    print(f"  Throughput: {stats2['batches_per_sec']:.2f} batches/sec")
    print(f"  Time/batch: {stats2['time_per_batch']*1000:.2f} ms")
    parse_overhead = stats2['time_per_batch'] - stats1['time_per_batch']
    print(f"  Parse overhead: {parse_overhead*1000:.2f} ms/batch")
    
    # Stage 3: + Crop generation
    print("\n[Stage 3] + Crop Generation")
    from beagle.dataset.crops import create_overlapping_crops
    crop_fn = partial(create_overlapping_crops, crop_size=crop_size, stride=stride)
    ds3 = tf.data.TFRecordDataset(files).map(
        parser, num_parallel_calls=tf.data.AUTOTUNE
    ).map(
        crop_fn, num_parallel_calls=tf.data.AUTOTUNE
    ).unbatch()
    stats3 = benchmark_pipeline(ds3.batch(batch_size).prefetch(2), num_batches=20)
    print(f"  Throughput: {stats3['batches_per_sec']:.2f} batches/sec")
    print(f"  Time/batch: {stats3['time_per_batch']*1000:.2f} ms")
    crop_overhead = stats3['time_per_batch'] - stats2['time_per_batch']
    print(f"  Crop overhead: {crop_overhead*1000:.2f} ms/batch")
    
    # Stage 4: + Standardization
    print("\n[Stage 4] + Standardization")
    mean, std = 0.5, 0.2  # Dummy values
    def standardize(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        data_dict['image'] = (data_dict['image'] - mean) / (std + 1e-8)
        return data_dict
    
    ds4 = ds3.map(standardize, num_parallel_calls=tf.data.AUTOTUNE)
    stats4 = benchmark_pipeline(ds4.batch(batch_size).prefetch(2), num_batches=20)
    print(f"  Throughput: {stats4['batches_per_sec']:.2f} batches/sec")
    print(f"  Time/batch: {stats4['time_per_batch']*1000:.2f} ms")
    std_overhead = stats4['time_per_batch'] - stats3['time_per_batch']
    print(f"  Standardization overhead: {std_overhead*1000:.2f} ms/batch")
    
    # Stage 5: + Augmentation (tf.py_function)
    print("\n[Stage 5] + Augmentation (tf.py_function)")
    transform = create_transform(MODERATE_AUGMENT)
    aug_fn = lambda img: apply_transform(transform, img)['image']
    
    def augment_wrapper(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        img = data_dict['image']
        img = tf.py_function(
            lambda x: aug_fn(x.numpy()).astype(np.float32),
            [img],
            tf.float32,
        )
        img.set_shape([crop_size, crop_size, 1])
        data_dict['image'] = img
        return data_dict
    
    ds5 = ds4.map(augment_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    stats5 = benchmark_pipeline(ds5.batch(batch_size).prefetch(2), num_batches=20)
    print(f"  Throughput: {stats5['batches_per_sec']:.2f} batches/sec")
    print(f"  Time/batch: {stats5['time_per_batch']*1000:.2f} ms")
    aug_overhead = stats5['time_per_batch'] - stats4['time_per_batch']
    print(f"  Augmentation overhead: {aug_overhead*1000:.2f} ms/batch")
    
    # Stage 6: + Shuffle
    print("\n[Stage 6] + Shuffle + Batch + Prefetch")
    ds6 = ds5.shuffle(buffer_size=2048).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    stats6 = benchmark_pipeline(ds6, num_batches=20)
    print(f"  Throughput: {stats6['batches_per_sec']:.2f} batches/sec")
    print(f"  Time/batch: {stats6['time_per_batch']*1000:.2f} ms")
    
    # Summary
    print("\n" + "=" * 60)
    print("OVERHEAD SUMMARY")
    print("=" * 60)
    print(f"Reading:         {stats1['time_per_batch']*1000:>8.2f} ms/batch")
    print(f"Parsing:         {parse_overhead*1000:>8.2f} ms/batch")
    print(f"Crop generation: {crop_overhead*1000:>8.2f} ms/batch")
    print(f"Standardization: {std_overhead*1000:>8.2f} ms/batch")
    print(f"Augmentation:    {aug_overhead*1000:>8.2f} ms/batch  <-- CHECK THIS")
    print("-" * 60)
    print(f"Total:           {stats6['time_per_batch']*1000:>8.2f} ms/batch")
    print()
    print(f"Final throughput: {stats6['batches_per_sec']:.2f} batches/sec")
    print(f"                  {stats6['batches_per_sec']*batch_size:.2f} samples/sec")
    
    # Identify bottleneck
    overheads = {
        'Reading': stats1['time_per_batch'],
        'Parsing': parse_overhead,
        'Crops': crop_overhead,
        'Standardization': std_overhead,
        'Augmentation': aug_overhead,
    }
    bottleneck = max(overheads, key=overheads.get)
    print(f"\n🔴 BOTTLENECK: {bottleneck}")


def test_cache_impact(
    tfrecord_pattern: str,
    parser,
    crop_size: int = 256,
    stride: int = 192,
    batch_size: int = 32,
) -> None:
    """Test impact of caching at different stages."""
    import glob
    from beagle.dataset.crops import create_overlapping_crops
    
    files = sorted(glob.glob(tfrecord_pattern))
    crop_fn = partial(create_overlapping_crops, crop_size=crop_size, stride=stride)
    
    print("\n" + "=" * 60)
    print("CACHE PLACEMENT ANALYSIS")
    print("=" * 60)
    
    # No cache
    print("\n[Test 1] No cache")
    ds = tf.data.TFRecordDataset(files).map(
        parser, num_parallel_calls=tf.data.AUTOTUNE
    ).map(
        crop_fn, num_parallel_calls=tf.data.AUTOTUNE
    ).unbatch().batch(batch_size).prefetch(2)
    stats = benchmark_pipeline(ds, num_batches=30)
    print(f"  Time/batch: {stats['time_per_batch']*1000:.2f} ms")
    
    # Cache after parse
    print("\n[Test 2] Cache after parsing")
    ds = tf.data.TFRecordDataset(files).map(
        parser, num_parallel_calls=tf.data.AUTOTUNE
    ).cache().map(
        crop_fn, num_parallel_calls=tf.data.AUTOTUNE
    ).unbatch().batch(batch_size).prefetch(2)
    
    # First epoch (fills cache)
    print("  First epoch (filling cache)...")
    iterator = iter(ds)
    for _ in range(20):
        next(iterator)
    
    # Second epoch (uses cache)
    print("  Second epoch (using cache)...")
    stats = benchmark_pipeline(ds, num_batches=30, warmup=0)
    print(f"  Time/batch: {stats['time_per_batch']*1000:.2f} ms")
    
    # Cache after crops
    print("\n[Test 3] Cache after crops (current setup)")
    ds = tf.data.TFRecordDataset(files).map(
        parser, num_parallel_calls=tf.data.AUTOTUNE
    ).map(
        crop_fn, num_parallel_calls=tf.data.AUTOTUNE
    ).unbatch().cache().batch(batch_size).prefetch(2)
    
    # First epoch
    print("  First epoch (filling cache)...")
    iterator = iter(ds)
    for _ in range(20):
        next(iterator)
    
    # Second epoch
    print("  Second epoch (using cache)...")
    stats = benchmark_pipeline(ds, num_batches=30, warmup=0)
    print(f"  Time/batch: {stats['time_per_batch']*1000:.2f} ms")


def main() -> None:
    """Run profiling."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    tfrecord_pattern = str(data_dir / "*.tfrecord")
    json_path = str(data_dir / "polymer.json")
    
    # Load parser
    feature_dict, shape_dict = load_tfr_dict(json_path)
    parser = make_polymer_parser(feature_dict, shape_dict)
    img_height, img_width = shape_dict['surface']
    
    print(f"Data directory: {data_dir}")
    print(f"Image shape: {img_height} x {img_width}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Devices: {tf.config.list_physical_devices()}")
    
    # Profile each stage
    profile_stages(
        tfrecord_pattern,
        parser,
        img_height,
        img_width,
        crop_size=256,
        stride=192,
        batch_size=32,
    )
    
    # Test cache impact
    test_cache_impact(
        tfrecord_pattern,
        parser,
        crop_size=256,
        stride=192,
        batch_size=32,
    )


if __name__ == "__main__":
    main()

