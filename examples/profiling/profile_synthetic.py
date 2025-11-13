"""
Profile dataloader performance with synthetic data.

This creates synthetic TFRecords to benchmark the pipeline without real data.

Usage:
    python examples/profile_synthetic.py
"""
from __future__ import annotations

import time
import tempfile
import os
from pathlib import Path
from functools import partial

import tensorflow as tf
import numpy as np

from beagle.augmentations import create_transform, MODERATE_AUGMENT, apply_transform
from beagle.dataset.crops import create_overlapping_crops


def create_synthetic_tfrecords(
    output_dir: str,
    num_files: int = 3,
    samples_per_file: int = 10,
    img_height: int = 512,
    img_width: int = 512,
) -> None:
    """Create synthetic TFRecords for testing."""
    for file_idx in range(num_files):
        output_path = os.path.join(output_dir, f"synthetic_{file_idx:03d}.tfrecord")
        
        with tf.io.TFRecordWriter(output_path) as writer:
            for _ in range(samples_per_file):
                # Generate random data
                image = np.random.randn(img_height, img_width, 1).astype(np.float32)
                
                # Create feature
                feature = {
                    'surface': tf.train.Feature(
                        float_list=tf.train.FloatList(value=image.flatten())
                    ),
                }
                
                # Create example
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature)
                )
                
                writer.write(example.SerializeToString())


def make_synthetic_parser(img_height: int, img_width: int):
    """Create parser for synthetic TFRecords."""
    def parse(example_proto: tf.Tensor) -> dict[str, tf.Tensor]:
        feature_dict = {
            'surface': tf.io.FixedLenFeature([img_height * img_width], tf.float32)
        }
        parsed = tf.io.parse_single_example(example_proto, feature_dict)
        surface = tf.reshape(parsed['surface'], [img_height, img_width, 1])
        return {'image': surface}
    
    return parse


def benchmark_pipeline(
    dataset: tf.data.Dataset,
    num_batches: int = 50,
    warmup: int = 5,
    is_raw: bool = False,
) -> dict[str, float]:
    """Benchmark pipeline throughput."""
    # Warmup
    iterator = iter(dataset)
    for _ in range(warmup):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(dataset)
            next(iterator)
    
    # Time
    start = time.perf_counter()
    for i in range(num_batches):
        try:
            batch = next(iterator)
            # Force computation
            if is_raw:
                # Raw TFRecord, just access the tensor
                if isinstance(batch, dict):
                    _ = batch['image'].numpy()
                else:
                    _ = batch.numpy()
            else:
                _ = batch['image'].numpy()
        except StopIteration:
            iterator = iter(dataset)
            batch = next(iterator)
            if is_raw:
                if isinstance(batch, dict):
                    _ = batch['image'].numpy()
                else:
                    _ = batch.numpy()
            else:
                _ = batch['image'].numpy()
    
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
    
    print("=" * 70)
    print("PROFILING DATALOADER PIPELINE")
    print("=" * 70)
    print(f"Files: {len(files)}")
    print(f"Batch size: {batch_size}")
    print(f"Crop size: {crop_size}, stride: {stride}")
    
    # Stage 1: Raw TFRecord reading
    print("\n[Stage 1] Raw TFRecord Reading")
    ds1 = tf.data.TFRecordDataset(files).repeat()
    stats1 = benchmark_pipeline(ds1.batch(batch_size).prefetch(2), num_batches=20, is_raw=True)
    print(f"  Throughput: {stats1['batches_per_sec']:.2f} batches/sec")
    print(f"  Time/batch: {stats1['time_per_batch']*1000:.2f} ms")
    
    # Stage 2: + Parsing
    print("\n[Stage 2] + Parsing")
    ds2 = tf.data.TFRecordDataset(files).map(
        parser, num_parallel_calls=tf.data.AUTOTUNE
    ).repeat()
    stats2 = benchmark_pipeline(ds2.batch(batch_size).prefetch(2), num_batches=20)
    print(f"  Throughput: {stats2['batches_per_sec']:.2f} batches/sec")
    print(f"  Time/batch: {stats2['time_per_batch']*1000:.2f} ms")
    parse_overhead = stats2['time_per_batch'] - stats1['time_per_batch']
    print(f"  Parse overhead: {parse_overhead*1000:.2f} ms/batch")
    
    # Stage 3: + Crop generation
    print("\n[Stage 3] + Crop Generation (extract_patches)")
    crop_fn = partial(create_overlapping_crops, crop_size=crop_size, stride=stride)
    ds3 = tf.data.TFRecordDataset(files).map(
        parser, num_parallel_calls=tf.data.AUTOTUNE
    ).map(
        crop_fn, num_parallel_calls=tf.data.AUTOTUNE
    ).unbatch().repeat()
    stats3 = benchmark_pipeline(ds3.batch(batch_size).prefetch(2), num_batches=20)
    print(f"  Throughput: {stats3['batches_per_sec']:.2f} batches/sec")
    print(f"  Time/batch: {stats3['time_per_batch']*1000:.2f} ms")
    crop_overhead = stats3['time_per_batch'] - stats2['time_per_batch']
    print(f"  Crop overhead: {crop_overhead*1000:.2f} ms/batch")
    
    # Stage 4: + Cache
    print("\n[Stage 4] + Cache")
    ds4 = tf.data.TFRecordDataset(files).map(
        parser, num_parallel_calls=tf.data.AUTOTUNE
    ).map(
        crop_fn, num_parallel_calls=tf.data.AUTOTUNE
    ).unbatch().cache().repeat()
    
    # Fill cache first
    print("  Filling cache...")
    iterator = iter(ds4.batch(batch_size).prefetch(2))
    for _ in range(50):
        next(iterator)
    
    # Now benchmark
    stats4 = benchmark_pipeline(ds4.batch(batch_size).prefetch(2), num_batches=20, warmup=0)
    print(f"  Throughput: {stats4['batches_per_sec']:.2f} batches/sec")
    print(f"  Time/batch: {stats4['time_per_batch']*1000:.2f} ms")
    cache_benefit = stats3['time_per_batch'] - stats4['time_per_batch']
    print(f"  Cache benefit: {cache_benefit*1000:.2f} ms/batch saved")
    
    # Stage 5: + Standardization
    print("\n[Stage 5] + Standardization")
    mean, std = 0.5, 0.2
    def standardize(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        data_dict['image'] = (data_dict['image'] - mean) / (std + 1e-8)
        return data_dict
    
    ds5 = ds4.map(standardize, num_parallel_calls=tf.data.AUTOTUNE)
    stats5 = benchmark_pipeline(ds5.batch(batch_size).prefetch(2), num_batches=20)
    print(f"  Throughput: {stats5['batches_per_sec']:.2f} batches/sec")
    print(f"  Time/batch: {stats5['time_per_batch']*1000:.2f} ms")
    std_overhead = stats5['time_per_batch'] - stats4['time_per_batch']
    print(f"  Standardization overhead: {std_overhead*1000:.2f} ms/batch")
    
    # Stage 6: + Augmentation (tf.py_function) - THE KILLER (OLD BAD WAY)
    print("\n[Stage 6] + Augmentation (tf.py_function) ⚠️ - OLD BAD WAY")
    transform = create_transform(MODERATE_AUGMENT)
    numpy_aug_fn = lambda img: apply_transform(transform, img)['image']
    
    def augment_wrapper_slow(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """Old slow way: numpy augmentation via tf.py_function"""
        img = data_dict['image']
        img = tf.py_function(
            lambda x: numpy_aug_fn(x.numpy()).astype(np.float32),
            [img],
            tf.float32,
        )
        img.set_shape([crop_size, crop_size, 1])
        data_dict['image'] = img
        return data_dict
    
    ds6 = ds5.map(augment_wrapper_slow, num_parallel_calls=tf.data.AUTOTUNE)
    stats6 = benchmark_pipeline(ds6.batch(batch_size).prefetch(2), num_batches=20)
    print(f"  Throughput: {stats6['batches_per_sec']:.2f} batches/sec")
    print(f"  Time/batch: {stats6['time_per_batch']*1000:.2f} ms")
    aug_overhead = stats6['time_per_batch'] - stats5['time_per_batch']
    print(f"  Augmentation overhead: {aug_overhead*1000:.2f} ms/batch  🔴 SLOW!")
    
    # Stage 7: + Shuffle + Batch + Prefetch (OLD SLOW WAY)
    print("\n[Stage 7] + Shuffle + Batch + Prefetch (OLD SLOW PIPELINE)")
    print("  Note: This shows the NAIVE approach with tf.py_function")
    ds7 = ds6.shuffle(buffer_size=2048).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    stats7 = benchmark_pipeline(ds7, num_batches=50)
    print(f"  Throughput: {stats7['batches_per_sec']:.2f} batches/sec")
    print(f"  Time/batch: {stats7['time_per_batch']*1000:.2f} ms")
    
    # Stage 8: Optimized pipeline (native TensorFlow augmentation)
    print("\n[Stage 8] OPTIMIZED: Native TensorFlow augmentation ✨")
    from beagle.dataset import create_iterator
    
    # Create TensorFlow augmentation function (FAST!)
    def tf_augment_fn(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """Fast TensorFlow-native augmentation"""
        img = data_dict['image']
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, k=k)
        img = tf.image.random_brightness(img, 0.2)
        data_dict['image'] = img
        return data_dict
    
    # Use create_iterator with TensorFlow augmentation
    files = sorted(glob.glob(tfrecord_pattern))
    iterator_opt, _ = create_iterator(
        tfrecord_pattern=tfrecord_pattern,
        parser=parser,
        crop_size=crop_size,
        stride=stride,
        image_shape=(img_height, img_width),
        batch_size=batch_size,
        augment_fn=tf_augment_fn,
        precomputed_stats=(0.0, 1.0),
        shuffle=True,
    )
    
    # Benchmark optimized version
    def benchmark_jax_iterator(iterator, num_batches=50, warmup=10):
        # Warmup
        for _ in range(warmup):
            next(iterator)
        
        # Time
        start = time.perf_counter()
        for _ in range(num_batches):
            batch = next(iterator)
            _ = np.array(batch['image'])
        elapsed = time.perf_counter() - start
        return elapsed / num_batches
    
    time_per_batch_opt = benchmark_jax_iterator(iterator_opt)
    throughput_opt = 1.0 / time_per_batch_opt
    
    print(f"  Throughput: {throughput_opt:.2f} batches/sec")
    print(f"  Time/batch: {time_per_batch_opt*1000:.2f} ms")
    speedup = stats7['time_per_batch'] / time_per_batch_opt
    print(f"  Speedup: {speedup:.2f}x faster than naive tf.py_function!")
    
    # Summary
    print("\n" + "=" * 70)
    print("OVERHEAD BREAKDOWN (Naive tf.py_function approach)")
    print("=" * 70)
    print(f"Reading:              {stats1['time_per_batch']*1000:>8.2f} ms/batch")
    print(f"Parsing:              {parse_overhead*1000:>8.2f} ms/batch")
    print(f"Crop generation:      {crop_overhead*1000:>8.2f} ms/batch")
    print(f"Cache (benefit):     {-cache_benefit*1000:>8.2f} ms/batch")
    print(f"Standardization:      {std_overhead*1000:>8.2f} ms/batch")
    print(f"Augmentation:         {aug_overhead*1000:>8.2f} ms/batch  🔴 LIKELY BOTTLENECK")
    print("-" * 70)
    print(f"Total pipeline:       {stats7['time_per_batch']*1000:>8.2f} ms/batch")
    print()
    print(f"Final throughput:     {stats7['batches_per_sec']:.2f} batches/sec")
    print(f"                      {stats7['batches_per_sec']*batch_size:.2f} samples/sec")
    
    # Calculate percentages
    total = stats7['time_per_batch']
    print("\n" + "=" * 70)
    print("PERCENTAGE OF TIME")
    print("=" * 70)
    if aug_overhead > 0:
        aug_pct = (aug_overhead / total) * 100
        print(f"Augmentation: {aug_pct:.1f}% of total time")
        if aug_pct > 50:
            print("  🔴 CRITICAL: Augmentation is the primary bottleneck!")
        elif aug_pct > 30:
            print("  ⚠️  WARNING: Augmentation is a major bottleneck")
    
    if crop_overhead > 0:
        crop_pct = (crop_overhead / total) * 100
        print(f"Crop generation: {crop_pct:.1f}% of total time")
    
    # Show optimized version
    print("\n" + "=" * 70)
    print("OPTIMIZED PIPELINE (Current create_iterator)")
    print("=" * 70)
    print(f"Time/batch: {time_per_batch_opt*1000:.2f} ms")
    print(f"Throughput: {throughput_opt:.2f} batches/sec ({throughput_opt*batch_size:.1f} samples/sec)")
    print(f"Speedup: {speedup:.2f}x faster!")
    print()
    print("✅ Augmentation runs natively in TensorFlow (not tf.py_function)")
    print("✅ No Python/TF boundary crossing overhead")
    print("✅ Fully parallelized with num_parallel_calls=AUTOTUNE")
    print("✅ Same API - just use create_iterator()")


def test_augmentation_detail(crop_size: int = 256) -> None:
    """Detailed profiling of augmentation operations."""
    print("\n" + "=" * 70)
    print("DETAILED AUGMENTATION PROFILING")
    print("=" * 70)
    
    # Old numpy-based augmentation
    transform = create_transform(MODERATE_AUGMENT)
    numpy_aug_fn = lambda img: apply_transform(transform, img)['image']
    
    # Generate test image
    test_img = np.random.randn(crop_size, crop_size, 1).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = numpy_aug_fn(test_img)
    
    # Time individual augmentations (numpy-based)
    num_iters = 100
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = numpy_aug_fn(test_img)
    elapsed_numpy = time.perf_counter() - start
    
    print("NUMPY-BASED AUGMENTATION (OLD WAY):")
    print(f"  Single augmentation: {elapsed_numpy/num_iters*1000:.2f} ms")
    print(f"  Per batch (32 samples): {elapsed_numpy/num_iters*1000*32:.2f} ms")
    print()
    print("⚠️  Note: tf.py_function adds even MORE overhead on top of this!")
    print("⚠️  Each py_function call crosses Python/TF boundary (~expensive)")
    print()
    print("RECOMMENDATION: Use native TensorFlow augmentations instead!")
    print("  - tf.image.random_flip_left_right()")
    print("  - tf.image.random_brightness()")
    print("  - tf.image.rot90() for 90-degree rotations")
    print("  - These run in the TF graph with full parallelization")


def main() -> None:
    """Run profiling."""
    print("Creating synthetic TFRecords...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic data
        img_height, img_width = 512, 512
        create_synthetic_tfrecords(
            tmpdir,
            num_files=3,
            samples_per_file=20,
            img_height=img_height,
            img_width=img_width,
        )
        
        tfrecord_pattern = os.path.join(tmpdir, "*.tfrecord")
        parser = make_synthetic_parser(img_height, img_width)
        
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Devices: {tf.config.list_physical_devices()}")
        print()
        
        # Profile pipeline
        profile_stages(
            tfrecord_pattern,
            parser,
            img_height=img_height,
            img_width=img_width,
            crop_size=256,
            stride=192,
            batch_size=32,
        )
        
        # Detailed augmentation profiling
        test_augmentation_detail(crop_size=256)


if __name__ == "__main__":
    main()

