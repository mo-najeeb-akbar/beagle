"""Tests for dataset splitting utilities."""

from __future__ import annotations

import os
import tempfile
from typing import Iterator

import pytest
import tensorflow as tf
import jax.numpy as jnp

from beagle.dataset.splitting import create_train_val_iterators


def create_test_parser():
    """Create a simple parser for test TFRecords."""
    def parser(serialized: tf.Tensor) -> dict[str, tf.Tensor]:
        features = {
            'value': tf.io.FixedLenFeature([1], tf.float32),
            'index': tf.io.FixedLenFeature([1], tf.float32),
        }
        parsed = tf.io.parse_single_example(serialized, features)
        return {
            'value': tf.reshape(parsed['value'], []),
            'index': tf.reshape(parsed['index'], []),
        }
    return parser


def create_test_tfrecord(
    path: str,
    n_samples: int,
    start_index: int = 0
) -> None:
    """Create a test TFRecord file with simple data."""
    with tf.io.TFRecordWriter(path) as writer:
        for i in range(n_samples):
            features = {
                'value': tf.train.Feature(
                    float_list=tf.train.FloatList(value=[float(i + start_index)])
                ),
                'index': tf.train.Feature(
                    float_list=tf.train.FloatList(value=[float(i + start_index)])
                ),
            }
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())


class TestCreateTrainValIteratorsInputValidation:
    """Test input validation for create_train_val_iterators."""

    def test_invalid_val_fraction_negative(self):
        """Test that negative val_fraction raises ValueError."""
        parser = create_test_parser()
        with pytest.raises(ValueError, match="val_fraction must be between 0.0 and 1.0"):
            create_train_val_iterators(
                files=['dummy.tfrecord'],
                parser=parser,
                batch_size=2,
                val_fraction=-0.1
            )

    def test_invalid_val_fraction_too_large(self):
        """Test that val_fraction > 1.0 raises ValueError."""
        parser = create_test_parser()
        with pytest.raises(ValueError, match="val_fraction must be between 0.0 and 1.0"):
            create_train_val_iterators(
                files=['dummy.tfrecord'],
                parser=parser,
                batch_size=2,
                val_fraction=1.5
            )

    def test_invalid_batch_size(self):
        """Test that batch_size < 1 raises ValueError."""
        parser = create_test_parser()
        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            create_train_val_iterators(
                files=['dummy.tfrecord'],
                parser=parser,
                batch_size=0,
                val_fraction=0.2
            )

    def test_empty_files_list(self):
        """Test that empty files list raises ValueError."""
        parser = create_test_parser()
        with pytest.raises(ValueError, match="files list cannot be empty"):
            create_train_val_iterators(
                files=[],
                parser=parser,
                batch_size=2,
                val_fraction=0.2
            )


class TestCreateTrainValIteratorsBasicFunctionality:
    """Test basic functionality of create_train_val_iterators."""

    def test_basic_split(self):
        """Test basic train/val split with single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data: 100 samples
            tfrecord_path = os.path.join(tmpdir, 'test.tfrecord')
            create_test_tfrecord(tfrecord_path, n_samples=100)

            parser = create_test_parser()
            train_iter, val_iter, n_train, n_val = create_train_val_iterators(
                files=[tfrecord_path],
                parser=parser,
                batch_size=10,
                val_fraction=0.2,
                shuffle=False,
                repeat=False
            )

            # Check batch counts: 100 samples, 20% val = 20 val, 80 train
            # With batch_size=10: 2 val batches, 8 train batches
            assert n_train == 8
            assert n_val == 2

            # Check that iterators work
            train_batch = next(train_iter)
            assert 'value' in train_batch
            assert 'index' in train_batch
            assert train_batch['value'].shape == (10,)

            val_batch = next(val_iter)
            assert val_batch['value'].shape == (10,)

    def test_sample_level_split_without_shuffle(self):
        """Test that split happens at sample level, not file level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data: 50 samples
            tfrecord_path = os.path.join(tmpdir, 'test.tfrecord')
            create_test_tfrecord(tfrecord_path, n_samples=50)

            parser = create_test_parser()
            train_iter, val_iter, n_train, n_val = create_train_val_iterators(
                files=[tfrecord_path],
                parser=parser,
                batch_size=5,
                val_fraction=0.2,
                shuffle=False,
                repeat=False
            )

            # 50 samples * 0.2 = 10 val, 40 train
            # With batch_size=5: 2 val batches, 8 train batches
            assert n_val == 2
            assert n_train == 8

            # Collect validation data (should be first 10 samples: 0-9)
            val_batch1 = next(val_iter)
            val_batch2 = next(val_iter)
            val_indices = jnp.concatenate([val_batch1['index'], val_batch2['index']])
            expected_val = jnp.arange(10, dtype=jnp.float32)
            assert jnp.allclose(val_indices, expected_val)

            # Collect first train batch (should start at sample 10)
            train_batch1 = next(train_iter)
            train_indices = train_batch1['index']
            expected_train_start = jnp.arange(10, 15, dtype=jnp.float32)
            assert jnp.allclose(train_indices, expected_train_start)

    def test_multiple_files(self):
        """Test splitting with multiple TFRecord files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files with 30 samples each
            files = []
            for i in range(3):
                path = os.path.join(tmpdir, f'test_{i}.tfrecord')
                create_test_tfrecord(path, n_samples=30, start_index=i * 30)
                files.append(path)

            parser = create_test_parser()
            train_iter, val_iter, n_train, n_val = create_train_val_iterators(
                files=files,
                parser=parser,
                batch_size=10,
                val_fraction=0.2,
                shuffle=False,
                repeat=False
            )

            # Total: 90 samples, 20% val = 18 val, 72 train
            # With batch_size=10: 1 val batch, 7 train batches
            assert n_val == 1
            assert n_train == 7

            # Verify data exists
            val_batch = next(val_iter)
            assert val_batch['value'].shape == (10,)

            train_batch = next(train_iter)
            assert train_batch['value'].shape == (10,)


class TestCreateTrainValIteratorsShuffle:
    """Test shuffling behavior."""

    def test_shuffle_deterministic_with_seed(self):
        """Test that shuffling is deterministic with same seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tfrecord_path = os.path.join(tmpdir, 'test.tfrecord')
            create_test_tfrecord(tfrecord_path, n_samples=50)

            parser = create_test_parser()

            # Create two iterators with same seed
            _, val_iter1, _, _ = create_train_val_iterators(
                files=[tfrecord_path],
                parser=parser,
                batch_size=5,
                val_fraction=0.2,
                shuffle=True,
                seed=42,
                repeat=False
            )

            _, val_iter2, _, _ = create_train_val_iterators(
                files=[tfrecord_path],
                parser=parser,
                batch_size=5,
                val_fraction=0.2,
                shuffle=True,
                seed=42,
                repeat=False
            )

            # Collect validation data from both
            val1_batch1 = next(val_iter1)
            val1_batch2 = next(val_iter1)

            val2_batch1 = next(val_iter2)
            val2_batch2 = next(val_iter2)

            # Should be identical with same seed
            assert jnp.allclose(val1_batch1['index'], val2_batch1['index'])
            assert jnp.allclose(val1_batch2['index'], val2_batch2['index'])

    def test_shuffle_different_seeds_produce_different_splits(self):
        """Test that different seeds produce different splits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tfrecord_path = os.path.join(tmpdir, 'test.tfrecord')
            create_test_tfrecord(tfrecord_path, n_samples=100)

            parser = create_test_parser()

            # Create iterators with different seeds
            _, val_iter1, _, _ = create_train_val_iterators(
                files=[tfrecord_path],
                parser=parser,
                batch_size=10,
                val_fraction=0.2,
                shuffle=True,
                seed=42,
                repeat=False
            )

            _, val_iter2, _, _ = create_train_val_iterators(
                files=[tfrecord_path],
                parser=parser,
                batch_size=10,
                val_fraction=0.2,
                shuffle=True,
                seed=123,
                repeat=False
            )

            val1_batch = next(val_iter1)
            val2_batch = next(val_iter2)

            # Different seeds should produce different orderings
            assert not jnp.allclose(val1_batch['index'], val2_batch['index'])


class TestCreateTrainValIteratorsRepeat:
    """Test repeat behavior."""

    def test_repeat_true_allows_multiple_epochs(self):
        """Test that repeat=True allows iterating beyond one epoch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tfrecord_path = os.path.join(tmpdir, 'test.tfrecord')
            create_test_tfrecord(tfrecord_path, n_samples=20)

            parser = create_test_parser()
            train_iter, _, n_train, _ = create_train_val_iterators(
                files=[tfrecord_path],
                parser=parser,
                batch_size=10,
                val_fraction=0.2,
                shuffle=False,
                repeat=True
            )

            # n_train = 1 batch (16 train samples / 10 batch_size = 1)
            # With repeat=True, should be able to iterate more than n_train times
            for _ in range(n_train * 3):
                batch = next(train_iter)
                assert batch['value'].shape == (10,)

    def test_repeat_false_limits_iteration(self):
        """Test that repeat=False stops after one epoch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tfrecord_path = os.path.join(tmpdir, 'test.tfrecord')
            create_test_tfrecord(tfrecord_path, n_samples=30)

            parser = create_test_parser()
            _, val_iter, _, n_val = create_train_val_iterators(
                files=[tfrecord_path],
                parser=parser,
                batch_size=5,
                val_fraction=0.2,
                shuffle=False,
                repeat=False
            )

            # Consume exactly n_val batches
            batches_consumed = 0
            for _ in range(n_val):
                next(val_iter)
                batches_consumed += 1

            # Next call should raise StopIteration
            with pytest.raises(StopIteration):
                next(val_iter)


class TestCreateTrainValIteratorsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_val_fraction(self):
        """Test with val_fraction=0.0 (all data goes to train)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tfrecord_path = os.path.join(tmpdir, 'test.tfrecord')
            create_test_tfrecord(tfrecord_path, n_samples=50)

            parser = create_test_parser()
            train_iter, val_iter, n_train, n_val = create_train_val_iterators(
                files=[tfrecord_path],
                parser=parser,
                batch_size=10,
                val_fraction=0.0,
                shuffle=False,
                repeat=False
            )

            # All samples go to training
            assert n_train == 5
            assert n_val == 0

            # Train iterator should work
            train_batch = next(train_iter)
            assert train_batch['value'].shape == (10,)

            # Val iterator should immediately stop
            with pytest.raises(StopIteration):
                next(val_iter)

    def test_one_val_fraction(self):
        """Test with val_fraction=1.0 (all data goes to val)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tfrecord_path = os.path.join(tmpdir, 'test.tfrecord')
            create_test_tfrecord(tfrecord_path, n_samples=50)

            parser = create_test_parser()
            train_iter, val_iter, n_train, n_val = create_train_val_iterators(
                files=[tfrecord_path],
                parser=parser,
                batch_size=10,
                val_fraction=1.0,
                shuffle=False,
                repeat=False
            )

            # All samples go to validation
            assert n_train == 0
            assert n_val == 5

            # Val iterator should work
            val_batch = next(val_iter)
            assert val_batch['value'].shape == (10,)

            # Train iterator should immediately stop
            with pytest.raises(StopIteration):
                next(train_iter)

    def test_small_dataset_with_large_batch(self):
        """Test behavior when dataset is smaller than batch size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tfrecord_path = os.path.join(tmpdir, 'test.tfrecord')
            create_test_tfrecord(tfrecord_path, n_samples=5)

            parser = create_test_parser()
            train_iter, val_iter, n_train, n_val = create_train_val_iterators(
                files=[tfrecord_path],
                parser=parser,
                batch_size=10,
                val_fraction=0.2,
                shuffle=False,
                repeat=False
            )

            # 5 samples, 20% = 1 val, 4 train
            # With batch_size=10, drop_remainder=True means 0 batches
            assert n_train == 0
            assert n_val == 0

    def test_batch_size_equals_dataset_size(self):
        """Test when batch size equals total dataset size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tfrecord_path = os.path.join(tmpdir, 'test.tfrecord')
            create_test_tfrecord(tfrecord_path, n_samples=20)

            parser = create_test_parser()
            train_iter, val_iter, n_train, n_val = create_train_val_iterators(
                files=[tfrecord_path],
                parser=parser,
                batch_size=20,
                val_fraction=0.2,
                shuffle=False,
                repeat=False
            )

            # 20 samples, 20% = 4 val, 16 train
            # batch_size=20: 0 val batches, 0 train batches (drop_remainder)
            assert n_train == 0
            assert n_val == 0


class TestCreateTrainValIteratorsJAXIntegration:
    """Test JAX integration and array types."""

    def test_returns_jax_arrays(self):
        """Test that iterators return JAX arrays, not TensorFlow tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tfrecord_path = os.path.join(tmpdir, 'test.tfrecord')
            create_test_tfrecord(tfrecord_path, n_samples=20)

            parser = create_test_parser()
            train_iter, _, _, _ = create_train_val_iterators(
                files=[tfrecord_path],
                parser=parser,
                batch_size=5,
                val_fraction=0.2,
                shuffle=False,
                repeat=False
            )

            batch = next(train_iter)

            # Check that returned values are JAX arrays
            assert isinstance(batch['value'], jnp.ndarray)
            assert isinstance(batch['index'], jnp.ndarray)

            # Verify they're float32 (default dtype)
            assert batch['value'].dtype == jnp.float32
            assert batch['index'].dtype == jnp.float32

    def test_jax_operations_on_batches(self):
        """Test that JAX operations work on returned batches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tfrecord_path = os.path.join(tmpdir, 'test.tfrecord')
            create_test_tfrecord(tfrecord_path, n_samples=30)

            parser = create_test_parser()
            train_iter, _, _, _ = create_train_val_iterators(
                files=[tfrecord_path],
                parser=parser,
                batch_size=10,
                val_fraction=0.2,
                shuffle=False,
                repeat=False
            )

            batch = next(train_iter)

            # Perform JAX operations
            result = batch['value'] * 2.0 + 1.0
            mean_val = jnp.mean(result)

            assert isinstance(result, jnp.ndarray)
            assert isinstance(mean_val, jnp.ndarray)


class TestCreateTrainValIteratorsWithCropping:
    """Test cropping functionality in create_train_val_iterators."""

    def test_cropping_validation_requires_stride(self):
        """Test that crop_size without stride raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tfrecord_path = os.path.join(tmpdir, 'test.tfrecord')
            create_test_tfrecord(tfrecord_path, n_samples=10)

            parser = create_test_parser()
            with pytest.raises(ValueError, match="stride must be provided when crop_size is set"):
                create_train_val_iterators(
                    files=[tfrecord_path],
                    parser=parser,
                    batch_size=2,
                    val_fraction=0.2,
                    crop_size=128
                )

    def test_cropping_validation_requires_image_shape(self):
        """Test that crop_size without image_shape raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tfrecord_path = os.path.join(tmpdir, 'test.tfrecord')
            create_test_tfrecord(tfrecord_path, n_samples=10)

            parser = create_test_parser()
            with pytest.raises(ValueError, match="image_shape must be provided when crop_size is set"):
                create_train_val_iterators(
                    files=[tfrecord_path],
                    parser=parser,
                    batch_size=2,
                    val_fraction=0.2,
                    crop_size=128,
                    stride=64
                )


class TestCreateTrainValIteratorsSampleCounts:
    """Test that sample counts are accurate."""

    def test_sample_counts_are_accurate(self):
        """Test that n_train_batches and n_val_batches match actual data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tfrecord_path = os.path.join(tmpdir, 'test.tfrecord')
            create_test_tfrecord(tfrecord_path, n_samples=100)

            parser = create_test_parser()
            train_iter, val_iter, n_train, n_val = create_train_val_iterators(
                files=[tfrecord_path],
                parser=parser,
                batch_size=8,
                val_fraction=0.3,
                shuffle=False,
                repeat=False
            )

            # Count actual batches
            actual_train_batches = sum(1 for _ in train_iter)
            actual_val_batches = sum(1 for _ in val_iter)

            assert actual_train_batches == n_train
            assert actual_val_batches == n_val

    def test_no_data_leakage_between_splits(self):
        """Verify that train and val don't share samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tfrecord_path = os.path.join(tmpdir, 'test.tfrecord')
            create_test_tfrecord(tfrecord_path, n_samples=40)

            parser = create_test_parser()
            train_iter, val_iter, _, _ = create_train_val_iterators(
                files=[tfrecord_path],
                parser=parser,
                batch_size=5,
                val_fraction=0.25,
                shuffle=False,
                repeat=False
            )

            # Collect all val indices
            val_indices = []
            for batch in val_iter:
                val_indices.extend(batch['index'].tolist())

            # Collect all train indices
            train_indices = []
            for batch in train_iter:
                train_indices.extend(batch['index'].tolist())

            # Convert to sets and check no overlap
            val_set = set(val_indices)
            train_set = set(train_indices)

            assert len(val_set & train_set) == 0, "Train and val sets should not overlap"
            # 40 samples * 0.25 = 10 val, 30 train
            # With batch_size=5 and drop_remainder=True: 2 val batches (10 samples), 6 train batches (30 samples)
            assert len(val_set) == 10, f"Expected 10 val samples, got {len(val_set)}"
            assert len(train_set) == 30, f"Expected 30 train samples, got {len(train_set)}"
