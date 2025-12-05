"""Training and validation step functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from beagle.wavelets import wavedec2
from beagle.training import TrainState


def create_train_step(wavelet_weights: tuple[float, ...]):
    """Create JIT-compiled training step."""
    @jax.jit
    def train_step(state: TrainState, batch: dict, rng_key):
        images = batch['depth']
        wavelets = wavedec2(images, wavelet="haar")

        def loss_fn(params):
            x_recon, x_wave, mu, log_var = state.apply_fn(
                {'params': params}, images, training=True, key=rng_key
            )
            recon_loss = jnp.mean(jnp.square(images - x_recon))
            ll_loss = jnp.mean(jnp.square(wavelets[..., 0] - x_wave[..., 0]))
            hl_loss = jnp.mean(jnp.square(wavelets[..., 1] - x_wave[..., 1]))
            lh_loss = jnp.mean(jnp.square(wavelets[..., 2] - x_wave[..., 2]))
            hh_loss = jnp.mean(jnp.square(wavelets[..., 3] - x_wave[..., 3]))

            w = wavelet_weights
            wave_loss = w[0] * ll_loss + w[1] * hl_loss + w[2] * lh_loss + w[3] * hh_loss
            return wave_loss, recon_loss

        (loss, recon_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, {'loss': loss, 'recon_loss': recon_loss}

    return train_step


def create_val_step(wavelet_weights: tuple[float, ...]):
    """Create JIT-compiled validation step."""
    @jax.jit
    def val_step(state: TrainState, batch: dict, rng_key):
        images = batch['depth']
        wavelets = wavedec2(images, wavelet="haar")

        x_recon, x_wave, _, _ = state.apply_fn(
            {'params': state.params}, images, training=False, key=rng_key
        )
        recon_loss = jnp.mean(jnp.square(images - x_recon))
        ll_loss = jnp.mean(jnp.square(wavelets[..., 0] - x_wave[..., 0]))

        return {'val_loss': recon_loss, 'val_ll_loss': ll_loss}

    return val_step


def run_epoch(
    state: TrainState,
    train_iter,
    val_iter,
    train_step_fn,
    val_step_fn,
    n_train: int,
    n_val: int,
    rng_key
) -> tuple[TrainState, dict, any]:
    """Run one epoch of training and validation."""
    # Train
    train_losses = []
    for i in range(n_train):
        batch = next(train_iter)
        rng_key, step_key = jax.random.split(rng_key)
        state, metrics = train_step_fn(state, batch, step_key)
        train_losses.append(float(metrics['loss']))

    # Validate
    val_metrics_list = []
    for i in range(n_val):
        batch = next(val_iter)
        rng_key, step_key = jax.random.split(rng_key)
        val_metrics = val_step_fn(state, batch, step_key)
        val_metrics_list.append(float(val_metrics['val_loss']))

    # Aggregate
    epoch_metrics = {
        'train_loss': sum(train_losses) / len(train_losses),
        'val_loss': sum(val_metrics_list) / len(val_metrics_list),
    }

    return state, epoch_metrics, rng_key
