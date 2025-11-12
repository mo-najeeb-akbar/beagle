# Training Module

Functional training utilities for JAX/Flax models.

## Quick Start

```python
import jax
import jax.numpy as jnp
import jax.random as random
from beagle.training import train_loop

# 1. Define train step (must return metrics dict)
@jax.jit
def train_step(state, batch, rng_key):
    def loss_fn(params):
        preds = state.apply_fn({'params': params}, batch['x'])
        return jnp.mean((preds - batch['y']) ** 2)
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, {'loss': loss}

# 2. Create data iterator function
def data_iterator_fn():
    for batch in my_dataloader:
        yield batch

# 3. Run training
final_state, history = train_loop(
    state=initial_state,
    train_step_fn=train_step,
    data_iterator_fn=data_iterator_fn,
    num_epochs=100,
    num_batches=50,
    rng_key=random.PRNGKey(0),
    checkpoint_dir='/checkpoints/my_model',
    checkpoint_every=10,
)

print(f"Final loss: {history['train_loss'][-1]}")
```

## API Reference

### `train_loop`

Main training loop with automatic checkpointing and metrics tracking.

```python
final_state, history = train_loop(
    state: TrainState,
    train_step_fn: Callable,
    data_iterator_fn: Callable,
    num_epochs: int,
    num_batches: int,
    rng_key: jax.Array,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 1,
    val_step_fn: Callable | None = None,
    val_data_iterator: Iterator | None = None,
    val_num_batches: int | None = None,
    show_progress: bool = True,
) -> tuple[TrainState, dict[str, list[float]]]
```

**Signature requirements:**
- `train_step_fn(state, batch, rng_key) -> (new_state, metrics_dict)`
- `data_iterator_fn() -> Iterator[dict]`
- `val_step_fn(state, batch, rng_key) -> (state, metrics_dict)` (optional)

**Returns:**
- `final_state`: Final training state
- `history`: Dict of metric lists, e.g. `{'train_loss': [...], 'val_accuracy': [...]}`

### `TrainState`

Extended Flax training state with optional batch statistics.

```python
from beagle.training import TrainState
import optax

state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.adam(0.001),
    batch_stats=batch_stats,  # Optional, for BatchNorm
)
```

### Checkpointing

```python
from beagle.training import save_checkpoint, load_checkpoint

# Save manually
save_checkpoint(state, '/checkpoints/model', step=100)

# Load
restored_state = load_checkpoint('/checkpoints/model/checkpoint_100', state)
```

### Metrics

```python
from beagle.training import average_metrics, format_metrics, Metrics

# Average across batches
batch_metrics = [Metrics(values={'loss': 1.0}), Metrics(values={'loss': 2.0})]
avg = average_metrics(batch_metrics)

# Pretty print
print(format_metrics(avg))  # "loss: 1.5000"
```

## Common Patterns

### Multiple Metrics

```python
return new_state, {
    'total_loss': total_loss,
    'recon_loss': recon_loss,
    'kl_loss': kl_loss,
    'accuracy': accuracy,
}
```

### Batch Normalization

```python
@jax.jit
def train_step(state, batch, rng_key):
    def loss_fn(params):
        outputs, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch['x'],
            mutable=['batch_stats'],
            train=True,
        )
        return loss, updates
    
    (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    new_state = new_state.replace(batch_stats=updates['batch_stats'])
    return new_state, {'loss': loss}
```

### Validation

```python
@jax.jit
def val_step(state, batch, rng_key):
    preds = state.apply_fn({'params': state.params}, batch['x'])
    accuracy = jnp.mean(preds.argmax(-1) == batch['y'])
    return state, {'accuracy': accuracy}

def val_data_iterator():
    for batch in val_dataloader:
        yield batch

final_state, history = train_loop(
    ...,
    val_step_fn=val_step,
    val_data_iterator=val_data_iterator(),
    val_num_batches=10,
)

# Access validation metrics
print(history['train_loss'])      # Training loss per epoch
print(history['val_accuracy'])    # Validation accuracy per epoch
```

## Design Philosophy

- **Functional**: Pure functions, immutable configs, side effects isolated to I/O
- **Composable**: Build complex training from simple functions
- **Type-safe**: Full type hints throughout
- **Minimal**: ~400 lines of code, no magic

## Examples

See `examples/training_example.py` for a complete working example.

## Testing

```bash
make run CMD='pytest tests/test_training.py -v'
```
