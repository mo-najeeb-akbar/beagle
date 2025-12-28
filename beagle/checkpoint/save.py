"""Checkpoint saving utilities with metadata.

Per-node checkpointing allows:
- Independent loading of graph components
- Partial graph initialization
- Robust handling of architecture changes
"""

import json
from pathlib import Path
from typing import Any
import jax
import orbax.checkpoint as ocp


def save_graph_state(state, path: str):
    """Save entire GraphState with metadata.

    Creates directory structure:
        path/
          ├─ graph_config.json    # Graph structure
          ├─ optimizer.msgpack    # Optimizer state
          ├─ node_encoder/
          │   ├─ metadata.json
          │   └─ variables.msgpack
          └─ node_decoder/
              ├─ metadata.json
              └─ variables.msgpack

    Args:
        state: GraphState to save
        path: Directory path to save checkpoint
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

    # Save graph configuration
    graph_config = {
        'nodes': {
            name: {
                'module_class': node.module.__class__.__name__,
                'module_module': node.module.__class__.__module__,
                'inputs': node.inputs,
                'outputs': node.outputs,
                'trainable': node.trainable,
                'mutable_collections': node.mutable_collections,
            }
            for name, node in state.graph.nodes.items()
        },
        'edges': state.graph.edges,
        'execution_order': state.graph.execution_order,
        'step': int(state.step),
        'version': '1.0',
    }

    with open(path / 'graph_config.json', 'w') as f:
        json.dump(graph_config, f, indent=2)

    # Save optimizer state
    checkpointer.save(
        path / 'optimizer.msgpack',
        state.opt_state,
        force=True
    )

    # Save each node
    for node_name, variables in state.variables.items():
        save_node(node_name, variables, str(path / f'node_{node_name}'))


def save_node(node_name: str, variables: dict[str, Any], path: str):
    """Save a single node's variables with metadata.

    Args:
        node_name: Name of the node
        variables: Variables dictionary {params: ..., batch_stats: ...}
        path: Directory path to save node checkpoint
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

    # Create metadata
    metadata = {
        'node_name': node_name,
        'collections': list(variables.keys()),
        'shapes': _pytree_shapes(variables),
        'dtypes': _pytree_dtypes(variables),
    }

    with open(path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save variables
    checkpointer.save(
        path / 'variables.msgpack',
        variables,
        force=True
    )


def save_node_to_single_file(
    node_name: str,
    variables: dict[str, Any],
    filepath: str
):
    """Save node variables to a single .msgpack file.

    Useful for sharing pretrained models.

    Args:
        node_name: Name of the node
        variables: Variables dictionary
        filepath: Path to .msgpack file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    checkpointer.save(str(filepath), variables, force=True)


def _pytree_shapes(pytree: Any) -> dict:
    """Extract shapes from a pytree recursively."""
    def get_shape(x):
        if hasattr(x, 'shape'):
            return list(x.shape)
        return None

    return jax.tree_util.tree_map(get_shape, pytree)


def _pytree_dtypes(pytree: Any) -> dict:
    """Extract dtypes from a pytree recursively."""
    def get_dtype(x):
        if hasattr(x, 'dtype'):
            return str(x.dtype)
        return None

    return jax.tree_util.tree_map(get_dtype, pytree)
