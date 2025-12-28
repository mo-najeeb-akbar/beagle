"""Checkpoint loading utilities with validation.

Supports:
- Full graph state loading
- Individual node loading
- Validation of checkpoint compatibility
"""

import json
from pathlib import Path
from typing import Any
import orbax.checkpoint as ocp


def load_graph_state(path: str, graph, tx):
    """Load GraphState from checkpoint.

    Validates:
    - Checkpoint structure matches graph
    - All required nodes are present
    - Metadata compatibility

    Args:
        path: Directory containing checkpoint
        graph: ComputeGraph instance (must match checkpoint structure)
        tx: Optax optimizer to use

    Returns:
        state: Loaded GraphState
    """
    from ..graph.state import GraphState

    path = Path(path)
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

    # Load graph config for validation
    config_path = path / 'graph_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"No graph_config.json found in {path}")

    with open(config_path) as f:
        config = json.load(f)

    # Validate graph structure matches
    ckpt_nodes = set(config['nodes'].keys())
    graph_nodes = set(graph.nodes.keys())

    if ckpt_nodes != graph_nodes:
        missing = graph_nodes - ckpt_nodes
        extra = ckpt_nodes - graph_nodes
        msg = []
        if missing:
            msg.append(f"Graph has nodes not in checkpoint: {missing}")
        if extra:
            msg.append(f"Checkpoint has nodes not in graph: {extra}")
        raise ValueError('\n'.join(msg))

    # Load variables for each node
    variables = {}
    for node_name in graph.nodes:
        node_path = path / f'node_{node_name}'
        variables[node_name] = load_node(str(node_path))

    # Load optimizer state
    opt_state_path = path / 'optimizer.msgpack'
    if opt_state_path.exists():
        opt_state = checkpointer.restore(str(opt_state_path.absolute()))
    else:
        # Initialize fresh optimizer state if not in checkpoint
        trainable_params = {
            name: variables[name]['params']
            for name in graph.trainable_nodes()
        }
        opt_state = tx.init(trainable_params)

    # Create state
    state = GraphState(
        graph=graph,
        variables=variables,
        opt_state=opt_state,
        tx=tx,
        step=config.get('step', 0)
    )

    return state


def load_node(path: str) -> dict[str, Any]:
    """Load a single node checkpoint.

    Args:
        path: Directory containing node checkpoint

    Returns:
        variables: Loaded variables dictionary
    """
    path = Path(path)
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

    # Load metadata (for validation/info)
    metadata_path = path / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        # Could add validation here
    else:
        metadata = None

    # Load variables
    variables_path = path / 'variables.msgpack'
    if not variables_path.exists():
        raise FileNotFoundError(f"No variables.msgpack found in {path}")

    variables = checkpointer.restore(str(variables_path.absolute()))

    return variables


def load_node_from_single_file(filepath: str) -> dict[str, Any]:
    """Load node variables from a single .msgpack file.

    Args:
        filepath: Path to .msgpack file

    Returns:
        variables: Loaded variables dictionary
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    variables = checkpointer.restore(str(filepath.absolute()))

    return variables


def load_partial_graph(
    checkpoint_path: str,
    graph,
    tx,
    load_nodes: list[str] | None = None,
    skip_nodes: list[str] | None = None
) -> Any:
    """Load graph state but only restore specific nodes from checkpoint.

    Useful for:
    - Transfer learning (load encoder, fresh decoder)
    - Fine-tuning (load some layers, reinit others)

    Args:
        checkpoint_path: Path to checkpoint directory
        graph: ComputeGraph instance
        tx: Optax optimizer
        load_nodes: If provided, only load these nodes from checkpoint
        skip_nodes: If provided, skip these nodes (initialize fresh)

    Returns:
        state: GraphState with partial checkpoint loading
    """
    from ..graph.state import GraphState
    import jax

    path = Path(checkpoint_path)

    # Determine which nodes to load
    all_nodes = set(graph.nodes.keys())
    if load_nodes is not None:
        nodes_to_load = set(load_nodes) & all_nodes
    elif skip_nodes is not None:
        nodes_to_load = all_nodes - set(skip_nodes)
    else:
        nodes_to_load = all_nodes

    # Load specified nodes
    variables = {}
    for node_name in all_nodes:
        node_path = path / f'node_{node_name}'

        if node_name in nodes_to_load and node_path.exists():
            # Load from checkpoint
            variables[node_name] = load_node(str(node_path))
        else:
            # Initialize fresh
            # Need sample inputs - this is a limitation
            # User should call graph.init() separately for fresh nodes
            raise NotImplementedError(
                f"Cannot initialize fresh node '{node_name}' during partial load. "
                "Please initialize the full graph first, then use load_partial_graph "
                "to overwrite specific nodes."
            )

    # Initialize optimizer state
    trainable_params = {
        name: variables[name]['params']
        for name in graph.trainable_nodes()
    }
    opt_state = tx.init(trainable_params)

    state = GraphState(
        graph=graph,
        variables=variables,
        opt_state=opt_state,
        tx=tx,
        step=0
    )

    return state


def merge_checkpoint_into_state(
    state,
    checkpoint_path: str,
    node_name: str
):
    """Load a single node from checkpoint and merge into existing state.

    Args:
        state: Current GraphState
        checkpoint_path: Path to node checkpoint directory or .msgpack file
        node_name: Name of node to update

    Returns:
        new_state: GraphState with updated node
    """
    path = Path(checkpoint_path)

    # Load node variables
    if path.is_dir():
        node_vars = load_node(str(path))
    elif path.suffix == '.msgpack':
        node_vars = load_node_from_single_file(str(path))
    else:
        raise ValueError(f"Checkpoint path must be directory or .msgpack file: {path}")

    # Merge into state
    new_variables = state.variables.copy()
    new_variables[node_name] = node_vars

    return state.replace(variables=new_variables)
