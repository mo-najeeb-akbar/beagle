"""Beagle Graph API: Composable compute graphs for neural networks.

Core primitives:
- ComputeNode: Wraps a Flax module as a graph node
- ComputeGraph: DAG of nodes with automatic execution
- GraphState: Training state with automatic gradient handling
- Training utilities: Generic training and evaluation functions
"""

from .node import ComputeNode
from .graph import ComputeGraph
from .state import GraphState
from .training import (
    create_train_step,
    create_eval_step,
    train_epoch,
    evaluate,
    simple_training_loop,
)

__all__ = [
    'ComputeNode',
    'ComputeGraph',
    'GraphState',
    'create_train_step',
    'create_eval_step',
    'train_epoch',
    'evaluate',
    'simple_training_loop',
]
