"""GraphState: Training state for a ComputeGraph.

Handles:
- Filtering trainable vs frozen variables
- Applying gradients only to trainable nodes
- Merging updated mutable collections
"""

from typing import Any
from flax.struct import dataclass
import optax
import jax
import jax.numpy as jnp
from .graph import ComputeGraph


@dataclass
class GraphState:
    """Training state for a computational graph.

    Automatically handles:
    - Optimizer state only for trainable nodes
    - Gradient application to trainable parameters
    - Mutable collection updates (batch_stats, etc.)

    Args:
        graph: The ComputeGraph being trained
        variables: {node_name: {params: ..., batch_stats: ...}}
        opt_state: Optimizer state for trainable nodes
        tx: Optax gradient transformation
        step: Current training step

    Example:
        >>> graph = ComputeGraph(nodes={...})
        >>> variables = graph.init(rng, sample_batch)
        >>> state = GraphState.create(
        ...     graph=graph,
        ...     variables=variables,
        ...     tx=optax.adam(1e-4)
        ... )
        >>> # Training
        >>> grads = compute_gradients(...)
        >>> state = state.apply_gradients(grads)
    """
    graph: ComputeGraph
    variables: dict[str, dict[str, Any]]
    opt_state: Any
    tx: optax.GradientTransformation
    step: int = 0

    @classmethod
    def create(
        cls,
        graph: ComputeGraph,
        variables: dict[str, dict[str, Any]],
        tx: optax.GradientTransformation
    ) -> "GraphState":
        """Create initial training state.

        Args:
            graph: ComputeGraph to train
            variables: Initialized variables from graph.init()
            tx: Optax optimizer (e.g., optax.adam(1e-4))

        Returns:
            state: Initial GraphState
        """
        # Initialize optimizer state only for trainable nodes
        trainable_params = {
            name: variables[name]['params']
            for name in graph.trainable_nodes()
        }

        opt_state = tx.init(trainable_params)

        return cls(
            graph=graph,
            variables=variables,
            opt_state=opt_state,
            tx=tx,
            step=0
        )

    def apply_gradients(
        self,
        grads: dict[str, dict[str, Any]]
    ) -> "GraphState":
        """Apply gradients to trainable nodes only.

        Args:
            grads: {node_name: {'params': grad_tree}}

        Returns:
            new_state: Updated GraphState with new parameters
        """
        # Filter to trainable nodes
        trainable_nodes = self.graph.trainable_nodes()
        trainable_grads = {
            name: grads[name]['params'] if isinstance(grads[name], dict) else grads[name]
            for name in trainable_nodes
            if name in grads
        }

        # Get current trainable params
        trainable_params = {
            name: self.variables[name]['params']
            for name in trainable_nodes
        }

        # Apply optimizer update
        updates, new_opt_state = self.tx.update(
            trainable_grads, self.opt_state, trainable_params
        )
        new_trainable_params = optax.apply_updates(trainable_params, updates)

        # Merge back into full variables
        new_variables = {}
        for name, node_vars in self.variables.items():
            if name in new_trainable_params:
                # Update trainable node params
                new_variables[name] = {
                    **node_vars,
                    'params': new_trainable_params[name]
                }
            else:
                # Keep frozen node vars unchanged
                new_variables[name] = node_vars

        return self.replace(
            variables=new_variables,
            opt_state=new_opt_state,
            step=self.step + 1
        )

    def merge_updates(
        self,
        updates: dict[str, dict[str, Any]]
    ) -> "GraphState":
        """Merge updated mutable collections back into variables.

        Args:
            updates: {node_name: {batch_stats: ..., cache: ...}}

        Returns:
            new_state: GraphState with updated collections
        """
        new_variables = {}

        for node_name, node_vars in self.variables.items():
            if node_name in updates and updates[node_name]:
                # Merge updates for this node
                new_variables[node_name] = {
                    **node_vars,
                    **updates[node_name]
                }
            else:
                # No updates for this node
                new_variables[node_name] = node_vars

        return self.replace(variables=new_variables)

    def trainable_variables(self) -> dict[str, dict[str, Any]]:
        """Get variables for trainable nodes only.

        Returns:
            variables: {node_name: {params: ..., batch_stats: ...}}
        """
        return {
            name: self.variables[name]
            for name in self.graph.trainable_nodes()
        }

    def trainable_params(self) -> dict[str, Any]:
        """Get only the params pytree for trainable nodes.

        Useful for gradient computation.

        Returns:
            params: {node_name: params_pytree}
        """
        return {
            name: self.variables[name]['params']
            for name in self.graph.trainable_nodes()
        }

    def __call__(
        self,
        inputs: dict[str, Any],
        *,
        train: bool = False
    ) -> dict[str, Any]:
        """Execute graph forward pass (inference mode).

        Args:
            inputs: Input dictionary
            train: Whether in training mode

        Returns:
            outputs: All outputs from the graph
        """
        outputs, _ = self.graph(self.variables, inputs, train=train)
        return outputs

    def apply_with_updates(
        self,
        inputs: dict[str, Any],
        *,
        train: bool = False
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Execute graph and return outputs + mutable collection updates.

        Use this during training to get batch_stats updates.

        Args:
            inputs: Input dictionary
            train: Whether in training mode

        Returns:
            outputs: All outputs from the graph
            updates: {node_name: {collection: ...}} for mutable collections
        """
        outputs, updates = self.graph(self.variables, inputs, train=train)
        return outputs, updates

    def freeze_node(self, node_name: str) -> "GraphState":
        """Freeze a node (make it non-trainable).

        Args:
            node_name: Name of node to freeze

        Returns:
            new_state: GraphState with node frozen and optimizer state updated
        """
        # Freeze in graph
        self.graph.freeze_node(node_name)

        # Reinitialize optimizer state without this node
        trainable_params = self.trainable_params()
        new_opt_state = self.tx.init(trainable_params)

        return self.replace(opt_state=new_opt_state)

    def unfreeze_node(self, node_name: str) -> "GraphState":
        """Unfreeze a node (make it trainable).

        Args:
            node_name: Name of node to unfreeze

        Returns:
            new_state: GraphState with node unfrozen and optimizer state updated
        """
        # Unfreeze in graph
        self.graph.unfreeze_node(node_name)

        # Reinitialize optimizer state with this node
        trainable_params = self.trainable_params()
        new_opt_state = self.tx.init(trainable_params)

        return self.replace(opt_state=new_opt_state)

    def __repr__(self) -> str:
        trainable = self.graph.trainable_nodes()
        return (
            f"GraphState(\n"
            f"  step={self.step},\n"
            f"  trainable_nodes={trainable},\n"
            f"  frozen_nodes={[n for n in self.graph.nodes if n not in trainable]}\n"
            f")"
        )
