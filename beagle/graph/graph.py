"""ComputeGraph: A directed acyclic graph of ComputeNodes.

Handles:
- Topological execution order
- Data routing between nodes
- Filtering trainable vs frozen nodes
"""

from typing import Any
import jax
from .node import ComputeNode


class ComputeGraph:
    """A directed acyclic graph of computational nodes.

    Args:
        nodes: Dictionary mapping node names to ComputeNode instances
        edges: Optional dependency graph {node_name: [dependency_names]}
               If None, inferred from node inputs/outputs

    Example:
        >>> encoder = ComputeNode("encoder", ResNet(), ["image"], ["features"])
        >>> decoder = ComputeNode("decoder", UNet(), ["features"], ["logits"])
        >>> graph = ComputeGraph(
        ...     nodes={"encoder": encoder, "decoder": decoder}
        ... )
        >>> # Edges auto-inferred: decoder depends on encoder
    """

    def __init__(
        self,
        nodes: dict[str, ComputeNode],
        edges: dict[str, list[str]] | None = None
    ):
        self.nodes = nodes

        # Auto-detect edges from inputs/outputs if not provided
        if edges is None:
            edges = self._infer_edges()
        self.edges = edges

        # Compute topological execution order
        self.execution_order = self._topological_sort()

    def _infer_edges(self) -> dict[str, list[str]]:
        """Infer dependency edges from node inputs/outputs.

        Returns:
            edges: {node_name: [dependency_names]}
        """
        edges = {name: [] for name in self.nodes}

        # Build mapping: output_key -> node_name that produces it
        output_providers = {}
        for name, node in self.nodes.items():
            for output in node.outputs:
                if output in output_providers:
                    raise ValueError(
                        f"Multiple nodes produce output '{output}': "
                        f"{output_providers[output]} and {name}"
                    )
                output_providers[output] = name

        # Find dependencies for each node
        for name, node in self.nodes.items():
            for input_key in node.inputs:
                if input_key in output_providers:
                    provider = output_providers[input_key]
                    if provider not in edges[name]:
                        edges[name].append(provider)

        return edges

    def _topological_sort(self) -> list[str]:
        """Compute topological ordering of nodes.

        Returns:
            order: List of node names in execution order

        Raises:
            ValueError: If graph has cycles
        """
        # Calculate in-degrees
        in_degree = {name: 0 for name in self.nodes}
        for deps in self.edges.values():
            for dep in deps:
                in_degree[dep] = in_degree.get(dep, 0)

        # Count incoming edges properly
        for node_name, deps in self.edges.items():
            for dep in deps:
                in_degree[node_name] += 1

        # Start with nodes that have no dependencies
        queue = [name for name, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            # Process node with no remaining dependencies
            node = queue.pop(0)
            order.append(node)

            # Reduce in-degree for nodes that depend on this one
            for child, deps in self.edges.items():
                if node in deps:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        # Check for cycles
        if len(order) != len(self.nodes):
            remaining = set(self.nodes.keys()) - set(order)
            raise ValueError(
                f"Graph has cycles or unreachable nodes. "
                f"Could not process: {remaining}"
            )

        return order

    def __call__(
        self,
        variables: dict[str, dict[str, Any]],
        inputs: dict[str, Any],
        *,
        train: bool = False
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Execute graph on inputs.

        Args:
            variables: {node_name: {params: ..., batch_stats: ...}}
            inputs: External inputs {key: array}
            train: Whether in training mode

        Returns:
            outputs: All outputs produced by nodes
            updates: {node_name: {collection: ...}} for mutable collections
        """
        # Start with external inputs
        data = inputs.copy()
        all_updates = {}

        # Execute nodes in topological order
        for node_name in self.execution_order:
            node = self.nodes[node_name]
            node_vars = variables[node_name]

            # Execute node (it will extract what it needs from data)
            outputs, updates = node(node_vars, data, train=train)

            # Accumulate outputs for downstream nodes
            data.update(outputs)
            all_updates[node_name] = updates

        return data, all_updates

    def init(
        self,
        rng: jax.Array,
        sample_inputs: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Initialize all nodes in the graph.

        Nodes with checkpoint_path: load from checkpoint
        Nodes without checkpoint: initialize fresh

        Args:
            rng: JAX random key
            sample_inputs: Sample batch for shape inference

        Returns:
            variables: {node_name: {params: ..., batch_stats: ...}}
        """
        variables = {}
        data = sample_inputs.copy()

        for node_name in self.execution_order:
            node = self.nodes[node_name]

            # Initialize this node
            rng, init_rng = jax.random.split(rng)
            node_vars = node.init(init_rng, data)
            variables[node_name] = node_vars

            # Run forward to generate outputs for downstream nodes
            # (needed for shape inference)
            outputs, _ = node(node_vars, data, train=False)
            data.update(outputs)

        return variables

    def trainable_nodes(self) -> list[str]:
        """Return names of trainable nodes.

        Returns:
            names: List of trainable node names
        """
        return [name for name, node in self.nodes.items() if node.trainable]

    def freeze_node(self, node_name: str):
        """Make a node non-trainable.

        Args:
            node_name: Name of node to freeze
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")
        self.nodes[node_name].trainable = False

    def unfreeze_node(self, node_name: str):
        """Make a node trainable.

        Args:
            node_name: Name of node to unfreeze
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")
        self.nodes[node_name].trainable = True

    def freeze_all(self):
        """Freeze all nodes in the graph."""
        for node in self.nodes.values():
            node.trainable = False

    def unfreeze_all(self):
        """Unfreeze all nodes in the graph."""
        for node in self.nodes.values():
            node.trainable = True

    def get_node(self, name: str) -> ComputeNode:
        """Get a node by name.

        Args:
            name: Node name

        Returns:
            node: The ComputeNode instance
        """
        if name not in self.nodes:
            raise ValueError(
                f"Node '{name}' not found. Available nodes: {list(self.nodes.keys())}"
            )
        return self.nodes[name]

    def __repr__(self) -> str:
        trainable = self.trainable_nodes()
        frozen = [n for n in self.nodes if n not in trainable]
        return (
            f"ComputeGraph(\n"
            f"  nodes={list(self.nodes.keys())},\n"
            f"  trainable={trainable},\n"
            f"  frozen={frozen},\n"
            f"  execution_order={self.execution_order}\n"
            f")"
        )
