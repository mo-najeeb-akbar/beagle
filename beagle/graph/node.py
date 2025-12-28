"""ComputeNode: A computational node wrapping a Flax module.

A node can be:
- A single neural network
- A preprocessing function
- An entire ComputeGraph (graphs are nodes!)
"""

from dataclasses import dataclass
from typing import Callable, Any
import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass
class ComputeNode:
    """A computational node in a compute graph.

    Args:
        name: Unique identifier for this node
        module: Flax nn.Module to wrap
        inputs: List of input keys this node expects
        outputs: List of output keys this node produces
        trainable: Whether this node should be trained
        checkpoint_path: Optional path to load pretrained weights
        mutable_collections: List of mutable collection names (auto-detected if None)

    Example:
        >>> encoder = ComputeNode(
        ...     name="encoder",
        ...     module=ResNet50(),
        ...     inputs=["image"],
        ...     outputs=["features"],
        ...     trainable=False,
        ...     checkpoint_path="pretrained/encoder.ckpt"
        ... )
    """
    name: str
    module: nn.Module
    inputs: list[str]
    outputs: list[str]
    trainable: bool = True
    checkpoint_path: str | None = None
    mutable_collections: list[str] | None = None

    def __call__(
        self,
        variables: dict[str, Any],
        inputs: dict[str, Any],
        *,
        train: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply node to inputs.

        Args:
            variables: Model variables {'params': ..., 'batch_stats': ..., ...}
            inputs: Input dictionary containing all available data
            train: Whether in training mode

        Returns:
            outputs: Dictionary of outputs from this node
            updates: Dictionary of updated mutable collections
        """
        # Extract only the inputs this node needs
        node_inputs = {k: inputs[k] for k in self.inputs if k in inputs}

        # Check all required inputs are present
        missing = set(self.inputs) - set(node_inputs.keys())
        if missing:
            raise ValueError(
                f"Node '{self.name}' missing required inputs: {missing}. "
                f"Available inputs: {list(inputs.keys())}"
            )

        # Auto-detect mutable collections if not specified
        mutable = self.mutable_collections
        if mutable is None:
            # All collections except 'params' are considered mutable
            mutable = [k for k in variables.keys() if k != 'params']

        # Convert inputs dict to positional args (in order of self.inputs)
        input_args = [node_inputs[k] for k in self.inputs]

        # Apply module
        if len(mutable) > 0:
            result, updates = self.module.apply(
                variables,
                *input_args,
                train=train,
                mutable=mutable
            )
        else:
            result = self.module.apply(
                variables,
                *input_args,
                train=train
            )
            updates = {}

        # Wrap outputs in dictionary
        if len(self.outputs) == 1:
            # Single output - wrap scalar/tensor result
            outputs = {self.outputs[0]: result}
        else:
            # Multiple outputs - assume result is tuple/list
            if not isinstance(result, (tuple, list)):
                raise ValueError(
                    f"Node '{self.name}' declares {len(self.outputs)} outputs "
                    f"but module returned non-sequence: {type(result)}"
                )
            if len(result) != len(self.outputs):
                raise ValueError(
                    f"Node '{self.name}' declares {len(self.outputs)} outputs "
                    f"but module returned {len(result)} values"
                )
            outputs = dict(zip(self.outputs, result))

        return outputs, updates

    def init(
        self,
        rng: jax.Array,
        sample_inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Initialize variables for this node.

        If checkpoint_path is set, loads from checkpoint.
        Otherwise, initializes fresh.

        Args:
            rng: JAX random key
            sample_inputs: Sample input batch for shape inference

        Returns:
            variables: Initialized variables dictionary
        """
        if self.checkpoint_path is not None:
            return self.load_checkpoint()

        # Fresh initialization
        node_inputs = {k: sample_inputs[k] for k in self.inputs if k in sample_inputs}

        # Check all required inputs are present
        missing = set(self.inputs) - set(node_inputs.keys())
        if missing:
            raise ValueError(
                f"Node '{self.name}' missing required inputs for init: {missing}"
            )

        # Convert to positional args
        input_args = [node_inputs[k] for k in self.inputs]

        variables = self.module.init(rng, *input_args, train=False)
        return variables

    def load_checkpoint(self) -> dict[str, Any]:
        """Load variables from checkpoint_path.

        Returns:
            variables: Loaded variables dictionary
        """
        if self.checkpoint_path is None:
            raise ValueError(f"Node '{self.name}' has no checkpoint_path set")

        import orbax.checkpoint as ocp
        from pathlib import Path

        path = Path(self.checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for node '{self.name}': {self.checkpoint_path}"
            )

        # If path is a directory, look for variables.msgpack inside
        if path.is_dir():
            ckpt_file = path / 'variables.msgpack'
            if not ckpt_file.exists():
                raise FileNotFoundError(
                    f"No variables.msgpack found in checkpoint directory: {path}"
                )
            path = ckpt_file

        checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        variables = checkpointer.restore(str(path.absolute()))

        return variables

    def save_checkpoint(self, variables: dict[str, Any], path: str):
        """Save node variables to checkpoint.

        Args:
            variables: Variables to save
            path: Path to save checkpoint
        """
        from pathlib import Path
        from .save import save_node

        save_node(self.name, variables, path)

    def __repr__(self) -> str:
        trainable_str = "trainable" if self.trainable else "frozen"
        ckpt_str = f", checkpoint={self.checkpoint_path}" if self.checkpoint_path else ""
        return (
            f"ComputeNode('{self.name}', "
            f"inputs={self.inputs}, outputs={self.outputs}, "
            f"{trainable_str}{ckpt_str})"
        )
