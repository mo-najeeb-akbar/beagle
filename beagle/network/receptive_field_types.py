"""Data types for receptive field computation.

Based on "Computing Receptive Fields of Convolutional Neural Networks"
by Araujo, Norris, and Sim (2019): https://distill.pub/2019/computing-receptive-fields/
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerConfig:
    """Configuration for a convolutional layer.

    Attributes:
        kernel_size: Spatial extent of the kernel (positive integer)
        stride: Stride of the operation (positive integer)
        padding: Padding applied (non-negative integer or tuple for left/right)
        dilation: Dilation factor for atrous convolution (positive integer)
        name: Optional layer name for debugging
    """

    kernel_size: int | tuple[int, int]
    stride: int | tuple[int, int] = 1
    padding: int | tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = 0
    dilation: int | tuple[int, int] = 1
    name: str = ""

    def __post_init__(self) -> None:
        """Validate layer configuration."""
        if isinstance(self.kernel_size, int):
            if self.kernel_size < 1:
                raise ValueError(f"kernel_size must be positive, got {self.kernel_size}")
        else:
            if any(k < 1 for k in self.kernel_size):
                raise ValueError(
                    f"kernel_size must be positive, got {self.kernel_size}"
                )

        if isinstance(self.stride, int):
            if self.stride < 1:
                raise ValueError(f"stride must be positive, got {self.stride}")
        else:
            if any(s < 1 for s in self.stride):
                raise ValueError(f"stride must be positive, got {self.stride}")


@dataclass(frozen=True)
class ReceptiveFieldInfo:
    """Receptive field information for a network or layer.

    Attributes:
        size: Size of the receptive field in the input
        stride: Effective stride from input to this layer
        padding: Effective padding from input
    """

    size: int | tuple[int, int]
    stride: int | tuple[int, int]
    padding: int | tuple[int, int]

    def __str__(self) -> str:
        """Human-readable representation."""
        if isinstance(self.size, int):
            return (
                f"Receptive Field: {self.size}x{self.size} pixels, "
                f"stride={self.stride}, padding={self.padding}"
            )
        return (
            f"Receptive Field: {self.size[0]}x{self.size[1]} pixels, "
            f"stride={self.stride}, padding={self.padding}"
        )

