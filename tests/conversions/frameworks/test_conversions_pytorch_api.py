from __future__ import annotations

import pytest

from beagle.conversions.frameworks.pytorch import create_pytorch_registry


def test_create_pytorch_registry() -> None:
    registry = create_pytorch_registry()

    assert registry.has_transfer("Conv2d")
    assert registry.has_transfer("ConvTranspose2d")
    assert registry.has_transfer("Linear")
    assert registry.has_transfer("GroupNorm")
    assert registry.has_transfer("BatchNorm2d")

    assert registry.has_verify("Conv2d")
    assert registry.has_verify("ConvTranspose2d")
    assert registry.has_verify("Linear")
    assert registry.has_verify("GroupNorm")
    assert registry.has_verify("BatchNorm2d")


def test_pytorch_registry_get_functions() -> None:
    registry = create_pytorch_registry()

    assert registry.get_transfer("Conv2d") is not None
    assert registry.get_transfer("ConvTranspose2d") is not None
    assert registry.get_transfer("Linear") is not None
    assert registry.get_verify("Conv2d") is not None
    assert registry.get_verify("ConvTranspose2d") is not None
    assert registry.get_verify("Linear") is not None


def test_pytorch_registry_no_unknown_layers() -> None:
    registry = create_pytorch_registry()

    assert registry.get_transfer("UnknownLayer") is None
    assert registry.get_verify("UnknownLayer") is None
    assert not registry.has_transfer("UnknownLayer")
    assert not registry.has_verify("UnknownLayer")

