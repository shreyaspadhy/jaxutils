"""Flax implementations of LeNet variants."""
from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp

Moduledef = Any


class conv3_block(nn.Module):
    num_filters: int
    stride: int = 1
    bias: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=self.bias, dtype=self.dtype)
        norm = partial(nn.BatchNorm,
                       use_running_average=not train,
                       momentum=0.9,
                       epsilon=1e-5,
                       dtype=self.dtype)

        x = conv(self.num_filters, (3, 3), (self.stride, self.stride),
                 padding=[(1, 1), (1, 1)])(x)
        x = nn.relu(x)
        x = norm()(x)

        return x


class conv5_block(nn.Module):
    num_filters: int
    stride: int = 1
    bias: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=self.bias, dtype=self.dtype)
        norm = partial(nn.BatchNorm,
                       use_running_average=not train,
                       momentum=0.9,
                       epsilon=1e-5,
                       dtype=self.dtype)

        x = conv(self.num_filters, (5, 5), (self.stride, self.stride),
                 padding=[(2, 2), (2, 2)])(x)
        x = nn.relu(x)
        x = norm()(x)

        return x


class LeNet(nn.Module):
    n_out: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = conv5_block(32, stride=2)(x)
        x = conv3_block(32, stride=2)(x)
        x = conv3_block(64, stride=2)(x)

        # Similar to PyTorch nn.AdaptiveAvgPool2d((1, 1))
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.n_out, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)

        return x


class LeNetSmall(nn.Module):
    n_out: int
    dtype: Any = jnp.float32

    def setup(self):
        self.conv1 = conv5_block(16, stride=2)
        self.conv2 = conv3_block(32, stride=2)
        self.conv3 = conv3_block(32, stride=2)

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = self.conv1(x, train=train)
        x = self.conv2(x, train=train)
        x = self.conv3(x, train=train)

        # Similar to PyTorch nn.AdaptiveAvgPool2d((1, 1))
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.n_out, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)

        return x
