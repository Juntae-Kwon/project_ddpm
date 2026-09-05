"""A small timestep-conditioned U-Net that predicts DDPM noise."""

from flax import linen as nn
import jax
import jax.numpy as jnp


def sinusoidal_timestep_embedding(timesteps, embedding_dim):
    """Encode scalar timesteps with fixed sine and cosine frequencies."""
    if embedding_dim < 4 or embedding_dim % 2:
        raise ValueError("embedding_dim must be even and at least 4")
    half_dim = embedding_dim // 2
    frequencies = jnp.exp(
        -jnp.log(10_000.0) * jnp.arange(half_dim) / (half_dim - 1)
    )
    angles = timesteps.astype(jnp.float32)[:, None] * frequencies[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class ResidualBlock(nn.Module):
    """Two convolutions conditioned by the shared timestep embedding."""

    channels: int

    @nn.compact
    def __call__(self, x, time_embedding):
        residual = x

        x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.silu(x)

        time_bias = nn.Dense(self.channels)(nn.silu(time_embedding))
        x = x + time_bias[:, None, None, :]

        x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=8)(x)
        if residual.shape[-1] != self.channels:
            residual = nn.Conv(self.channels, kernel_size=(1, 1))(residual)
        return nn.silu(x + residual)


class UNet(nn.Module):
    """A compact 28 -> 14 -> 7 -> 14 -> 28 denoising network."""

    base_channels: int = 32

    @nn.compact
    def __call__(self, noisy_images, timesteps):
        if self.base_channels < 8 or self.base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of 8")

        time_embedding = sinusoidal_timestep_embedding(
            timesteps, self.base_channels
        )
        time_embedding = nn.Dense(4 * self.base_channels)(time_embedding)
        time_embedding = nn.silu(time_embedding)
        time_embedding = nn.Dense(4 * self.base_channels)(time_embedding)

        x = nn.Conv(self.base_channels, kernel_size=(3, 3), padding="SAME")(
            noisy_images
        )
        skip_28 = ResidualBlock(self.base_channels)(x, time_embedding)

        x = nn.Conv(
            2 * self.base_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
        )(skip_28)
        skip_14 = ResidualBlock(2 * self.base_channels)(x, time_embedding)

        x = nn.Conv(
            4 * self.base_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
        )(skip_14)
        x = ResidualBlock(4 * self.base_channels)(x, time_embedding)
        x = ResidualBlock(4 * self.base_channels)(x, time_embedding)

        x = nn.ConvTranspose(
            2 * self.base_channels,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
        )(x)
        x = jnp.concatenate([x, skip_14], axis=-1)
        x = ResidualBlock(2 * self.base_channels)(x, time_embedding)

        x = nn.ConvTranspose(
            self.base_channels,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
        )(x)
        x = jnp.concatenate([x, skip_28], axis=-1)
        x = ResidualBlock(self.base_channels)(x, time_embedding)

        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.silu(x)
        return nn.Conv(noisy_images.shape[-1], kernel_size=(3, 3), padding="SAME")(
            x
        )
