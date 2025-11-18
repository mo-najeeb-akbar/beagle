import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ResidualBlock(layers.Layer):
    """Residual block with group normalization."""

    def __init__(self, filters: int, name: str | None = None) -> None:
        super().__init__(name=name)
        self.filters = filters

    def build(self, input_shape: tuple[int, ...]) -> None:
        self.conv1 = layers.Conv2D(
            self.filters, (3, 3), padding="same", use_bias=False, name="conv1"
        )
        self.gn1 = tf.keras.layers.GroupNormalization(groups=8, epsilon=1e-5, name="gn1")
        self.conv2 = layers.Conv2D(
            self.filters, (3, 3), padding="same", use_bias=False, name="conv2"
        )
        self.gn2 = tf.keras.layers.GroupNormalization(groups=8, epsilon=1e-5, name="gn2")
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        skip = x

        out = self.conv1(x)
        out = self.gn1(out, training=training)
        out = tf.nn.swish(out)

        out = self.conv2(out)
        out = self.gn2(out, training=training)

        return tf.nn.swish(out + skip)


class Encoder(layers.Layer):
    """VAE encoder with downsampling conv blocks."""

    def __init__(
        self, latent_dim: int, features: int, name: str | None = None
    ) -> None:
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.features = features

    def build(self, input_shape: tuple[int, ...]) -> None:
        self.conv_layers = []
        self.gn_layers = []
        self.residual_blocks = []

        self.haar_conv = layers.Conv2D(
            4, kernel_size=2, strides=2, use_bias=False, name="haar_conv"
        )
        self.gn_haar = tf.keras.layers.GroupNormalization(groups=4, epsilon=1e-5, name="gn_haar")

        for i in range(5):
            self.conv_layers.append(
                layers.Conv2D(
                    self.features,
                    (3, 3),
                    strides=(2, 2),
                    padding="same",
                    use_bias=False,
                    name=f"conv_layers.{i}",
                )
            )
            self.gn_layers.append(
                tf.keras.layers.GroupNormalization(groups=8, epsilon=1e-5, name=f"gn_layers.{i}")
            )
            self.residual_blocks.append(ResidualBlock(self.features))

        self.dense_mu = layers.Dense(self.latent_dim, name="dense_mu")
        self.dense_logvar = layers.Dense(self.latent_dim, name="dense_logvar")
        super().build(input_shape)

    def call(
        self, x: tf.Tensor, training: bool = True
    ) -> tuple[tf.Tensor, tf.Tensor]:
        x = self.haar_conv(x)
        x = self.gn_haar(x, training=training)
        for i in range(5):
            x = self.conv_layers[i](x)
            x = self.gn_layers[i](x, training=training)
            x = tf.nn.swish(x)
            x = self.residual_blocks[i](x, training=training)

        mu = self.dense_mu(x)
        log_var = self.dense_logvar(x)

        return mu, log_var


class Decoder(layers.Layer):
    """VAE decoder with upsampling conv blocks."""

    def __init__(
        self,
        latent_dim: int,
        bottle_neck: int,
        features: int,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.bottle_neck = bottle_neck
        self.features = features

    def build(self, input_shape: tuple[int, ...]) -> None:
        self.conv_layers = []
        self.gn_layers = []
        self.residual_blocks = []

        for i in range(5):
            self.conv_layers.append(
                layers.Conv2D(
                    self.features, (3, 3), padding="same", use_bias=False, name=f"conv_layers.{i}"
                )
            )
            self.gn_layers.append(
                tf.keras.layers.GroupNormalization(groups=8, epsilon=1e-5, name=f"gn_layers.{i}")
            )
            self.residual_blocks.append(ResidualBlock(self.features))

        self.out_conv = layers.Conv2D(4, (3, 3), padding="same", name="out_conv")
        self.haar_conv_transpose = layers.Conv2DTranspose(
            1, kernel_size=(2, 2), strides=(2, 2), use_bias=False, name="haar_conv_transpose"
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        for i in range(5):
            batch, height, width, channels = tf.unstack(tf.shape(x))
            x = keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
            x = self.conv_layers[i](x)
            x = self.gn_layers[i](x, training=training)
            x = tf.nn.swish(x)
            x = self.residual_blocks[i](x, training=training)

        x_haar = self.out_conv(x)
        x_recon = self.haar_conv_transpose(x_haar)
        return x_recon, x_haar


class VAETF(keras.Model):
    """Wavelet-based VAE for image data."""

    def __init__(
        self, latent_dim: int = 128, base_features: int = 32, block_size: int = 8
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.base_features = base_features
        self.block_size = block_size

        self.encoder = Encoder(latent_dim, base_features)
        self.decoder = Decoder(latent_dim, block_size, base_features)


    def encode(
        self, x: tf.Tensor, training: bool = True
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Encode image to latent distribution."""
        return self.encoder(x, training=training)

    def decode(self, z: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Decode latent to wavelet coefficients."""
        x_recon, x_haar = self.decoder(z, training=training)
        return x_recon, x_haar

    def reparameterize(
        self, mu: tf.Tensor, log_var: tf.Tensor, training: bool = True
    ) -> tf.Tensor:
        """Reparameterization (currently disabled, returns mu only)."""
        return mu

    def call(
        self, x: tf.Tensor, training: bool = True
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Forward pass through VAE.

        Returns:
            reconstructed: Full reconstructed image
            x_waves: Wavelet coefficients
            mu: Latent mean
            log_var: Latent log variance
        """
 
        mu, log_var = self.encode(x, training)
        z = self.reparameterize(mu, log_var, training)
        # Decode to wavelet coefficients
        x_recon, x_haar = self.decode(z, training)

        return x_recon, x_haar, mu, log_var
