__env__ = '''Envs]
    Python 3.9.7 64-bit(conda 4.11.0)
    macOS 12.1
'''
__version__ = '''Version]
    version 0.01(beta)
'''
__doc__ = '''\
This module contains various Deep Learning layers.
''' + __env__+__version__


import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Patch Embedding--------------------------
# https://keras.io/examples/vision/image_classification_with_vision_transformer/
@tf.keras.utils.register_keras_serializable()
class CPatches(layers.Layer):
    '''Split images into the patches.
    '''
    def __init__(self, patch_size, *args, **kwargs):
        super(CPatches, self).__init__(*args, **kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = {"patch_size":self.patch_size}
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Random Sampling-------------------------
@tf.keras.utils.register_keras_serializable()
class CRandomSampling(layers.Layer):    
    '''MAE - Randomly sample the patches except for masked ones.
    '''
    def __init__(self, num_patches, mask_ratio=0.75, *args, **kwargs):
        super(CRandomSampling, self).__init__(*args, **kwargs)
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio

        self.num_mask = int(mask_ratio * num_patches)
        self.un_masked_indices = None
        self.mask_indices = None


    def get_indices(self):
        return [self.mask_indices, self.un_masked_indices]

    def call(self, patches):
        """
            Returns]
            unmasked patches, self.mask_indices, self.un_masked_indices
        """
        self.mask_indices = np.random.choice(self.num_patches, size=self.num_mask,
                            replace=False)
        self.un_masked_indices = np.delete(np.array(range(self.num_patches)), self.mask_indices)

        return tf.gather(patches, self.un_masked_indices, axis=1), self.mask_indices, self.un_masked_indices

    def get_config(self):
        config = {"num_patches":self.num_patches,
                  "mask_ratio":self.mask_ratio}
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Mask Token---------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class CMaskToken(layers.Layer):
    """MAE - Append a mask token to encoder output."""
    def __init__(self, *args, **kwargs):
        super(CMaskToken, self).__init__(*args, **kwargs)
        self.mask_indices = None
        self.un_masked_indices = None
        self.indices = None
        self.mst = None
        self.hidden_size = None

    def build(self, input_shape):
        self.hidden_size = input_shape[-1]
        self.mst = tf.Variable(
            name="mst",
            initial_value = tf.random.normal(
                shape=(1, 1, self.hidden_size), dtype='float32'), 
            trainable=True
        )
        
    def call(self, inputs, mask_indices, un_masked_indices):
        self.mask_indices = mask_indices
        self.un_masked_indices = un_masked_indices
     
        batch_size = tf.shape(inputs)[0]
        mask_num = self.mask_indices.shape[0]
        
        # broadcast mask token for batch
        mst_broadcasted = tf.cast(
                            tf.broadcast_to(self.mst, [batch_size, mask_num, self.hidden_size]),
                            dtype=inputs.dtype,
                        )
        
        # concat
        self.indices = tf.concat([self.mask_indices, self.un_masked_indices], axis=0)
        updates = tf.concat([mst_broadcasted, inputs], axis=1)
                       
        out = tf.gather(updates, self.indices, axis=1, batch_dims=0)
        return out

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Patches to image layer--------------------
@tf.keras.utils.register_keras_serializable()
class CPatchesToImage(layers.Layer):
    '''Reconstruct an image from image patches.
    '''
    def __init__(self, img_h, img_w, img_c, patch_size, *args, **kwargs):
        super(CPatchesToImage, self).__init__(*args, **kwargs)
        self.H = img_h
        self.W = img_w
        self.C = img_c
        self.patch_size = patch_size
        self.n_patch = (self.W * self.H) // (self.patch_size * self.patch_size)
        # assume that the patches can be concatenated without overlapping or padding

    def call(self, inputs):
        patches = [tf.reshape(tensor, (-1, self.patch_size, self.patch_size, self.C)) \
                  for tensor in tf.split(inputs, num_or_size_splits=inputs.shape[1], axis=1)]

        # n_row, n_col should be interger
        n_row = self.H // self.patch_size
        n_col = self.W // self.patch_size

        rows = []
        for row in range(n_row):
            cols = []
            for col in range(n_col):
                cols.append(patches[row*n_col + col])
            col_t = tf.concat(cols, axis=2)  
            rows.append(col_t)
        img = tf.concat(rows, axis=1)

        return img

    def get_config(self):
        config = {"img_h":self.H,
                  "img_w":self.W,
                  "img_c":self.C,
                  "patch_size":self.patch_size}
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Positional Encoding---------------------------------------
# https://www.tensorflow.org/text/tutorials/transformer#positional_encoding

class CPosEncoder(layers.Layer):
    '''Sinusoidal positional encoding.
    '''
    def __init__(self, pos, d_model, *args, **kwargs):
        super(CPosEncoder, self).__init__(*args, **kwargs)
        self.pos = pos
        self.d_model = d_model
        
    def __get_angle(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    
    def call(self, inputs):
        angle_rads = self.__get_angle(np.arange(self.pos)[:, np.newaxis],
                           np.arange(self.d_model)[np.newaxis, :],
                           self.d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return inputs + tf.cast(pos_encoding, dtype=tf.float32)    

    def get_config(self):
        config = {"pos":self.pos,
                  "d_model":self.d_model,
                 }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Patch Encoder--------------------------
# https://keras.io/examples/vision/image_classification_with_vision_transformer/
# edited positional encoding part 
@tf.keras.utils.register_keras_serializable()
class CPatchEncoder(layers.Layer):
    '''Patch encoding. (Linear projection followed by positional encoding)
    '''
    def __init__(self, num_patches, projection_dim, *args, **kwargs):
        super(CPatchEncoder, self).__init__(*args, **kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = CPosEncoder(
            pos=num_patches, d_model=projection_dim
        )

    def call(self, patch):
        encoded = self.position_embedding(self.projection(patch))
        return encoded
        
    def get_config(self):
        config ={"num_patches":self.num_patches,
                 "projection_dim":self.projection_dim}
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# Transformer Block--------------------------------------
# https://github.com/faustomorales/vit-keras/blob/master/vit_keras/layers.py
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    """A Transformer Encoder block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=input_shape[-1] // self.num_heads,  #input_shape[-1] = d_model
            name="MultiHeadDotProductAttention_1", 
          
        )
        self.mlpblock = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.mlp_dim,
                    activation="linear",
                    name=f"{self.name}/Dense_0",
                    
                ),
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.activations.gelu(x, approximate=False),
                  
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1"),
                tf.keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        x = self.att(inputs, inputs)
        x = self.dropout_layer(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        x = x + y
        x = self.layernorm1(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Downsample layer for UNet Encoder(pix2pix)
class CDownsample(layers.Layer):
    def __init__(self, filter_num, kernel_size, stride, batch_norm=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initializer = None
        self.downsample = None
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.stride = stride
        self.batch_norm = batch_norm
        self.input_shape = None

    def build(self, input_shape):
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.downsample = tf.keras.Sequential()
        self.downsample.add(
            tf.keras.layers.Conv2D(self.filter_num, self.kernel_size, self.stride, padding='same',
                        kernel_initializer=self.initializer, use_bias=False)
        )
        if self.batch_norm:
            self.downsample.add(tf.keras.layers.BatchNormalization())
        self.downsample.add(tf.keras.layers.LeakyReLU())
        self.input_shape = input_shape

    def call(self, input, training):
        return self.downsample(input, training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filter_num':self.filter_num,
            'kernel_size':self.kernel_size,
            'stride':self.stride,
            'batch_norm':self.batch_norm,
        })

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Upsample layer for UNet Decoder(pix2pix)
class CUpsample(layers.Layer):
    def __init__(self, filter_num, kernel_size, stride, dropout=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initializer = None
        self.upsample = None
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout = dropout
        self.input_shape = None

    def build(self, input_shape):
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.upsample = tf.keras.Sequential()
        self.upsample.add(
            tf.keras.layers.Conv2DTranspose(self.filter_num, self.kernel_size, self.stride, padding='same',
                        kernel_initializer=self.initializer, use_bias=False)
        )
        self.upsample.add(tf.keras.layers.BatchNormalization())
        
        if self.dropout:
            self.upsample.add(tf.keras.layers.Dropout(0.5))
        self.upsample.add(tf.keras.layers.ReLU())
        self.input_shape = input_shape

    def call(self, input, training):
        return self.upsample(input, training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filter_num':self.filter_num,
            'kernel_size':self.kernel_size,
            'stride':self.stride,
            'dropout':self.dropout,
        })

    @classmethod
    def from_config(cls, config):
        return cls(**config)

