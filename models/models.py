__env__ = '''Envs]
    Python 3.9.7 64-bit(conda 4.11.0)
    macOS 12.1
'''
__version__ = '''Version]
    version 0.01(beta)
'''
__doc__ = '''\
This module contains various Deep Learning models.
''' + __env__+__version__

from ..layers.layers import CPatches, CPatchEncoder, CRandomSampling, TransformerBlock, CMaskToken, CPosEncoder
from tensorflow.keras.layers import Input, Dense, Add, LayerNormalization, Reshape, Permute, GlobalAveragePooling1D
from tensorflow.keras.models import Model

# Masked Auto Encoder(MAE) model
def mae_model(input_shape, img_h, img_w, patch_size=40, mask_ratio=0.75, d_model=32, d_decoder=32, dff=128, dff_decoder=128, num_heads=4, drop=0.1, N_e=3, N_d=1):
    # --------------- Embeddnig -------------------
    input = Input(shape=input_shape)
    x = CPatches(patch_size) (input)
    num_patches = int(img_h*img_w//(patch_size*patch_size))
    x = CPatchEncoder(num_patches, d_model) (x)
    
    random_sampler = CRandomSampling(num_patches, mask_ratio=mask_ratio)
    x, mask_indices, un_masked_indices = random_sampler (x)

    # ----------------- Encoder -------------------
    for _ in range(N_e):
        x = TransformerBlock(num_heads=num_heads, mlp_dim=dff, dropout=drop) (x) 
    mst = CMaskToken()
    x = mst (x, mask_indices, un_masked_indices )

    # -------------- Positional Embedding -------------------
    x = CPosEncoder(x.shape[1], d_model) (x)  
    x = Dense(units=d_decoder) (x)
    # ----------------- Decoder -------------------
    for _ in range(N_d):
        x = TransformerBlock(num_heads=num_heads, mlp_dim=dff_decoder, dropout=drop) (x)
    x = Dense(units=patch_size*patch_size*3) (x)
    model = Model(inputs=input, outputs=x)
    return model


# MLP-Mixer model
def mlp_mixer_model(sequence_len=14, C=6, d_model = 1024, N=13, Ds=1042, Dc=12):
    input = Input(shape=(sequence_len, ))
    
    # Embedding Layes -----------------------------------
    x = Dense(units=d_model*C, activation='gelu') (input)
    x = Reshape((C, -1))(x)
    x = Dense(units=d_model) (x)
    x = Permute((2, 1)) (x)
    #----------------------------------------------------
    for i in range(N):
        x_in = x
        x = LayerNormalization() (x_in)
        x = Permute((2, 1)) (x)
        # -------------Token Mixing------
        MLP1 = Dense(units=Ds, activation='gelu')
        lin1 = Dense(units=d_model)
    
        x = MLP1 (x)
        x = lin1 (x)
    
        x = Permute((2, 1)) (x)
        x = Add() ([x, x_in])
        # ---------Channel Mixing--------
        x_in = x
        x = LayerNormalization() (x)

        MLP2 = Dense(units=Dc, activation='gelu')
        lin2 = Dense(units=C)
    
        x = MLP2 (x)
        x = lin2 (x)
    
        x = Add() ([x, x_in])
  
    x = Permute((2, 1)) (x)
  
    x = GlobalAveragePooling1D() (x)
    x = Dense(units=45*60*2) (x)
    out = Reshape((45, 60, 2)) (x)

    model = Model(inputs=input, outputs=out)
    return model
