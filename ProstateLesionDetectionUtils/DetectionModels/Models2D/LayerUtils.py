import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

class ConvLayer2D(tf.keras.layers.Layer):
    def __init__(self, filters,**kwargs):
        """Convolutional Layer, consists of Conv -> BN -> ReLU

        Args:
            filters (int): number of Conv Filters
        """
        super(ConvLayer2D,self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(
            filters = filters,
            kernel_size = 3,
            padding = 'same',
            activation = None,
            kernel_initializer='he_normal'
        )
        self.bn   = tf.keras.layers.BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))
        self.relu = tf.keras.layers.ReLU()
        self.lyrs = [self.conv, self.bn, self.relu]

    def call(self, input_tensor):

        output_tensor = input_tensor

        for layer in self.lyrs:
            output_tensor = layer(output_tensor)

        return output_tensor

class ConvModule2D(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ConvModule2D,self).__init__(**kwargs)
        self.conv1  = ConvLayer2D(filters)
        self.conv2  = ConvLayer2D(filters)
        self.idcnv = tf.keras.layers.Conv2D(filters, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')
        self.bn    = tf.keras.layers.BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))
        self.act   = tf.keras.layers.ReLU()
        
    def call(self, input_tensor):

        intermediate_tensor = self.conv1(input_tensor)
        conv_op = self.conv2(intermediate_tensor)
        output_tensor = self.act(conv_op)
        return output_tensor

class ConvResModule2D(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ConvResModule2D,self).__init__(**kwargs)
        self.conv1  = ConvLayer2D(filters)
        self.conv2  = ConvLayer2D(filters)
        self.idcnv = tf.keras.layers.Conv2D(filters, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')
        self.bn    = tf.keras.layers.BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))
        self.add   = tf.keras.layers.Add()
        self.act   = tf.keras.layers.ReLU()
        
    def call(self, input_tensor):

        intermediate_tensor = self.conv1(input_tensor)
        conv_op = self.conv2(intermediate_tensor)
        sc = self.idcnv(input_tensor)
        sc = self.bn(sc)
        res = self.add([sc, conv_op])
        output_tensor = self.act(res)
        return output_tensor
    
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(SpatialAttention,self).__init__(**kwargs)
        self.maxpool = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))
        self.avepool = tf.keras.layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))
        self.conc    = tf.keras.layers.Concatenate(axis = 3)
        self.conv    = Conv2D(filters = 1,
        kernel_size=5,
        strides=1,
        padding='same',
        activation='sigmoid',
        kernel_initializer='he_normal',
        use_bias=False)
        self.add     = tf.keras.layers.Add()
        self.mul     = tf.keras.layers.Multiply()
    
    def call(self,input_tensor):

        maxp   = self.maxpool(input_tensor)
        avep   = self.avepool(input_tensor)
        conc   = self.conc([avep,maxp])
        cnv    = self.conv(conc)
        mul    = self.mul([input_tensor,cnv])
        output = self.add([input_tensor, mul])

        return output

class SqueezeAndExcitation(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(SqueezeAndExcitation,self).__init__(**kwargs)
        self.gba     = tf.keras.layers.GlobalAveragePooling2D()
        self.add     = tf.keras.layers.Add()
        self.mul     = tf.keras.layers.Multiply()
    
    def call(self,input_tensor):
        filters   = input_tensor.shape[-1]
        se1_shape = (1,1,filters)
        gba       = self.gba(input_tensor)
        se1 = tf.keras.layers.Reshape(se1_shape)(gba)
        se1 = tf.keras.layers.Dense(filters // 8, activation='relu', kernel_initializer='he_normal', use_bias=False)(se1)
        se1 = tf.keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se1)
        mul = self.mul([input_tensor, se1])
        output = self.add([mul, input_tensor])
        return output

class SaseModule(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SaseModule,self).__init__(**kwargs)
        self.sa = SpatialAttention()
        self.se = SqueezeAndExcitation()
        self.bn    = tf.keras.layers.BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))
        self.act   = tf.keras.layers.ReLU()
        self.add     = tf.keras.layers.Add()
        self.mul     = tf.keras.layers.Multiply()

    def call(self, input_tensor):
        sa_out = self.sa(input_tensor)
        se_out = self.se(input_tensor)
        mul = self.mul([sa_out, se_out])
        bn  = self.bn(mul)
        act = self.act(bn)
        add = self.add([act, input_tensor])
        bn2  = self.bn(add)
        output = self.act(bn2)

        return output

class DecoderSAseModule(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(DecoderSAseModule,self).__init__(**kwargs)
        self.SAse = SaseModule()
        self.ConvRes = ConvResModule2D(filters = filters)
        self.up = tf.keras.layers.UpSampling2D((2,2))
    
    def call(self,input_tensor):
        cnv  = self.ConvRes(input_tensor)
        sase = self.SAse(cnv)
        upsample = self.up(sase)
        return upsample