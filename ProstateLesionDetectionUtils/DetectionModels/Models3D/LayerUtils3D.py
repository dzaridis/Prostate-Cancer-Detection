import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv3D, BatchNormalization, ReLU, Dropout, Concatenate, Add, Multiply, UpSampling3D

class ConvLayer3D(tf.keras.layers.Layer):
    def __init__(self, filters,**kwargs):
        """Convolutional Layer, consists of Conv -> BN -> ReLU

        Args:
            filters (int): number of Conv Filters
        """
        self.kernel_size = kwargs.get('kernel_size', (3,3,3))
        super(ConvLayer3D,self).__init__()
        self.conv = Conv3D(
            filters = filters,
            kernel_size = self.kernel_size,
            padding = 'same',
            activation = None,
            kernel_initializer='he_normal'
        )
        self.bn   = BatchNormalization(axis=-1, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))
        self.relu = ReLU()
        self.lyrs = [self.conv, self.bn, self.relu]

    def call(self, input_tensor):

        output_tensor = input_tensor

        for layer in self.lyrs:
            output_tensor = layer(output_tensor)

        return output_tensor
    
class DenseCNNlayer3D(tf.keras.layers.Layer):
    def __init__(self,filters, GrowthRate, DropOut,**kwargs):
        """Construction for the DenseCNNlayer3D
        Args:
            filters (int): Number of CNN filters 
            GrowthRate (int): growth rate for the Dense layers
            DropOut (float): dropout ratio 0 to 1
        """
        self.kernel_size = kwargs.get('kernel_size', (3,3,3))
        super(DenseCNNlayer3D,self).__init__()
        self.GrowthRate = GrowthRate
        self.layrs = []
        for _ in range(GrowthRate):
            self.layrs.append([Conv3D(
        filters = filters,
        kernel_size =  self.kernel_size,
        padding = 'same',
        activation = None,
        kernel_initializer='he_normal'
    ),
                BatchNormalization(axis=-1, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4)),
                ReLU(),
                Dropout(DropOut),
                Concatenate(axis = -1)])

    def call(self, input_tensor):
        x = input_tensor
        for lyr in self.layrs:
            cnv = lyr[0](x)
            cnv = lyr[1](cnv)
            cnv = lyr[2](cnv)
            cnv = lyr[3](cnv)
            x = lyr[4]([x,cnv])

        return x

class TransitionLayer3D(tf.keras.layers.Layer):
    def __init__(self, filters, DropOut, **kwargs):
        """Constructor for the transition layer

        Args:
            filters (int): number of CNN filters to Homogenize the output of the DenseCNNlayer3D
            DropOut (float): dropout ratio 0 to 1
        """
        super(TransitionLayer3D, self).__init__(**kwargs)
        self.conv = Conv3D(filters = filters,kernel_size = 1, padding='same',kernel_initializer='he_normal')
        self.bn   = BatchNormalization(axis=-1, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))
        self.relu = ReLU()
        self.Drop = Dropout(DropOut)

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.bn(x)
        x = self.relu(x)
        x = self.Drop(x)
        return x

class DenseCNNModule3D(tf.keras.layers.Layer):
    def __init__(self,FilterDense,FilterTrans, GrowthRate,DropOut,NumDenseBlocks = 2,**kwargs):
        """
        DenseBlock3DCNN
        Args:
            FilterDense (int): Number of filters for the Dense Layer
            FilterTrans (int): number of CNN filters to Homogenize the output of the DenseCNNlayer3D
            GrowthRate (int): growth rate for the Dense layers
            DropOut (float): dropout ratio 0 to 1
            NumDenseBlocks (int, optional): Number of DenseBlocks. Defaults to 2.
        """
        self.kernel_size = kwargs.get('kernel_size', (3,3,3))
        super(DenseCNNModule3D, self).__init__()
        self.dns = []
        self.NumDenseBlocks = NumDenseBlocks
        for _ in range(NumDenseBlocks):
            self.dns.append([DenseCNNlayer3D(FilterDense,GrowthRate,DropOut,kernel_size= self.kernel_size),TransitionLayer3D(FilterTrans,DropOut)])

    def call(self, input_tensor):
        x = input_tensor
        for dbl in self.dns:
            x = dbl[0](x)
            x = dbl[1](x)
        return x


class AttentionGate3D(tf.keras.layers.Layer):
    def __init__(self, filters, unpool_size):
        super(AttentionGate3D, self).__init__()
        self.filters = filters

        self.theta = tf.keras.layers.Conv3D(filters, 1, activation='relu', kernel_initializer='he_normal')
        self.phi = tf.keras.layers.Conv3D(filters, 1, activation='relu', kernel_initializer='he_normal')
        self.psi = tf.keras.layers.Conv3D(1, 1, activation='sigmoid', kernel_initializer='he_normal')
        self.upsample = tf.keras.layers.UpSampling3D(size=unpool_size)

    def call(self, input_tensor, skip):
        theta = self.theta(skip)
        phi = self.upsample(self.phi(input_tensor))

        attn = tf.keras.activations.softmax(theta * phi, axis=[1, 2, 3])

        psi = self.psi(attn * skip)

        return psi


class ConvModule3D(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        """Convolutional Layer, consists of Conv -> BN -> ReLU -> Conv -> BN -> ReLU
        Args:
            filters (int): number of Conv Filters
        """
        self.kernel_size = kwargs.get('kernel_size', (3,3,3))
        super(ConvModule3D,self).__init__()
        self.conv1  = ConvLayer3D(filters, kernel_size = self.kernel_size)
        self.conv2  = ConvLayer3D(filters, kernel_size = self.kernel_size)
        self.bn    = BatchNormalization(axis=-1, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))
        self.act   = ReLU()
        
    def call(self, input_tensor):

        intermediate_tensor = self.conv1(input_tensor)
        conv_op = self.conv2(intermediate_tensor)
        output_tensor = self.act(conv_op)
        return output_tensor

class ConvResModule3D(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        self.kernel_size = kwargs.get('kernel_size', (3,3,3))
        super(ConvResModule3D,self).__init__()
        self.conv1  = ConvLayer3D(filters, kernel_size = self.kernel_size)
        self.conv2  = ConvLayer3D(filters, kernel_size = self.kernel_size)
        self.idcnv = Conv3D(filters, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')
        self.bn    = BatchNormalization(axis=-1, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))
        self.add   = Add()
        self.act   = ReLU()
        
    def call(self, input_tensor):

        intermediate_tensor = self.conv1(input_tensor)
        conv_op = self.conv2(intermediate_tensor)
        sc = self.idcnv(input_tensor)
        sc = self.bn(sc)
        res = self.add([sc, conv_op])
        output_tensor = self.act(res)
        return output_tensor
    
# class SpatialAttention3D(tf.keras.layers.Layer):
#     def __init__(self,**kwargs):
#         super(SpatialAttention3D,self).__init__(**kwargs)
#         self.maxpool = Lambda(lambda x: K.mean(x, axis=4, keepdims=True))
#         self.avepool = Lambda(lambda x: K.max(x, axis=4, keepdims=True))
#         self.conc    = Concatenate(axis = 3)
#         self.conv    = Conv3D(filters = 1,
#         kernel_size=5,
#         strides=1,
#         padding='same',
#         activation='sigmoid',
#         kernel_initializer='he_normal',
#         use_bias=False)
#         self.add     = Add()
#         self.mul     = Multiply()
    
#     def call(self,input_tensor):

#         maxp   = self.maxpool(input_tensor)
#         avep   = self.avepool(input_tensor)
#         conc   = self.conc([avep,maxp])
#         cnv    = self.conv(conc)
#         mul    = self.mul([input_tensor,cnv])
#         output = self.add([input_tensor, mul])

#         return output
class SpatialAttention3D(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttention3D, self).__init__()
        self.mul     = Multiply()
        self.conv    = Conv3D(filters = 1,
        kernel_size=5,
        strides=1,
        padding='same',
        activation='sigmoid',
        kernel_initializer='he_normal',
        use_bias=False)
        self.activation = tf.keras.layers.Activation('sigmoid')
    
    def call(self, input_tensor):
        avg_pool = tf.math.reduce_mean(input_tensor, axis=(1,2,3), keepdims=True)
        max_pool = tf.math.reduce_max(input_tensor, axis=(1,2,3), keepdims=True)
        concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
        x = self.conv1(concat)
        x = self.activation(x)
        x = self.mul([input_tensor, x])
        return x


class SqueezeAndExcitation3D(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(SqueezeAndExcitation3D, self).__init__()
        self.filters = filters
        self.reshape = tf.keras.layers.Reshape([-1, 1, 1, self.filters])
        self.global_pool = tf.keras.layers.GlobalAveragePooling3D()
        self.fc1 = tf.keras.layers.Dense(units=self.filters // 8, activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=self.filters, activation='sigmoid')
        self.mul = tf.keras.layers.Multiply()
    def call(self, input_tensor):
        x = self.global_pool(input_tensor)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshape(x)
        x = self.mul([input_tensor,x])
        return x


class SaseModule3D(tf.keras.layers.Layer):
    def __init__(self,filters, **kwargs):
        super(SaseModule3D,self).__init__(**kwargs)
        self.sa = SpatialAttention3D()
        self.se = SqueezeAndExcitation3D(filters)
        self.bn    = BatchNormalization(axis=-1, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))
        self.act   = ReLU()
        self.add     = Add()
        self.mul     = Multiply()

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


class PatchEmbeddings(tf.keras.layers.Layer):
    def __init__(self, embed_dim, patch_size=(1, 4, 4)):
        """
        Args:
            embed_dim [int]: dimensions of embeddings
            patch_size [tuple]:
        """
        super(PatchEmbeddings, self).__init__()
        self.projection = tf.keras.layers.Conv3D(
            filters=embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid', activation = None,use_bias=False
        )
        self.flatten = tf.keras.layers.Reshape((-1, embed_dim))

    def call(self, input_tensor):
        projected_patches = self.projection(input_tensor)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches


class PositionalEncoder(tf.keras.layers.Layer):
    def __init__(self,embed_dim):
        super(PositionalEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.positions = None

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = tf.keras.layers.Embedding(input_dim =num_tokens,
                                                            output_dim = self.embed_dim)
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)
        super(PositionalEncoder, self).build(input_shape)

    def call(self, encoded_tokens):
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, mlp_dim, dropout_rate=0.1):
        super(TransformerLayer, self).__init__()

        self.attention = tf.keras.layers.MultiHeadAttention(num_heads, d_model)

        self.ffn1 = tf.keras.layers.Dense(mlp_dim, activation=tf.nn.gelu)
        self.ffn2 = tf.keras.layers.Dropout(dropout_rate)
        self.ffn3 = tf.keras.layers.Dense(d_model)

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output)
        x1 = self.layer_norm1(inputs + attention_output)

        ffn_output = self.ffn3(self.ffn2(self.ffn1(x1)))

        return ffn_output


class ViT3D(tf.keras.Model):
    def __init__(self, inp=[1,24,192,192,1],
                 patch_sizexy= 16,
                 patch_sizez= 4,
                 d_model=512,
                 num_heads=12,
                 mlp_dim=512,
                 num_layers=8,
                 dropout_rate=0.1):
        """

        Args:
            inp [list]:  Input shape, Defaults at [1,24,192,192,1]
            patch_sizexy [int]: patch size for x and y (spatial dimensions)
            patch_sizez [int]: patch size for z axis (spatial dimensions)
            d_model [int]: embedded dims
            num_heads [int]: Number of Transformer heads
            mlp_dim [int]: MLP Units
            num_layers [int]: Number of transformers layers
            dropout_rate:  Dropout rate for the MLP

        Returns:
            object:
        """
        super(ViT3D, self).__init__()

        self.inp = inp
        self.d_model = d_model
        self.patch_embed = PatchEmbeddings(embed_dim=d_model,
                                           patch_size=(patch_sizez, patch_sizexy, patch_sizexy))
        self.pos_enc = PositionalEncoder(embed_dim=d_model)
        # Transformer layers
        self.transformer_layers = [
            TransformerLayer(d_model, num_heads, mlp_dim, dropout_rate) for _ in range(num_layers)
        ]
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):

        # Embeddings
        x = self.patch_embed(inputs)
        embeddings = self.pos_enc(x)

        # Transformer layers
        for transformer_layer in self.transformer_layers:
            embeddings = transformer_layer(embeddings) + embeddings
        embeddings = self.layer_norm(embeddings)
        return embeddings
