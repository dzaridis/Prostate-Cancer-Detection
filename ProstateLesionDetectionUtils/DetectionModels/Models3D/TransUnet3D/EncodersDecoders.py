import  tensorflow as tf
from ProstateLesionDetectionUtils.DetectionModels.Models3D import LayerUtils3D


class EncoderBlock(tf.keras.Model):

    def __init__(self, filters, pool_size, **kwargs):
        self.kernel_size = kwargs.get('kernel_size', (3,3,3))
        super(EncoderBlock,self).__init__()
        self.convnet = LayerUtils3D.ConvResModule3D(filters, kernel_size = self.kernel_size)
        self.maxpool = tf.keras.layers.MaxPool3D(pool_size=pool_size, padding='same')

    def call(self, input_tensor):
        lyrs = {}
        cnv = self.convnet(input_tensor)
        mp = self.maxpool(cnv)
        lyrs.update({"residual": cnv, "Downsampling": mp})
        return lyrs


class Bottleneck(tf.keras.Model):

    def __init__(self,
                 patch_sizexy = 4,
                 patch_sizez = 1,
                 d_model = 512,
                 num_heads = 12,
                 mlp_dim = 512,
                 num_layers = 8,
                 dropout_rate=0.1,
                 **kwargs):
        super(Bottleneck, self).__init__(**kwargs)

        self.patch_sizexy = patch_sizexy
        self.patch_sizez = patch_sizez
        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.vit3d = LayerUtils3D.ViT3D(
            inp=input_shape,
            patch_sizexy=self.patch_sizexy,
            patch_sizez=self.patch_sizez,
            d_model=self.d_model,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
        )
        self.reshape = tf.keras.layers.Reshape((input_shape[1]//self.patch_sizez,
                                                input_shape[2]//self.patch_sizexy,
                                                input_shape[3]//self.patch_sizexy,
                                                self.d_model))
        self.cnvtr = tf.keras.layers.Conv3DTranspose(
            filters=self.d_model//2,
            kernel_size=(self.patch_sizez,
                     2*self.patch_sizexy,
                     2*self.patch_sizexy),
            strides=(self.patch_sizez,
                     2*self.patch_sizexy,
                     2*self.patch_sizexy),
            padding="same")
        super(Bottleneck, self).build(input_shape)

    def call(self, input_tensor):

        vit_out = self.vit3d(input_tensor)

        vit_reshaped = self.reshape(vit_out)
        x = self.cnvtr(vit_reshaped)

        return x


class DecoderBlock(tf.keras.Model):

    def __init__(self, filters, up_size, upsample=True, **kwargs):
        self.kernel_size = kwargs.get('kernel_size', (3,3,3))
        super(DecoderBlock,self).__init__()
        self.upsample = upsample
        self.transpose = tf.keras.layers.Conv3DTranspose(
            filters=filters,
            kernel_size = self.kernel_size,
            strides=up_size,
            padding="same")
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.conv2 = LayerUtils3D.ConvResModule3D(filters, kernel_size = self.kernel_size)

    def call(self, input_tensor, residual):
        x = input_tensor
        if self.upsample:
            x = self.transpose(x)
        x = self.concat([x, residual])
        x = self.conv2(x)
        return x


class Classifier(tf.keras.Model):

    def __init__(self, **kwargs):
        super(Classifier,self).__init__(**kwargs)

        self.conv1 = tf.keras.layers.Conv3D(filters=1,
                                            kernel_size=1,
                                            padding='same',
                                            activation="sigmoid")

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        return x

