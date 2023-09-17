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

    def __init__(self, filters, **kwargs):
        self.kernel_size =  kwargs.get('kernel_size', (3,3,3))
        super(Bottleneck,self).__init__()
        self.convnet = LayerUtils3D.ConvResModule3D(filters, kernel_size = self.kernel_size)

    def call(self, input_tensor):
        cnv = self.convnet(input_tensor)
        return cnv


class DecoderBlock(tf.keras.Model):

    def __init__(self, filters, up_size, **kwargs):
        self.kernel_size = kwargs.get('kernel_size', (3,3,3))
        super(DecoderBlock,self).__init__()
        self.transpose = tf.keras.layers.Conv3DTranspose(
            filters=filters,
            kernel_size = self.kernel_size,
            strides=up_size,
            padding="same")
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.conv2 = LayerUtils3D.ConvResModule3D(filters, kernel_size = self.kernel_size)

    def call(self, input_tensor, residual):
        x = self.transpose(input_tensor)
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

