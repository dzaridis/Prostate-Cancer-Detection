import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from ProstateLesionDetectionUtils.DetectionModels.Models3D import LayerUtils3D
import tensorflow_addons as tfa
import time
from ProstateLesionDetectionUtils.DetectionModels.Models3D.VNet.EncodersDecoders import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import time
from ProstateLesionDetectionUtils.DetectionModels.Models3D.VNet.EncodersDecoders import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class VnetMod(tf.keras.Model):
    def __init__(self, num_filters, pool_size, **kwargs):
        self.kernel_size = kwargs.get('kernel_size', [3 for _ in range(len(num_filters))])
        super(VnetMod, self).__init__()
        self.enc_parts = []
        self.dec_parts = []
        self.num_filters = num_filters
        self.inverse = num_filters[::-1]
        self.inv_kernels = self.kernel_size[::-1]
        unpool_size = pool_size[::-1]

        for i in range(len(num_filters) - 1):
            self.enc_parts.append(EncoderBlock(num_filters[i], pool_size[i], kernel_size = self.kernel_size[i]))

        self.btlneck = Bottleneck(num_filters[-1], kernel_size = self.kernel_size[-1])

        for i in range(len(num_filters) - 1):
            self.dec_parts.append(DecoderBlock(self.inverse[i + 1], unpool_size[i], kernel_size = self.inv_kernels[i+1]))

        self.clf = Classifier()

    def call(self, input_tensor):
        x = input_tensor
        res = []

        for lyr in self.enc_parts:
            dic_lyrs = lyr(x)
            res.append(dic_lyrs["residual"])
            x = dic_lyrs["Downsampling"]

        x = self.btlneck(x)

        for i, lyr in enumerate(self.dec_parts):
            x = lyr(x, res[-i - 1])

        x = self.clf(x)

        return x


class TrainVnet:

    def __init__(self, params):
        """Constructor for Model Training

        Args:
            Params (dict): dictionary that contains the parameters for training
        """
        self.params = params
        self.model = None
        self.history = None
    def ModelBuild(self):
        """
        Create unet model keras instance based
        """
        # self.Params["INPUT_SIZE"], self.Params["VOLUME_SIZE"],
        self.model = VnetMod(self.params["FILTERS"], self.params["POOL_SIZE"], kernel_size = self.params["kernel_size"])
        self.model.build(input_shape=self.params["INPUT_SIZE"])

    def ModelCompile(self):
        """
        Compile based on the parameters dictionary
        """
        self.model.compile(loss=self.params["LOSS"],
                           # , #SigmoidFocalCrossEntropy, sigmoid_focal_crossentropy, surface_loss_keras
                           optimizer=self.params["OPTIMIZER"],  # lr_schedule
                           metrics=self.params["METRICS"], run_eagerly=True)

    def ModelFit(self):
        """
        Model Training
        """
        Start = time.time()
        self.history = self.model.fit(
            x=self.params["TRAIN_DATASET"]["DATA"],
            y=self.params["TRAIN_DATASET"]["LABELS"],
            validation_data = (self.params["VAL_DATASET"]["DATA"],self.params["VAL_DATASET"]["LABELS"]),
            validation_steps = 5,
            batch_size = self.params["BATCH_SIZE"],
            epochs=self.params["EPOCHS"],
            verbose=1,
            callbacks=self.params["CALLBACKS"])
        end = (time.time() - Start) / 60
        print("Time needed for training:",end)

    def SaveInitWeights(self):
        self.model.save_weights(self.params["INITIAL_WEIGHTS"])
        
    def LoadTrained(self):
        self.model.load_weights(self.params["INITIAL_WEIGHTS"])
    
    def GetHistory(self):
        return self.history

    def GetModel(self):
        return self.model