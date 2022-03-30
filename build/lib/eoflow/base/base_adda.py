from sklearn.utils import shuffle
import numpy as np

from .base_custom_training import BaseModelCustomTraining
from .base_dann import BaseModelAdapt

class BaseModelAdapt(BaseModelAdapt):
    def __init__(self, config_specs):
        BaseModelCustomTraining.__init__(self, config_specs)
        self.eps = 1e-8

    def train_target_discriminator(self, x_source, x_target,
                                   model_directory,num_epochs =  500):
        #Initialize models with inputs
        source_encoder = self._get_encoder(x_source, model_directory)
        for layer in self.source_encoder.layers:
            layer.trainable = False
        target_encoder = self._get_encoder(x_target)
        discriminator = self._get_discriminator(x_source)
        #Then,during training, discriminator will see encoded source and target and must classify it

        y_source = np.zeros_like(x_source.shape[0])
        y_target = np.ones_like(x_target.shape[0])








