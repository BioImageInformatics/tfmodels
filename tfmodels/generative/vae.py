import tensorflow as tf
import numpy as np
import sys, os

from basemodel import BaseGenerativeModel

class VAE(BaseGenerativeModel):

    vae_defaults = {

    }

    def __init__(self, **kwargs):
        self.vae_defaults.update(**kwargs)
        super(VAE, self).__init__(**self.vae_defaults)
