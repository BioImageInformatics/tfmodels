import tensorflow as tf
from ..utilities.basemodel import BaseModel

class BaseGenerator(BaseModel):
    generator_defaults = {
        'gen_kernels': [128, 64, 64, 32],
        'name': 'generator',
        'n_upsamples': 4,
        'x_dims': [256, 256, 3],
        # 'z_in': None
    }

    def __init__(self, **kwargs):
        self.generator_defaults.update(**kwargs)
        super(BaseGenerator, self).__init__(**self.generator_defaults)

        # assert self.z_in is not None
        assert len(self.gen_kernels) == self.n_upsamples

        ## calculate the reshape size
        lo_res_size = self.x_dims[0]//(2**self.n_upsamples)
        self.project_shape = (lo_res_size**2)
        self.resize_shape = [lo_res_size, lo_res_size, 1]


    def model(self, z_in, keep_prob=0.5, reuse=False):
        raise Exception(NotImplementedError)
