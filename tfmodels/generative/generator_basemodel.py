import tensorflow as tf
from ..utilities.basemodel import BaseModel

class BaseGenerator(BaseModel):
    generator_defaults = {
        'gen_kernels': [128, 64, 64, 32],
        'name': 'generator',
        'x_dims': [256, 256, 3],
        'project_shape': None,
        'resize_shape': None,
        # 'z_in': None
    }

    def __init__(self, **kwargs):
        self.generator_defaults.update(**kwargs)
        super(BaseGenerator, self).__init__(**self.generator_defaults)

        if self.project_shape is None:
            self.n_upsamples = len(self.gen_kernels)
            print 'Calculating projection factor for {} upsamples'.format(self.n_upsamples)
            ## calculate the reshape size
            lo_res_size = self.x_dims[0]//(2**self.n_upsamples)
            self.project_shape = (lo_res_size**2) * self.gen_kernels[0]
            self.resize_shape = [-1, lo_res_size, lo_res_size, self.gen_kernels[0]]
        else:
            assert self.project_shape is not None
            assert self.resize_shape is not None
            print 'Using projection shape: {}'.format(self.project_shape)
            print 'Using resize shape: {}'.format(self.resize_shape)


    """ generator.model()

    Inputs:
        z_in:        tensor rank 2: [batch_size, z_dim]
        keep_prob:   tensor rank 0: for dropout
        reuse:       tensor or python bool
    Outputs:
        x_hat:       tensor rank 4: [batch_size, h, w, c]
                        outout of tf.nn.sigmoid
        x_hat_logit: tensor rank 4: [batch_size, h, w, c]
                        before non-linearity
    """
    def model(self, z_in, keep_prob=0.5, reuse=False):
        raise Exception(NotImplementedError)
