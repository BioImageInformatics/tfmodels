import tensorflow as tf
from ..utilities.basemodel import BaseModel

class BaseDiscriminator(BaseModel):
    ## Overload the base class.. do I even need the base class?
    ## TODO expose number of kernels and number of upsample steps to the world
    discriminator_defaults = {
        'dis_kernels': [32, 64, 128],
        'adversary_feature_matching': False,
        'name': 'discriminator',
        'soften_labels': False,
        'soften_sddev': 0.01,
    }

    def __init__(self, **kwargs):
        self.discriminator_defaults.update(**kwargs)
        super(BaseDiscriminator, self).__init__(**self.discriminator_defaults)

        # assert self.real is not None
        # assert self.fake is not None

        self.p_real_fake = None
        self.p_real_real = None

        ## Set up discriminator and inputs


    def model(self, x_in, keep_prob=0.5, reuse=False):
        raise Exception(NotImplementedError)


    ## TODO switch to Wasserstein loss. Remember to clip the outputs
    ## Just put this into the model def since so many things are going to change
    ## Can't put these into the __init__ method because we have to have the
    ## model defined, and we could also change the loss function later.
    ## these are defaults for now
    # def make_loss(self, p_real_fake, p_real_real):
    #     real_target = tf.ones_like(p_real_real)
    #     fake_target = tf.zeros_like(p_real_fake)
    #
    #     if self.soften_labels:
    #         real_epsilon = tf.random_normal(shape=tf.shape(real_target),
    #             mean=0.0, stddev=self.soften_sddev)
    #         fake_epsilon = tf.random_normal(shape=tf.shape(fake_target),
    #             mean=0.0, stddev=self.soften_sddev)
    #         real_target = real_target + real_epsilon
    #         fake_target = fake_target + fake_epsilon
    #
    #     loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #         labels=real_target, logits=p_real_real))
    #     loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #         labels=fake_target, logits=p_real_fake))
    #     # return (loss_real + loss_fake) / 2.0
    #     return loss_real + loss_fake
    #
    # def make_training_op(self, p_real_fake, p_real_real):
    #     self.var_list = self.get_update_list()
    #     self.optimizer = tf.train.AdamOptimizer(self.learning_rate,
    #         name='{}_Adam'.format(self.name))
    #
    #     self.loss = self.make_loss(p_real_fake, p_real_real)
    #     self.train_op = self.optimizer.minimize(self.loss,
    #         var_list=self.var_list)
    #     self.training_op_list.append(self.train_op)
    #
    #     # Summary
    #     self.disciminator_loss_sum = tf.summary.scalar('{}_loss'.format(self.name),
    #         self.loss)
    #     self.summary_op_list.append(self.disciminator_loss_sum)
