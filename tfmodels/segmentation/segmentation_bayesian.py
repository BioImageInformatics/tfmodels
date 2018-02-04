import tensorflow as tf
import numpy as np
import sys, os

from segmentation_basemodel import Segmentation

class SegmentationBayesian(Segmentation):
    ## Defaults. arXiv links correspond to inspirational materials
    bayesian_segmentation_defaults={
        'class_weights': None, ## https://arxiv.org/abs/1511.00561
        'dataset': None,
        'aleatoric': False, ## https://arxiv.org/abs/1703.04977
        'aleatoric_T': 50,
        'global_step': 0,
        'k_size': 3,
        'learning_rate': 1e-3,
        'log_dir': None,
        'mode': 'TRAIN',
        'name': 'Segmentation',
        'n_classes': None,
        'save_dir': None,
        'sess': None,
        'seg_training_op_list': [],
        'summarize_grads': False,
        'summary_iters': 50,
        'summary_image_iters': 250,
        'summary_op_list': [],
        'x_dims': [256, 256, 3],
     }

    def __init__(self, **kwargs):
        self.bayesian_segmentation_defaults.update(**kwargs)

        super(SegmentationBayesian, self).__init__(**self.bayesian_segmentation_defaults)
        assert self.sess is not None

        if self.mode=='TRAIN':
            self._training_mode()
        elif self.mode=='TEST':
            self._test_mode()

    def _training_mode(self):
        print 'Setting up {} in training mode'.format(self.name)
        ## ------------------- Input ops ------------------- ##
        self.x_in = tf.placeholder_with_default(self.dataset.image_op,
            shape=[None, self.x_dims[0], self.x_dims[1], self.x_dims[2]],
                name='x_in')
        self.y_in = tf.placeholder_with_default(self.dataset.mask_op,
            shape=[None, self.x_dims[0], self.x_dims[1], 1], name='y_in')

        ## ------------------- Model ops ------------------- ##
        # self.keep_prob = tf.placeholder('float', name='keep_prob')
        self.keep_prob = tf.placeholder_with_default(0.5, shape=(), name='keep_prob')
        self.y_hat, self.sigma = self.model(self.x_in, keep_prob=self.keep_prob, reuse=False)

        self.y_hat_smax = tf.nn.softmax(self.y_hat)
        self.y_hat_mask = tf.expand_dims(tf.argmax(self.y_hat, -1), -1)
        self.y_hat_mask = tf.cast(self.y_hat_mask, tf.float32)
        print 'Model output y_hat:', self.y_hat.get_shape()

        ## ------------------- Training ops ------------------- ##
        self.var_list = self._get_update_list()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate,
            name='{}_Adam'.format(self.name))

        self._make_training_ops()

        ## ------------------- Gather Summary ops ------------------- ##
        self._make_summaries()

        ## ------------------- TensorFlow helpers ------------------- ##
        self._tf_ops()
        self.sess.run(tf.global_variables_initializer())

        self._print_info_to_file(filename=os.path.join(self.save_dir,
            '{}_settings.txt'.format(self.name)))

        if self.adversary:
            self.discriminator._print_info_to_file(filename=os.path.join(self.save_dir,
                '{}_settings.txt'.format(self.discriminator.name)))

        ## Somehow calling this during init() makes it not work
        # if self.adversary and self.pretraining_iters:
        #     self.pretrain()

    def _test_mode(self):
        print 'Setting up {} in inference mode'.format(self.name)
        ## ------------------- Input ops ------------------- ##
        self.x_in = tf.placeholder('float',
            shape=[None, self.x_dims[0], self.x_dims[1], self.x_dims[2]],
            name='x_in')

        ## ------------------- Model ops ------------------- ##
        # self.keep_prob = tf.placeholder('float', name='keep_prob')
        self.keep_prob = tf.placeholder_with_default(0.5, shape=[], name='keep_prob')
        self.y_hat = self.model(self.x_in, keep_prob=self.keep_prob, reuse=False)
        self.y_hat_smax = tf.nn.softmax(self.y_hat)

        self.saver = tf.train.Saver(max_to_keep=5)

        self.sess.run(tf.global_variables_initializer())

    
    def _heteroscedastic_aleatoric_loss(self):
        assert self.sigma is not None, 'Model does not have attribute sigma. Define sigma in model()'

        print 'Setting up heteroscedastic aleatoric loss:'
        ## Make a summary for sigma
        self.sigma_summary = tf.summary.histogram('sigma', self.sigma)
        self.summary_op_list.append(self.sigma_summary)

        with tf.variable_scope('aleatoric') as scope:
            ## (batch_size, h*w, n_classes)
            sigma_v = tf.reshape(self.sigma, [-1, np.prod(self.x_dims[:2]), 1])
            dist = tf.distributions.Normal(loc=0.0, scale=sigma_v, name='dist')
            y_hat_v = tf.reshape(self.y_hat, [-1, np.prod(self.x_dims[:2]), self.n_classes])
            y_in_v = tf.reshape(self.y_in, [-1, np.prod(self.x_dims[:2]), self.n_classes])

            print '\t sigma_v', sigma_v.get_shape()
            print '\t y_hat_v', y_hat_v.get_shape()
            print '\t y_in_v', y_in_v.get_shape()

            def _corrupt_with_noise(yhat, dist):
                ## sample is the same shape as the sigma output
                ## LOL
                epsilon = tf.transpose(dist.sample(self.n_classes), perm=[1,2,0,3])
                epsilon = tf.squeeze(epsilon)
                return yhat + epsilon

            def loss_fn(yhat):
                y_hat_eps = _corrupt_with_noise(yhat, dist)
                print '\t y_hat_eps', y_hat_eps.get_shape()

                loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_in_v, logits=y_hat_eps)
                print '\t loss', loss.get_shape()

                # y_hat_c = tf.reduce_sum(y_hat_eps * y_in_v, axis=-1, keep_dims=True)
                # print 'y_hat_c', y_hat_c.get_shape()
                #
                # y_hat_c = tf.tile(y_hat_c, [1, 1, self.n_classes])
                # print 'y_hat_c', y_hat_c.get_shape()

                # return tf.reduce_logsumexp(y_hat_eps - y_hat_c, 1, name='delta')
                return loss

            y_hat_tile = tf.tile(tf.expand_dims(y_hat_v, 0), [self.epistemic_T,1,1,1], name='y_hat_tile')
            print 'y_hat_tile', y_hat_tile.get_shape()

            loss_fn_map = lambda x: loss_fn(x)
            losses = tf.map_fn(loss_fn_map, y_hat_tile)
            print 'losses', losses.get_shape()

            ## Sum over classes
            # loss = tf.reduce_sum(losses, axis=-1)
            # print 'loss', loss.get_shape()

            ## Average over pixels
            # loss = tf.reduce_mean(loss, axis=-1)

            ## Average over batch, pixels and T
            self.seg_loss = tf.reduce_mean(losses)


    def _make_training_ops(self):
        with tf.name_scope('segmentation_losses'):
            ## Logic to define the correct version of segmentation loss
            ## BUG if aleatoric is requested, the constructor will ignore
            ##  the request for weighted classes
            if self.aleatoric:
                self._heteroscedastic_aleatoric_loss()
            else:
                self._segmentation_loss()

            ## Unused except in pretraining or specificially requested
            self.seg_training_op = self.optimizer.minimize(
                self.seg_loss, var_list=self.var_list, name='{}_seg_train'.format(self.name))

            self.seg_loss_sum = tf.summary.scalar('seg_loss', self.seg_loss)
            self.summary_op_list.append(self.seg_loss_sum)

            self.loss = self.seg_loss

            self.train_op = self.optimizer.minimize(
                self.loss, var_list=self.var_list, name='{}_train'.format(self.name))

            self.seg_training_op_list.append(self.train_op)


    """ function for approximate bayesian inference via dropout

    if ret_all is true, then we return the mean and variance
    if ret_all is false, then we return just the mean, similar to returning
        a single y_hat_
    """
    def bayesian_inference(self, x_in, samples=25, keep_prob=0.5, ret_all=False):
        assert keep_prob < 1.0
        assert samples > 1
        assert x_in.shape[0] == 1 and len(x_in.shape) == 4

        ## Option A:
        x_in_stack = np.concatenate([x_in]*samples, axis=0)

        y_hat = self.inference(x_in=x_in_stack, keep_prob=keep_prob)
        y_bar_mean = np.mean(y_hat, axis=0) ## (1, h, w, n_classes)

        ## Option B:
        # y_hat = [self.inference(x_in=x_in, keep_prob=keep_prob) for _ in xrange(samples)]
        # y_hat = np.stack(y_hat, axis=0)
        # y_bar_mean = np.mean(y_hat, axis=0)

        if ret_all:
            y_bar_var = np.var(y_hat, axis=0)
            # y_bar_argmax = np.argmax(y_bar_mean, axis=-1)
            # y_bar_argmax = np.expand_dims(y_bar_argmax, 0)
            return y_bar_mean, y_bar_var
        else:
            return y_bar_mean