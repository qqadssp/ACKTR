import numpy as np
#import tf_util as U
import tensorflow as tf
import kfac

def dense(x, size, name, weight_init=None, bias_init=0, weight_loss_dict=None, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        assert (len(tf.get_variable_scope().name.split('/')) == 2)

        w = tf.get_variable("w", [x.get_shape()[1], size], initializer=weight_init)
        b = tf.get_variable("b", [size], initializer=tf.constant_initializer(bias_init))
        weight_decay_fc = 3e-4

        if weight_loss_dict is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(w), weight_decay_fc, name='weight_decay_loss')
            if weight_loss_dict is not None:
                weight_loss_dict[w] = weight_decay_fc
                weight_loss_dict[b] = 0.0

            tf.add_to_collection(tf.get_variable_scope().name.split('/')[0] + '_' + 'losses', weight_decay)

        return tf.nn.bias_add(tf.matmul(x, w), b)

def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer

def kl_div(action_dist1, action_dist2, action_size):
    mean1, std1 = action_dist1[:, :action_size], action_dist1[:, action_size:]
    mean2, std2 = action_dist2[:, :action_size], action_dist2[:, action_size:]

    numerator = tf.square(mean1 - mean2) + tf.square(std1) - tf.square(std2)
    denominator = 2 * tf.square(std2) + 1e-8
    return tf.reduce_sum(numerator/denominator + tf.log(std2) - tf.log(std1),reduction_indices=-1)

class GaussianMlpPolicy(object):
    def __init__(self, ob_dim, ac_dim):
        # Here we'll construct a bunch of expressions, which will be used in two places:
        # (1) When sampling actions
        # (2) When computing loss functions, for the policy update
        # Variables specific to (1) have the word "sampled" in them,
        # whereas variables specific to (2) have the word "old" in them
        ob_no = tf.placeholder(tf.float32, shape=[None, ob_dim*2], name="ob") # batch of observations
        oldac_na = tf.placeholder(tf.float32, shape=[None, ac_dim], name="ac") # batch of actions previous actions
        oldac_dist = tf.placeholder(tf.float32, shape=[None, ac_dim*2], name="oldac_dist") # batch of actions previous action distributions
        adv_n = tf.placeholder(tf.float32, shape=[None], name="adv") # advantage function estimate
        wd_dict = {}
        h1 = tf.nn.tanh(dense(ob_no, 64, "h1", weight_init=normc_initializer(1.0), bias_init=0.0, weight_loss_dict=wd_dict))
        h2 = tf.nn.tanh(dense(h1, 64, "h2", weight_init=normc_initializer(1.0), bias_init=0.0, weight_loss_dict=wd_dict))
        mean_na = dense(h2, ac_dim, "mean", weight_init=normc_initializer(0.1), bias_init=0.0, weight_loss_dict=wd_dict) # Mean control output
        self.wd_dict = wd_dict
        self.logstd_1a = logstd_1a = tf.get_variable("logstd", [ac_dim], tf.float32, tf.zeros_initializer()) # Variance on outputs
        logstd_1a = tf.expand_dims(logstd_1a, 0)
        std_1a = tf.exp(logstd_1a)
        std_na = tf.tile(std_1a, [tf.shape(mean_na)[0], 1])
        ac_dist = tf.concat([tf.reshape(mean_na, [-1, ac_dim]), tf.reshape(std_na, [-1, ac_dim])], 1)
        sampled_ac_na = tf.random_normal(tf.shape(ac_dist[:,ac_dim:])) * ac_dist[:,ac_dim:] + ac_dist[:,:ac_dim] # This is the sampled action we'll perform.
        logprobsampled_n = - tf.reduce_sum(tf.log(ac_dist[:,ac_dim:]), axis=1) - 0.5 * tf.log(2.0*np.pi)*ac_dim - 0.5 * tf.reduce_sum(tf.square(ac_dist[:,:ac_dim] - sampled_ac_na) / (tf.square(ac_dist[:,ac_dim:])), axis=1) # Logprob of sampled action
        logprob_n = - tf.reduce_sum(tf.log(ac_dist[:,ac_dim:]), axis=1) - 0.5 * tf.log(2.0*np.pi)*ac_dim - 0.5 * tf.reduce_sum(tf.square(ac_dist[:,:ac_dim] - oldac_na) / (tf.square(ac_dist[:,ac_dim:])), axis=1) # Logprob of previous actions under CURRENT policy (whereas oldlogprob_n is under OLD policy)
        kl = tf.reduce_mean(kl_div(oldac_dist, ac_dist, ac_dim))
        #kl = .5 * tf.reduce_mean(tf.square(logprob_n - oldlogprob_n)) # Approximation of KL divergence between old policy used to generate actions, and new policy used to compute logprob_n
        surr = - tf.reduce_mean(adv_n * logprob_n) # Loss function that we'll differentiate to get the policy gradient
        surr_sampled = - tf.reduce_mean(logprob_n) # Sampled loss of the policy

        def _act(ob):
            sess = tf.get_default_session()
            return sess.run([sampled_ac_na, ac_dist, logprobsampled_n], {ob_no: ob})

        def compute_kl(ob, oldac):
            sess = tf.get_default_session()
            return sess.run(kl, {ob_no: ob, oldac_dist: oldac})

        self._act = _act
        self.compute_kl = compute_kl

        self.update_info = (ob_no, oldac_na, adv_n, surr, surr_sampled) # Input and output variables needed for computing loss
#        U.initialize() # Initialize uninitialized TF variables

    def act(self, ob):
        ac, ac_dist, logp = self._act(ob[None])
        return ac[0], ac_dist[0], logp[0]

class NeuralNetValueFunction(object):
    def __init__(self, ob_dim, ac_dim): #pylint: disable=W0613
        X = tf.placeholder(tf.float32, shape=[None, ob_dim*2+ac_dim*2+1]) # batch of observations
        vtarg_n = tf.placeholder(tf.float32, shape=[None], name='vtarg')
        wd_dict = {}
        h1 = tf.nn.elu(dense(X, 64, "h1", weight_init=normc_initializer(1.0), bias_init=0, weight_loss_dict=wd_dict))
        h2 = tf.nn.elu(dense(h1, 64, "h2", weight_init=normc_initializer(1.0), bias_init=0, weight_loss_dict=wd_dict))
        vpred_n = dense(h2, 1, "hfinal", weight_init=normc_initializer(1.0), bias_init=0, weight_loss_dict=wd_dict)[:,0]
        sample_vpred_n = vpred_n + tf.random_normal(tf.shape(vpred_n))
        self.wd_dict = wd_dict
        wd_loss = tf.get_collection("vf_losses", None)
        loss = tf.reduce_mean(tf.square(vpred_n - vtarg_n)) + tf.add_n(wd_loss)
        loss_sampled = tf.reduce_mean(tf.square(vpred_n - tf.stop_gradient(sample_vpred_n)))

        def _predict(x):
            sess = tf.get_default_session()
            return sess.run(vpred_n, {X: x})

        self._predict = _predict
        self.update_info = (X, vtarg_n, loss, loss_sampled)

    def predict(self, path):
        l = pathlength(path)
        act = path["action_dist"].astype('float32')
        X = np.concatenate([path['observation'], act, np.ones((l, 1))], axis=1)
        return self._predict(X)

def pathlength(path):
    return path["reward"].shape[0]
