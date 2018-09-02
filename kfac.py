import tensorflow as tf
import numpy as np
import re
from functools import reduce

KFAC_OPS = ['MatMul', 'BiasAdd']
KFAC_DEBUG = False
KFAC_DEBUG_ = True

def gmatmul(a, b, transpose_a=False, transpose_b=False, reduce_dim=None):
    assert reduce_dim is not None

    # weird batch matmul
    if len(a.get_shape()) == 2 and len(b.get_shape()) > 2:
        # reshape reduce_dim to the left most dim in b
        b_shape = b.get_shape()
        if reduce_dim != 0:
            b_dims = list(range(len(b_shape)))
            b_dims.remove(reduce_dim)
            b_dims.insert(0, reduce_dim)
            b = tf.transpose(b, b_dims)
        b_t_shape = b.get_shape()
        b = tf.reshape(b, [int(b_shape[reduce_dim]), -1])
        result = tf.matmul(a, b, transpose_a=transpose_a,
                           transpose_b=transpose_b)
        result = tf.reshape(result, b_t_shape)
        if reduce_dim != 0:
            b_dims = list(range(len(b_shape)))
            b_dims.remove(0)
            b_dims.insert(reduce_dim, 0)
            result = tf.transpose(result, b_dims)
        return result

    elif len(a.get_shape()) > 2 and len(b.get_shape()) == 2:
        # reshape reduce_dim to the right most dim in a
        a_shape = a.get_shape()
        outter_dim = len(a_shape) - 1
        reduce_dim = len(a_shape) - reduce_dim - 1
        if reduce_dim != outter_dim:
            a_dims = list(range(len(a_shape)))
            a_dims.remove(reduce_dim)
            a_dims.insert(outter_dim, reduce_dim)
            a = tf.transpose(a, a_dims)
        a_t_shape = a.get_shape()
        a = tf.reshape(a, [-1, int(a_shape[reduce_dim])])
        result = tf.matmul(a, b, transpose_a=transpose_a,
                           transpose_b=transpose_b)
        result = tf.reshape(result, a_t_shape)
        if reduce_dim != outter_dim:
            a_dims = list(range(len(a_shape)))
            a_dims.remove(outter_dim)
            a_dims.insert(reduce_dim, outter_dim)
            result = tf.transpose(result, a_dims)
        return result

    elif len(a.get_shape()) == 2 and len(b.get_shape()) == 2:
        return tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)

    assert False, 'something went wrong'


def clipoutNeg(vec, threshold=1e-6):
    mask = tf.cast(vec > threshold, tf.float32)
    return mask * vec


def detectMinVal(input_mat, var, threshold=1e-6, name='', debug=False):
    eigen_min = tf.reduce_min(input_mat)
    eigen_max = tf.reduce_max(input_mat)
    eigen_ratio = eigen_max / eigen_min
    input_mat_clipped = clipoutNeg(input_mat, threshold)

    if debug:
        input_mat_clipped = tf.cond(tf.logical_or(tf.greater(eigen_ratio, 0.), tf.less(eigen_ratio, -500)), lambda: input_mat_clipped, lambda: tf.Print(
            input_mat_clipped, [tf.convert_to_tensor('screwed ratio ' + name + ' eigen values!!!'), tf.convert_to_tensor(var.name), eigen_min, eigen_max, eigen_ratio]))

    return input_mat_clipped


def factorReshape(Q, e, grad, facIndx=0, ftype='act'):
    grad_shape = grad.get_shape()
    if ftype == 'act':
        assert e.get_shape()[0] == grad_shape[facIndx]
        expanded_shape = [1, ] * len(grad_shape)
        expanded_shape[facIndx] = -1
        e = tf.reshape(e, expanded_shape)
    if ftype == 'grad':
        assert e.get_shape()[0] == grad_shape[len(grad_shape) - facIndx - 1]
        expanded_shape = [1, ] * len(grad_shape)
        expanded_shape[len(grad_shape) - facIndx - 1] = -1
        e = tf.reshape(e, expanded_shape)

    return Q, e

class KfacOptimizer():

    def __init__(self, learning_rate=0.01, momentum=0.9, clip_kl=0.01, kfac_update=2, stats_accum_iter=60, full_stats_init=False, cold_iter=100, cold_lr=None, async=False, async_stats=False, epsilon=1e-2, stats_decay=0.95, blockdiag_bias=False, channel_fac=False, factored_damping=False, approxT2=False, use_float64=False, weight_decay_dict={},max_grad_norm=0.5):
        self.max_grad_norm = max_grad_norm
        self._lr = learning_rate
        self._momentum = momentum
        self._clip_kl = clip_kl
        self._channel_fac = channel_fac
        self._kfac_update = kfac_update
        self._async = async
        self._async_stats = async_stats
        self._epsilon = epsilon
        self._stats_decay = stats_decay
        self._blockdiag_bias = blockdiag_bias
        self._approxT2 = approxT2
        self._use_float64 = use_float64
        self._factored_damping = factored_damping
        self._cold_iter = cold_iter
        if cold_lr == None:
            # good heuristics
            self._cold_lr = self._lr# * 3.
        else:
            self._cold_lr = cold_lr
        self._stats_accum_iter = stats_accum_iter
        self._weight_decay_dict = weight_decay_dict
        self._diag_init_coeff = 0.
        self._full_stats_init = full_stats_init # not used for simplification
#        if not self._full_stats_init:
        self._stats_accum_iter = self._cold_iter

        self.sgd_step = tf.Variable(0, name='KFAC/sgd_step', trainable=False)
        self.global_step = tf.Variable(0, name='KFAC/global_step', trainable=False)
        self.cold_step = tf.Variable(0, name='KFAC/cold_step', trainable=False)
        self.factor_step = tf.Variable(0, name='KFAC/factor_step', trainable=False)
        self.stats_step = tf.Variable(0, name='KFAC/stats_step', trainable=False)
        self.vFv = tf.Variable(0., name='KFAC/vFv', trainable=False)

        self.factors = {}
        self.param_vars = []
        self.stats = {}
        self.stats_eigen = {}

    def getKfacPrecondUpdates(self, gradlist, varlist):
        updatelist = []
        vg = 0.

        assert len(self.stats) > 0
        assert len(self.stats_eigen) > 0
        assert len(self.factors) > 0
        counter = 0

        grad_dict = {var: grad for grad, var in zip(gradlist, varlist)}

        for grad, var in zip(gradlist, varlist):
            GRAD_RESHAPE = False
            GRAD_TRANSPOSE = False

            fpropFactoredFishers = self.stats[var]['fprop_concat_stats']
            bpropFactoredFishers = self.stats[var]['bprop_concat_stats']

            if (len(fpropFactoredFishers) + len(bpropFactoredFishers)) > 0:
                counter += 1
                GRAD_SHAPE = grad.get_shape()
                if len(grad.get_shape()) == 1:
                    # reshape bias or 1D parameters
                    grad = tf.expand_dims(grad, 0)
                    GRAD_RESHAPE = True

                if (self.stats[var]['assnBias'] is not None) and not self._blockdiag_bias:
                    var_assnBias = self.stats[var]['assnBias']
                    grad = tf.concat([grad, tf.expand_dims(grad_dict[var_assnBias], 0)], 0)

                # project gradient to eigen space and reshape the eigenvalues for broadcasting
                eigVals = []

                for idx, stats in enumerate(self.stats[var]['fprop_concat_stats']):
                    Q = self.stats_eigen[stats]['Q']
                    Q = tf.Print(Q, [Q, Q.shape])
                    e = detectMinVal(self.stats_eigen[stats]['e'], var, name='act', debug=KFAC_DEBUG)

                    Q, e = factorReshape(Q, e, grad, facIndx=idx, ftype='act')
                    eigVals.append(e)
                    grad = gmatmul(Q, grad, transpose_a=True, reduce_dim=idx)

                for idx, stats in enumerate(self.stats[var]['bprop_concat_stats']):
                    Q = self.stats_eigen[stats]['Q']
                    e = detectMinVal(self.stats_eigen[stats]['e'], var, name='grad', debug=KFAC_DEBUG)

                    Q, e = factorReshape(Q, e, grad, facIndx=idx, ftype='grad')
                    eigVals.append(e)
                    grad = gmatmul(grad, Q, transpose_b=False, reduce_dim=idx)

                # whiten using eigenvalues
                weightDecayCoeff = 0.
                if var in self._weight_decay_dict:
                    weightDecayCoeff = self._weight_decay_dict[var]
                    if KFAC_DEBUG:
                        print(('weight decay coeff for %s is %f' % (var.name, weightDecayCoeff)))

                if self._factored_damping:
                    if KFAC_DEBUG:
                        print(('use factored damping for %s' % (var.name)))
                    coeffs = 1.
                    num_factors = len(eigVals)

                    # compute the ratio of two trace norm of the left and right KFac matrices, and their generalization
                    if len(eigVals) == 1:
                        damping = self._epsilon + weightDecayCoeff
                    else:
                        damping = tf.pow(self._epsilon + weightDecayCoeff, 1. / num_factors)
                    eigVals_tnorm_avg = [tf.reduce_mean(tf.abs(e)) for e in eigVals]
                    for e, e_tnorm in zip(eigVals, eigVals_tnorm_avg):
                        eig_tnorm_negList = [item for item in eigVals_tnorm_avg if item != e_tnorm]
                        if len(eigVals) == 1:
                            adjustment = 1.
                        elif len(eigVals) == 2:
                            adjustment = tf.sqrt(e_tnorm / eig_tnorm_negList[0])
                        else:
                            eig_tnorm_negList_prod = reduce(lambda x, y: x * y, eig_tnorm_negList)
                            adjustment = tf.pow(tf.pow(e_tnorm, num_factors - 1.) / eig_tnorm_negList_prod, 1. / num_factors)
                        coeffs *= (e + adjustment * damping)
                else:
                    coeffs = 1.
                    damping = (self._epsilon + weightDecayCoeff)
                    for e in eigVals:
                        coeffs *= e
                    coeffs += damping

                grad /= coeffs

                # project gradient back to euclidean space
                for idx, stats in enumerate(self.stats[var]['fprop_concat_stats']):
                    Q = self.stats_eigen[stats]['Q']
                    grad = gmatmul(Q, grad, transpose_a=False, reduce_dim=idx)

                for idx, stats in enumerate(self.stats[var]['bprop_concat_stats']):
                    Q = self.stats_eigen[stats]['Q']
                    grad = gmatmul(grad, Q, transpose_b=True, reduce_dim=idx)

                if (self.stats[var]['assnBias'] is not None) and not self._blockdiag_bias:
                    var_assnBias = self.stats[var]['assnBias']
                    C_plus_one = int(grad.get_shape()[0])
                    grad_assnBias = tf.reshape(tf.slice(grad, begin=[C_plus_one - 1, 0], size=[1, -1]), var_assnBias.get_shape())
                    grad_assnWeights = tf.slice(grad, begin=[0, 0], size=[C_plus_one - 1, -1])
                    grad_dict[var_assnBias] = grad_assnBias
                    grad = grad_assnWeights

                if GRAD_RESHAPE:
                    grad = tf.reshape(grad, GRAD_SHAPE)

                grad_dict[var] = grad

        print(('projecting %d gradient matrices' % counter))

        for g, var in zip(gradlist, varlist):
            grad = grad_dict[var]
            ### clipping ###
            if KFAC_DEBUG:
                print(('apply clipping to %s' % (var.name)))
            tf.Print(grad, [tf.sqrt(tf.reduce_sum(tf.pow(grad, 2)))], "Euclidean norm of new grad")
            local_vg = tf.reduce_sum(grad * g * (self._lr * self._lr))
            vg += local_vg

        # recale everything
        if KFAC_DEBUG:
            print('apply vFv clipping')

        scaling = tf.minimum(1., tf.sqrt(self._clip_kl / vg))
        if KFAC_DEBUG:
            scaling = tf.Print(scaling, [tf.convert_to_tensor('clip: '), scaling, tf.convert_to_tensor(' vFv: '), vg])
        with tf.control_dependencies([tf.assign(self.vFv, vg)]):
            updatelist = [grad_dict[var] for var in varlist]
            for i, item in enumerate(updatelist):
                updatelist[i] = scaling * item

        return updatelist

    def computeStatsEigen(self):
        """ compute the eigen decomp using copied var stats to avoid concurrent read/write from other queue """
        with tf.device('/cpu:0'):

            stats_eigen = self.stats_eigen
            computedEigen = {}
            eigen_reverse_lookup = {}
            updateOps = []

            for stats_var in stats_eigen:
                if stats_var not in computedEigen:
                    eigens = tf.self_adjoint_eig(stats_var)
                    e = eigens[0]
                    Q = eigens[1]
                    if self._use_float64:
                        e = tf.cast(e, tf.float32)
                        Q = tf.cast(Q, tf.float32)
                    updateOps.append(e)
                    updateOps.append(Q)
#                    tf.Print(Q, [Q, Q.shape])
                    computedEigen[stats_var] = {'e': e, 'Q': Q}
                    eigen_reverse_lookup[e] = stats_eigen[stats_var]['e']
                    eigen_reverse_lookup[Q] = stats_eigen[stats_var]['Q']

            self.eigen_reverse_lookup = eigen_reverse_lookup
            self.eigen_update_list = updateOps

            if KFAC_DEBUG:
                self.eigen_update_list = [item for item in updateOps]
                with tf.control_dependencies(updateOps):
                    updateOps.append(tf.Print(tf.constant(0.), [tf.convert_to_tensor('computed factor eigen')]))

        return updateOps

    def applyStatsEigen(self, eigen_list):
        updateOps = []
        print(('updating %d eigenvalue/vectors' % len(eigen_list)))
        for i, (tensor, mark) in enumerate(zip(eigen_list, self.eigen_update_list)):
            stats_eigen_var = self.eigen_reverse_lookup[mark]
            updateOps.append(tf.assign(stats_eigen_var, tensor, use_locking=True))

        with tf.control_dependencies(updateOps):
            factor_step_op = tf.assign_add(self.factor_step, 1)
            updateOps.append(factor_step_op)
            if KFAC_DEBUG:
                updateOps.append(tf.Print(tf.constant(0.), [tf.convert_to_tensor('updated kfac factors')]))
        return updateOps

    def getStatsEigen(self, stats=None):
        if len(self.stats_eigen) == 0:
            stats_eigen = {}
            if stats is None:
                stats = self.stats

            tmpEigenCache = {}
            with tf.device('/cpu:0'):
                for var in stats:
                    for key in ['fprop_concat_stats', 'bprop_concat_stats']:
                        for stats_var in stats[var][key]:
                            if stats_var not in tmpEigenCache:
                                stats_dim = stats_var.get_shape()[1].value
                                e = tf.Variable(tf.ones([stats_dim]), name='KFAC_FAC/' + stats_var.name.split(':')[0] + '/e', trainable=False)
                                Q = tf.Variable(tf.diag(tf.ones([stats_dim])), name='KFAC_FAC/' + stats_var.name.split(':')[0] + '/Q', trainable=False)
                                stats_eigen[stats_var] = {'e': e, 'Q': Q}
                                tmpEigenCache[stats_var] = stats_eigen[stats_var]
                            else:
                                stats_eigen[stats_var] = tmpEigenCache[stats_var]
            self.stats_eigen = stats_eigen
        return self.stats_eigen

    def apply_gradients_kfac(self, grads):
        g, varlist = list(zip(*grads))

        if len(self.stats_eigen) == 0:
            self.getStatsEigen()

        updateOps = []
        global_step_op = tf.assign_add(self.global_step, 1)
        updateOps.append(global_step_op)

        with tf.control_dependencies([global_step_op]):

            # compute updates
            assert self._update_stats_op != None
            updateOps.append(self._update_stats_op)
            dependency_list = []
            if not self._async:
                dependency_list.append(self._update_stats_op)

            with tf.control_dependencies(dependency_list):

                updateFactorOps = tf.cond(tf.logical_and(tf.equal(tf.mod(self.stats_step, self._kfac_update), tf.convert_to_tensor(0)),
                                                         tf.greater_equal(self.stats_step, self._stats_accum_iter)),
                                          lambda: tf.group(*self.applyStatsEigen(self.computeStatsEigen())), tf.no_op)
                updateOps.append(updateFactorOps)

                with tf.control_dependencies([updateFactorOps]):

                    def gradOp():
                        return list(g)

                    def getKfacGradOp():
                        return self.getKfacPrecondUpdates(g, varlist)

                    u = tf.cond(tf.greater(self.factor_step, tf.convert_to_tensor(0)), getKfacGradOp, gradOp)
                    optim = tf.train.MomentumOptimizer(self._lr * (1. - self._momentum), self._momentum)
                    optim_op = optim.apply_gradients(list(zip(u, varlist)))

                    updateOps.append(optim_op)

        return tf.group(*updateOps)

    def apply_gradients(self, grads):
        coldOptim = tf.train.MomentumOptimizer(self._cold_lr, self._momentum)

        def coldSGDstart():
            sgd_grads, sgd_var = zip(*grads)

            if self.max_grad_norm != None:
                sgd_grads, sgd_grad_norm = tf.clip_by_global_norm(sgd_grads,self.max_grad_norm)

            sgd_grads = list(zip(sgd_grads,sgd_var))

            sgd_step_op = tf.assign_add(self.sgd_step, 1)
            coldOptim_op = coldOptim.apply_gradients(sgd_grads)
            if KFAC_DEBUG:
                with tf.control_dependencies([sgd_step_op, coldOptim_op]):
                    sgd_step_op = tf.Print(sgd_step_op, [self.sgd_step, tf.convert_to_tensor('doing cold sgd step')])
            return tf.group(*[sgd_step_op, coldOptim_op])

        kfacOptim_op = self.apply_gradients_kfac(grads)

        def warmKFACstart():
            return kfacOptim_op

        return tf.cond(tf.greater(self.sgd_step, self._cold_iter), warmKFACstart, coldSGDstart)

    def _apply_stats(self, statsUpdates, accumulate=False, accumulateCoeff=0.):
        updateOps = []
        # obtain the stats var list
        for stats_var in statsUpdates:
            stats_new = statsUpdates[stats_var]
            if accumulate:
                # simple superbatch averaging
                update_op = tf.assign_add(stats_var, accumulateCoeff * stats_new, use_locking=True)
            else:
                # exponential running averaging
                update_op = tf.assign(stats_var, stats_var * self._stats_decay, use_locking=True)
                update_op = tf.assign_add(update_op, (1. - self._stats_decay) * stats_new, use_locking=True)
            updateOps.append(update_op)

        with tf.control_dependencies(updateOps):
            stats_step_op = tf.assign_add(self.stats_step, 1)

        if KFAC_DEBUG:
            stats_step_op = (tf.Print(stats_step_op,
                                      [tf.convert_to_tensor('step:'),
                                       self.global_step,
                                       tf.convert_to_tensor('fac step:'),
                                       self.factor_step,
                                       tf.convert_to_tensor('sgd step:'),
                                       self.sgd_step,
                                       tf.convert_to_tensor('Accum:'),
                                       tf.convert_to_tensor(accumulate),
                                       tf.convert_to_tensor('Accum coeff:'),
                                       tf.convert_to_tensor(accumulateCoeff),
                                       tf.convert_to_tensor('stat step:'),
                                       self.stats_step, updateOps[0], updateOps[1]]))
        return [stats_step_op, ]

    def apply_stats(self, statsUpdates):
        """ compute stats and update/apply the new stats to the running average
        """
        def updateAccumStats():
            return tf.group(*self._apply_stats(statsUpdates, accumulate=True, accumulateCoeff=1. / self._stats_accum_iter))

        def updateRunningAvgStats(statsUpdates, fac_iter=1):
            return tf.group(*self._apply_stats(statsUpdates))

        update_stats_op = tf.cond(tf.greater_equal(self.stats_step, self._stats_accum_iter), lambda: updateRunningAvgStats(statsUpdates), updateAccumStats)
        self._update_stats_op = update_stats_op
        return None

    def getStats(self, factors, varlist):
        if len(self.stats) == 0:
            # initialize stats variables on CPU because eigen decomp is computed on CPU
            with tf.device('/cpu'):
                tmpStatsCache = {}

                for var in varlist:
                    fpropFactor = factors[var]['fpropFactors_concat']
                    bpropFactor = factors[var]['bpropFactors_concat']
                    opType = factors[var]['opName']
                    self.stats[var] = {'opName': opType,
                                       'fprop_concat_stats': [],
                                       'bprop_concat_stats': [],
                                       'assnWeights': factors[var]['assnWeights'],
                                       'assnBias': factors[var]['assnBias'],
                                       }
                    if fpropFactor is not None:
                        if fpropFactor not in tmpStatsCache:
                            # D x D covariance matrix
                            fpropFactor_size = fpropFactor.get_shape()[-1]

                            # use homogeneous coordinate
                            if not self._blockdiag_bias and self.stats[var]['assnBias']:
                                fpropFactor_size += 1

                            slot_fpropFactor_stats = tf.Variable(tf.diag(tf.ones([fpropFactor_size])) * self._diag_init_coeff,
                                                                 name='KFAC_STATS/' + fpropFactor.op.name, trainable=False)
                            self.stats[var]['fprop_concat_stats'].append(slot_fpropFactor_stats)
                            tmpStatsCache[fpropFactor] = self.stats[var]['fprop_concat_stats']
                        else:
                            self.stats[var]['fprop_concat_stats'] = tmpStatsCache[fpropFactor]

                    if bpropFactor is not None:
                        # no need to collect backward stats for bias vectors if using homogeneous coordinates
                        if not((not self._blockdiag_bias) and self.stats[var]['assnWeights']):
                            if bpropFactor not in tmpStatsCache:
                                slot_bpropFactor_stats = tf.Variable(tf.diag(tf.ones([bpropFactor.get_shape()[-1]])) * self._diag_init_coeff,
                                                                     name='KFAC_STATS/' + bpropFactor.op.name, trainable=False)
                                self.stats[var]['bprop_concat_stats'].append(slot_bpropFactor_stats)
                                tmpStatsCache[bpropFactor] = self.stats[var]['bprop_concat_stats']
                            else:
                                self.stats[var]['bprop_concat_stats'] = tmpStatsCache[bpropFactor]

        return self.stats

    def getFactors(self, g, varlist):
        graph = tf.get_default_graph()
        factorTensors = {}
        fpropTensors = []
        bpropTensors = []
        opTypes = []
        fops = []

        def searchFactors(gradient, graph):
            # hard coded search stratergy
            bpropOp = gradient.op
            bpropOp_name = bpropOp.name

            bTensors = []
            fTensors = []

            fpropOp_name = re.search('gradientsSampled(_[0-9]+|)/(.+?)_grad', bpropOp_name).group(2)
            fpropOp = graph.get_operation_by_name(fpropOp_name)
            if fpropOp.op_def.name in KFAC_OPS:
                # Known OPs
                bTensor = [i for i in bpropOp.inputs if 'gradientsSampled' in i.name][-1]
                bTensorShape = fpropOp.outputs[0].get_shape()
                if bTensor.get_shape()[0].value == None:
                    bTensor.set_shape(bTensorShape)
                bTensors.append(bTensor)
                if fpropOp.op_def.name == 'BiasAdd':
                    fTensors = []
                else:
                    fTensors.append([i for i in fpropOp.inputs if param.op.name not in i.name][0])
                fpropOp_name = fpropOp.op_def.name
            else:
                # unknown OPs, block approximation used
                bInputsList = [i for i in bpropOp.inputs[0].op.inputs if 'gradientsSampled' in i.name if 'Shape' not in i.name]
                if len(bInputsList) > 0:
                    bTensor = bInputsList[0]
                    bTensorShape = fpropOp.outputs[0].get_shape()
                    if len(bTensor.get_shape()) > 0 and bTensor.get_shape()[0].value == None:
                        bTensor.set_shape(bTensorShape)
                    bTensors.append(bTensor)
                fpropOp_name = opTypes.append('UNK-' + fpropOp.op_def.name)

            return {'opName': fpropOp_name, 'op': fpropOp, 'fpropFactors': fTensors, 'bpropFactors': bTensors}

        for t, param in zip(g, varlist):
            if KFAC_DEBUG:
                print(('get factor for '+param.name))
            factors = searchFactors(t, graph)
            factorTensors[param] = factors

        ########
        # check associated weights and bias for homogeneous coordinate representation and check redundent factors
        # TO-DO: there may be a bug to detect associate bias and weights for forking layer, e.g. in inception models.
        for param in varlist:
            factorTensors[param]['assnWeights'] = None
            factorTensors[param]['assnBias'] = None
        for param in varlist:
            if factorTensors[param]['opName'] == 'BiasAdd':
                factorTensors[param]['assnWeights'] = None
                for item in varlist:
                    if len(factorTensors[item]['bpropFactors']) > 0:
                        if (set(factorTensors[item]['bpropFactors']) == set(factorTensors[param]['bpropFactors'])) and (len(factorTensors[item]['fpropFactors']) > 0):
                            factorTensors[param]['assnWeights'] = item
                            factorTensors[item]['assnBias'] = param
                            factorTensors[param]['bpropFactors'] = factorTensors[item]['bpropFactors']

        # concatenate the additive gradients along the batch dimension, i.e. assuming independence structure
        for key in ['fpropFactors', 'bpropFactors']:
            for i, param in enumerate(varlist):
                if len(factorTensors[param][key]) > 0:
                    if (key + '_concat') not in factorTensors[param]:
                        name_scope = factorTensors[param][key][0].name.split(':')[0]
                        with tf.name_scope(name_scope):
                            factorTensors[param][key + '_concat'] = tf.concat(factorTensors[param][key], 0)
                else:
                    factorTensors[param][key + '_concat'] = None
                for j, param2 in enumerate(varlist[(i + 1):]):
                    if (len(factorTensors[param][key]) > 0) and (set(factorTensors[param2][key]) == set(factorTensors[param][key])):
                        factorTensors[param2][key] = factorTensors[param][key]
                        factorTensors[param2][key + '_concat'] = factorTensors[param][key + '_concat']

        if KFAC_DEBUG:
            for items in zip(varlist, fpropTensors, bpropTensors, opTypes):
                print((items[0].name, factorTensors[item]))
        self.factors = factorTensors
        return factorTensors

    def compute_stats(self, loss_sampled, var_list=None):
        varlist = var_list
        if varlist is None:
            varlist = tf.trainable_variables()

        gs = tf.gradients(loss_sampled, varlist, name='gradientsSampled')
        self.gs = gs
        factors = self.getFactors(gs, varlist)
        stats = self.getStats(factors, varlist)

        statsUpdates = {}
        statsUpdates_cache = {}
        for var in varlist:
            opType = factors[var]['opName']
            fops = factors[var]['op']
            fpropFactor = factors[var]['fpropFactors_concat']
            fpropStats_vars = stats[var]['fprop_concat_stats']
            bpropFactor = factors[var]['bpropFactors_concat']
            bpropStats_vars = stats[var]['bprop_concat_stats']
            SVD_factors = {}
            for stats_var in fpropStats_vars:
                stats_var_dim = int(stats_var.get_shape()[0])
                if stats_var not in statsUpdates_cache:
                    old_fpropFactor = fpropFactor
                    B = (tf.shape(fpropFactor)[0])  # batch size

                    fpropFactor_size = int(fpropFactor.get_shape()[-1])
                    if stats_var_dim == (fpropFactor_size + 1) and not self._blockdiag_bias:
                        # use homogeneous coordinates
                        fpropFactor = tf.concat([fpropFactor, tf.ones([tf.shape(fpropFactor)[0], 1])], 1)

                    # average over the number of data points in a batch divided by B
                    cov = tf.matmul(fpropFactor, fpropFactor, transpose_a=True) / tf.cast(B, tf.float32)
                    statsUpdates[stats_var] = cov
                    statsUpdates_cache[stats_var] = cov

            for stats_var in bpropStats_vars:
                stats_var_dim = int(stats_var.get_shape()[0])
                if stats_var not in statsUpdates_cache:
                    old_bpropFactor = bpropFactor
                    bpropFactor_shape = bpropFactor.get_shape()
                    B = tf.shape(bpropFactor)[0]  # batch size
#                    C = int(bpropFactor_shape[-1])  # num channels

                    # assume sampled loss is averaged. TO-DO:figure out better way to handle this
                    bpropFactor *= tf.to_float(B)

                    cov_b = tf.matmul(bpropFactor, bpropFactor, transpose_a=True) / tf.to_float(tf.shape(bpropFactor)[0])

                    statsUpdates[stats_var] = cov_b
                    statsUpdates_cache[stats_var] = cov_b

        if KFAC_DEBUG:
            aKey = list(statsUpdates.keys())[0]
            statsUpdates[aKey] = tf.Print(statsUpdates[aKey], [tf.convert_to_tensor('step:'), self.global_step, tf.convert_to_tensor('computing stats'), ])
        self.statsUpdates = statsUpdates
        return statsUpdates

    def compute_and_apply_stats(self, loss_sampled, var_list=None):
        varlist = var_list
        if varlist is None:
            varlist = tf.trainable_variables()

        stats = self.compute_stats(loss_sampled, var_list=varlist)
        self.apply_stats(stats)
        return None

    def compute_gradients(self, loss, var_list=None):
        varlist = var_list
        if varlist is None:
            varlist = tf.trainable_variables()
        g = tf.gradients(loss, varlist)

        return [(a, b) for a, b in zip(g, varlist)]

    def minimize(self, loss, loss_sampled, var_list=None):
        grads = self.compute_gradients(loss, var_list=var_list)
        self.compute_and_apply_stats(loss_sampled, var_list=var_list)
        return self.apply_gradients(grads)
