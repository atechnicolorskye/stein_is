# from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as ds
# import time
# from tensorflow.python.client import timeline

# import pdb


def median(x):
    x = tf.reshape(x, [-1])
    med = tf.floordiv(tf.shape(x)[0], 2)
    check_parity = tf.equal(tf.to_double(med), tf.divide(tf.to_double(tf.shape(x)[0]), 2.))

    def is_true():
        return 0.5 * tf.reduce_sum(tf.nn.top_k(x, med + 1).values[-2:])

    def is_false():
        return tf.nn.top_k(x, med + 1).values[-1]

    return tf.cond(check_parity, is_true, is_false)


# https://stackoverflow.com/questions/44194063/calculate-log-of-determinant-in-tensorflow-when-determinant-overflows-underflows
# from https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342
# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


# from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/linalg_grad.py
# Gradient for logdet
def logdet_grad(op, grad):
    a = op.inputs[0]
    a_adj_inv = tf.matrix_inverse(a, adjoint=True)
    out_shape = tf.concat([tf.shape(a)[:-2], [1, 1]], axis=0)
    return tf.reshape(grad, out_shape) * a_adj_inv


# Define logdet by calling numpy.linalg.slogdet
def logdet(a, name=None):
    with tf.name_scope(name, 'LogDet', [a]) as name:
        res = py_func(lambda a: np.linalg.slogdet(a)[1],
                      [a],
                      tf.float64,
                      name=name,
                      grad=logdet_grad)  # set the gradient
        return res


class GMM(object):
    # Gaussian Mixture Model
    def __init__(self, mu, sigma, weights, dim):
        # Required parameters
        self.weights = weights
        self.dim = dim

        self.distributions = []
        for i in range(weights.shape[0]):
            mu_, sigma_ = mu[i] * np.ones(dim), sigma[i] * np.ones(dim)
            # print(mu_, sigma_)
            mvnd_i = tf.contrib.distributions.MultivariateNormalDiag(mu_, sigma_)
            self.distributions.append(mvnd_i)
        self.mix = tf.contrib.distributions.Mixture(tf.contrib.distributions.Categorical(probs=self.weights), self.distributions)

    # Utility Functions
    def reshape_fish_comp(self, vec):
        # Reshapes a component of a Fisher vector
        vec_shape = tf.shape(vec)
        return tf.reshape(tf.transpose(vec, perm=[1, 0, 2]), [vec_shape[1], vec_shape[0] * vec_shape[2]])

    # Differentials
    def log_px(self, x):
        x_t = tf.convert_to_tensor(x)
        return self.mix.log_prob(x_t)

    def dx_log_px(self, x):
        # dx_log_px = 1 / exp(log(sum(exp(log(w_i) + log(p_i(x))))))
        #            * sum(exp(log(w_i) + log(p_i(x)) + log(-(x - mu)/sigma^2)))
        # Use symbolic differentiation instead
        x_t = tf.convert_to_tensor(x)
        log_px = self.mix.log_prob(x_t)
        return tf.gradients(log_px, [x_t])[0], log_px # , d_log_px_gc

    def dtheta_log_px(self, x):
        # Returns an n * # of components matrix
        # x_t = x
        n_points = x.shape[0]
        x_t = tf.convert_to_tensor(x)
        # Have to use log_px, log_px_id for stability
        log_px = tf.reshape(self.mix.log_prob(x_t), (-1, 1))
        dw_log_px_, dmu_log_px_, dsigma2_log_px_, w_px_i_px, exponent_, xi_ = [], [], [], [], [], []
        for i in range(self.weights.shape[0]):
            log_px_i = tf.reshape(self.distributions[i].log_prob(x_t), [-1, 1])
            dw_log_px_.append(tf.exp(log_px_i - log_px))
            w_px_i_px_ = self.weights[i] * dw_log_px_[i]
            w_px_i_px.append(w_px_i_px_)
            exponent_i = tf.divide((x_t - self.distributions[i].mean()), self.distributions[i].variance())
            exponent_.append(exponent_i)
            dmu_log_px_.append(w_px_i_px_ * exponent_i)
            xi_i = 0.5 * (tf.multiply(exponent_i, exponent_i) - tf.divide(1, self.distributions[i].variance()))
            xi_.append(xi_i)
            dsigma2_log_px_.append(w_px_i_px_ * xi_i)
        dw_log_px_, dmu_log_px_, dsigma2_log_px_ =  tf.stack(dw_log_px_), tf.stack(dmu_log_px_), tf.stack(dsigma2_log_px_)
        dw_log_px, dmu_log_px, dsigma2_log_px = self.reshape_fish_comp(dw_log_px_), self.reshape_fish_comp(dmu_log_px_), self.reshape_fish_comp(dsigma2_log_px_)
        dtheta_log_px_ = tf.concat([dw_log_px, dmu_log_px, dsigma2_log_px], 1)
        dtheta_log_px_norm = tf.norm(dtheta_log_px_, axis=1, keepdims=True)
        dtheta_log_px = dtheta_log_px_ / dtheta_log_px_norm
        return dtheta_log_px, [dw_log_px_ / dtheta_log_px_norm, dmu_log_px_ / dtheta_log_px_norm, dsigma2_log_px_ / dtheta_log_px_norm], dmu_log_px_, w_px_i_px / dtheta_log_px_norm, exponent_, xi_

    def dx_dtheta_log_px(self, dmu_log_px_, w_px_i_px_norm, exponent_, xi_):
        # Returns a n * d * # of components matrix
        dx_dw_log_px_, dx_dmu_log_px_, dx_dsigma2_log_px_ = [], [], []
        zeta = tf.reduce_sum(dmu_log_px_, [0])
        for i in range(self.weights.shape[0]):
            zeta_m_exponent = tf.expand_dims(zeta - exponent_[i], 1)
            w_px_i_px_norm_i = tf.expand_dims(w_px_i_px_norm[i], -1)
            dx_dw_log_px_.append(zeta_m_exponent * w_px_i_px_norm_i)
            diag_precision = tf.diag(1. / self.distributions[i].variance())
            exponent_i_tensor = tf.expand_dims(exponent_[i], -1)
            dx_dmu_log_px_.append((tf.matmul(exponent_i_tensor, zeta_m_exponent) + diag_precision) * w_px_i_px_norm_i)
            xi_i_tensor = tf.expand_dims(xi_[i], -1)
            exponent_i_diag = tf.matrix_diag(exponent_[i])
            diag_precision = tf.expand_dims(diag_precision, 0)
            dx_dsigma2_log_px_.append((tf.matmul(xi_i_tensor, zeta_m_exponent) + exponent_i_diag * diag_precision) * w_px_i_px_norm_i)
        dx_dw_log_px, dx_dmu_log_px, dx_dsigma2_log_px = tf.stack(dx_dw_log_px_), tf.stack(dx_dmu_log_px_), tf.stack(dx_dsigma2_log_px_)
        return [dx_dw_log_px, dx_dmu_log_px, dx_dsigma2_log_px]


class SteinIS(object):
    def __init__(self, m_model, dim, n_leaders, n_followers, mu, sigma, kernel):  # n_trials, step_size=0.01):
        # Required parameters
        self.m_model = m_model
        self.dim = dim
        self.n_leaders = n_leaders
        self.n_followers = n_followers
        self.mu = mu
        self.sigma = sigma
        self.kernel = kernel

        # Inputs
        self.step_size = tf.placeholder(tf.float64, [])
        self.seed = tf.placeholder(tf.int32, [])
        # if kernel == 'all':
        #     self.B_f, self.B_se = self.B
        #     self.q_density_f, self.q_density_se = self.q_density
        #     self.log_q_update_f, self.log_q_update_se = self.log_q_update
        #     self.A_f, self.A_se = self.A

        # Register functions for debugging
        # k_A_A, sum_d_A_k_A_A, A_Squared, h = self.construct_map()
        # k_A_B, sum_d_A_k_A_B = self.apply_map()
        # A, B, n_q_density = self.svgd_update()
        # # self.q_density = self.density_update()

        # Register functions
        self.B, self.q_density, self.log_q_update, self.A = self.initialise_variables(self.mu, self.sigma, self.n_leaders, self.n_followers, self.dim)
        self.construct_leader_map()
        self.construct_follower_map()
        self.svgd_update()

    # Utiliy Functions
    def k_dx_x(self, d_x, x):
        return tf.einsum('ijkl,imk->jml', d_x, x)

    def k_x_dx(self, x, d_x):
        return tf.einsum('imk,ijkl->mjl', x, d_x)

    def sum_dx_log_px_T_k_x_dx(self, dx_log_px, k_x_dx):
        return tf.einsum('ij,ikl->kjl', dx_log_px, k_x_dx)

    def sum_d_d_kernel(self, d_x, d_y):
        return tf.einsum('ijkl,imkn->mln', d_x, d_y)

    def initialise_variables(self, mu, sigma, n_leaders, n_followers, dim):
        # followers = tf.reshape(init_distribution.sample(self.n_trials * self.n_followers, seed=123), [self.n_trials, self.n_followers, self.h_dim]
        # leaders = tf.reshape(init_distribution.sample(self.n_trials * self.n_leaders, seed=123), [self.n_trials, self.n_leaders, self.h_dim]

        init_distribution = tf.contrib.distributions.MultivariateNormalDiag(mu * tf.zeros(dim, tf.float64), sigma * tf.ones(dim, tf.float64))
        # Not absolute
        followers = tf.Variable(init_distribution.sample(sample_shape=n_followers))
        leaders = tf.Variable(init_distribution.sample(sample_shape=n_leaders))
        # Absolute
        # followers = tf.Variable(tf.abs(init_distribution.sample(n_followers)))
        # leaders = tf.Variable(tf.abs(init_distribution.sample(n_leaders)))
        q_density = init_distribution.prob(followers)
        log_q_update = tf.Variable(tf.zeros([n_followers], dtype=np.float64))
        return followers, q_density, log_q_update, leaders

    # Algorithm Proper
    def construct_leader_map(self):
        # Refer to leader as A as in SteinIS
        with tf.variable_scope('k_A_A'):
            if self.kernel == 'se':
                x2_A_A_T = 2. * tf.matmul(self.A, tf.transpose(self.A))
                self.A_Squared = tf.reduce_sum(tf.square(self.A), keepdims=True, axis=1)
                A_A_Distance_Squared = self.A_Squared - x2_A_A_T + tf.transpose(self.A_Squared)
                # h_num = tf.square(median(tf.sqrt(A_A_Distance_Squared)))
                h_num = median(A_A_Distance_Squared)
                h_dem = 2. * tf.log(tf.to_double(self.n_leaders) + 1.)
                h = h_num / h_dem
                self.h = h
                self.k_A_A = tf.exp(-A_A_Distance_Squared / h)
            elif self.kernel == 'fisher':
                self.A_dtheta_log_px, [self.A_dw_log_px, self.A_dmu_log_px, self.A_dsigma2_log_px], A_dmu_log_px_, w_px_i_px_norm, exponent_, xi_ = self.m_model.dtheta_log_px(self.A)
                [self.A_dx_dw_log_px, self.A_dx_dmu_log_px, self.A_dx_dsigma2_log_px] = self.m_model.dx_dtheta_log_px(A_dmu_log_px_, w_px_i_px_norm, exponent_, xi_)
                self.k_A_A = tf.matmul(self.A_dtheta_log_px, tf.transpose(self.A_dtheta_log_px))
        # Can't use vanilla tf.gradients as it sums dy/dx wrt to dx, want sum dy/dx wrt to dy, tf.map_fn is not available
        # tf.gradients also do not provide accurate gradients in this case
        with tf.variable_scope('sum_d_A_k_A_A'):
            if self.kernel == 'se':
                self.sum_d_A_k_A_A = tf.stack([tf.squeeze(tf.matmul(tf.reshape(self.k_A_A[:, i], (1, -1)), (-2 * (self.A - self.A[i]) / h))) for i in range(self.n_leaders)])
                # return self.k_A_A, self.sum_d_A_k_A_A, self.A_Squared, self.h  # , self.sum_d_A_k_A_A_gc
            elif self.kernel == 'fisher':
                self.sum_d_A_k_A_A = tf.reduce_sum(self.k_dx_x(self.A_dx_dw_log_px, self.A_dw_log_px) + self.k_dx_x(self.A_dx_dmu_log_px, self.A_dmu_log_px) + self.k_dx_x(self.A_dx_dsigma2_log_px, self.A_dsigma2_log_px), 0)
                # return self.k_A_A, self.sum_d_A_k_A_A, self.A_dtheta_log_px, self.A_dx_dw_log_px, self.A_dx_dmu_log_px, self.A_dx_dsigma2_log_px

    def construct_follower_map(self):
        # Refer to follower as B as in SteinIS
        with tf.variable_scope('k_A_B'):
            if self.kernel == 'se':
                x2_A_B_T = 2. * tf.matmul(self.A, tf.transpose(self.B))
                B_Squared = tf.reduce_sum(tf.square(self.B), keepdims=True, axis=1)
                A_B_Distance_Squared = self.A_Squared - x2_A_B_T + tf.transpose(B_Squared)
                self.k_A_B = tf.exp(-A_B_Distance_Squared / self.h)
            elif self.kernel == 'fisher':
                self.B_dtheta_log_px, [dw_log_px, dmu_log_px, dsigma2_log_px], dmu_log_px_, w_px_, exponent_, xi_ = self.m_model.dtheta_log_px(self.B)
                [self.B_dx_dw_log_px, self.B_dx_dmu_log_px, self.B_dx_dsigma2_log_px] = self.m_model.dx_dtheta_log_px(dmu_log_px_, w_px_, exponent_, xi_)
                # Normalise properly!
                self.k_A_B = tf.matmul(self.A_dtheta_log_px, tf.transpose(self.B_dtheta_log_px))
        with tf.variable_scope('sum_d_A_k_A_B'):
            if self.kernel == 'se':
                self.sum_d_A_k_A_B = tf.stack([tf.squeeze(tf.matmul(tf.reshape(self.k_A_B[:, i], (1, -1)), (-2 * (self.A - self.B[i]) / self.h))) for i in range(self.n_followers)])
                # return self.k_A_B, self.sum_d_A_k_A_B  # , self.sum_d_A_k_A_Bp, self.sum_d_A_k_A_Bn, self.h
            elif self.kernel == 'fisher':
                self.sum_d_A_k_A_B = tf.reduce_sum(self.k_dx_x(self.A_dx_dw_log_px, dw_log_px) + self.k_dx_x(self.A_dx_dmu_log_px, dmu_log_px) + self.k_dx_x(self.A_dx_dsigma2_log_px, dsigma2_log_px), 0)
                # return self.k_A_B, self.sum_d_A_k_A_B, self.B_dtheta_log_px, self.B_dx_dw_log_px, self.B_dx_dmu_log_px, self.B_dx_dsigma2_log_px

    def svgd_update(self):
        with tf.variable_scope('dx_log_pA'):
            self.dx_log_pA, self.log_pA = self.m_model.dx_log_px(self.A)
        with tf.variable_scope('n_A'):
            self.sum_d_log_pA_T_k_A_A = tf.matmul(self.k_A_A, self.dx_log_pA)
            phi_A = (self.sum_d_log_pA_T_k_A_A + self.sum_d_A_k_A_A) / self.n_leaders
            n_A = self.A.assign(self.A + self.step_size * phi_A)
        with tf.variable_scope('n_B'):
            self.sum_d_log_pA_T_k_A_B = tf.matmul(tf.transpose(self.k_A_B), self.dx_log_pA)
            phi_B = (self.sum_d_log_pA_T_k_A_B + self.sum_d_A_k_A_B) / self.n_leaders
            n_B = self.B.assign(self.B + self.step_size * phi_B)
        with tf.variable_scope('d_B_phi_B'):
            if self.kernel == 'se':
            # See http://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf
                d_B_phi_B = []
                # sum_d_B_d_A_k_A_B_gc = []
                for i in range(self.n_followers):
                    x = 2 * (self.A - self.B[i]) / self.h
                    term_1 = (2 / self.h) * tf.reduce_sum(self.k_A_B[:, i]) * tf.eye(self.dim, dtype=tf.float64)
                    term_2 = tf.matmul((tf.transpose(self.dx_log_pA) - tf.transpose(x)), tf.matmul(tf.diag(self.k_A_B[:, i]), x))
                    d_B_phi_B.append(term_1 + term_2)
                    # sum_d_B_d_A_k_A_B_gc.append(tf.reduce_sum((self.sum_d_A_k_A_Bp[i, :] - self.sum_d_A_k_A_Bn[i, :]) / 2e-05))
                # self.sum_d_B_d_A_k_A_B, self.sum_d_B_d_A_k_A_B_gc = tf.stack(sum_d_B_d_A_k_A_B), sum_d_B_d_A_k_A_B_gc
                # sum_d_log_pA_T_d_B_k_A_B = tf.stack([ tf.matmul(tf.diag(self.k_A_B[:, i]), 2 * (self.A - self.B[i]) / self.h)) for i in range(self.n_followers)])
                # d_B_phi_B = (sum_d_log_pA_T_d_B_k_A_B + tf.stack(sum_d_B_d_A_k_A_B)) / self.n_leaders
                d_B_phi_B = tf.stack(d_B_phi_B) / self.n_leaders
                # self.d_B_phi_B = d_B_phi_B
            elif self.kernel == 'fisher':
                d_B_k_A_B = self.k_x_dx(self.A_dw_log_px, self.B_dx_dw_log_px) + self.k_x_dx(self.A_dmu_log_px, self.B_dx_dmu_log_px) + self.k_x_dx(self.A_dsigma2_log_px, self.B_dx_dsigma2_log_px)
                # self.dx_log_pA, self.d_B_k_A_B = d_log_pA, d_B_k_A_B
                sum_d_log_pA_d_B_k_A_B_T = self.sum_dx_log_px_T_k_x_dx(self.dx_log_pA, d_B_k_A_B)
                sum_d_B_d_A_k_A_B = self.sum_d_d_kernel(self.A_dx_dw_log_px, self.B_dx_dw_log_px) + self.sum_d_d_kernel(self.A_dx_dmu_log_px, self.B_dx_dmu_log_px) + self.sum_d_d_kernel(self.A_dx_dsigma2_log_px, self.B_dx_dsigma2_log_px)
                d_B_phi_B = (sum_d_log_pA_d_B_k_A_B_T + sum_d_B_d_A_k_A_B) / self.n_leaders
        with tf.variable_scope('density_update'):
            I = tf.eye(self.dim, dtype=tf.float64)
            self.d_B_phi_B_update = self.step_size * d_B_phi_B
            self.I_d_B_phi_B_update = I + self.d_B_phi_B_update
            # self.I_d_B_phi_B = tf.map_fn(lambda x: (I + self.step_size * x), d_B_phi_B)
            log_abs_det_I_d_B_phi_B = tf.map_fn(lambda x: logdet(I + self.step_size * x), d_B_phi_B)
            n_log_q_update = self.log_q_update.assign(self.log_q_update + log_abs_det_I_d_B_phi_B)
        self.updates = tf.group(n_A, n_B, n_log_q_update)

