import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow.contrib.distributions as ds
import time
from tensorflow.python.client import timeline

import pdb


def median(x):
    x = tf.reshape(x, [-1])
    med = tf.floordiv(tf.shape(x)[0], 2)
    check_parity = tf.equal(tf.to_double(med), tf.divide(tf.to_double(tf.shape(x)[0]), 2.))
    def is_true():
        return 0.5 * tf.reduce_sum(tf.nn.top_k(x, med+1).values[-2:]) 
    def is_false():
        return tf.nn.top_k(x, med+1).values[-1]
    return tf.cond(check_parity, is_true, is_false) 


class GMM(object):
    def __init__(self, mu, sigma, weights, dim):
        # Required parameters 
        self.mu = mu
        self.sigma = sigma
        self.weights = weights
        self.dim = dim

        distributions = []
        for i in range(weights.shape[0]):
            mu_, sigma_ = self.mu[i] * np.ones(dim), self.sigma[i] * np.ones(dim)
            mvnd_i = tf.contrib.distributions.MultivariateNormalDiag(mu_, sigma_)
            distributions.append(mvnd_i)
        self.mix = tf.contrib.distributions.Mixture(tf.contrib.distributions.Categorical(probs=self.weights), distributions)


    def log_px(self, x):
        x_t = tf.convert_to_tensor(x)
        return self.mix.log_prob(x_t)

    def d_log_px(self, x):
        # d_log_px = 1 / exp(log(sum(exp(log(w_i) + log(p_i(x))))))
        #            * sum(exp(log(w_i) + log(p_i(x)) + log(-(x - mu)/sigma^2)))
        # Use symbolic differentiation instead
        x_t = tf.convert_to_tensor(x)
        return tf.gradients(self.log_px(x_t), [x_t])[0]


def initialise_variables(mu, sigma, n_leaders, n_followers):
    # For replicating Neal
    # followers = tf.reshape(init_distribution.sample(self.n_trials * self.n_followers, seed=123), [self.n_trials, self.n_followers, self.h_dim]
    # leaders = tf.reshape(init_distribution.sample(self.n_trials * self.n_leaders, seed=123), [self.n_trials, self.n_leaders, self.h_dim]

    init_distribution = tf.contrib.distributions.MultivariateNormalDiag(mu * tf.ones(dim, tf.float64), sigma * tf.ones(dim, tf.float64))
    followers = init_distribution.sample(n_followers)
    q_density = init_distribution.log_prob(followers)
    leaders = init_distribution.sample(n_leaders)
    return followers, q_density, leaders


class SteinIS(object):
    def __init__(self, gmm_model, dim, n_leaders, n_followers): # n_trials, step_size=0.01):
        # Required parameters
        self.gmm_model = gmm_model
        self.dim = dim
        self.n_leaders = n_leaders
        self.n_followers = n_followers
        # self.n_trials = n_trials

        # Inputs
        self.A = tf.placeholder(tf.float64, [self.n_leaders, self.dim])
        self.B = tf.placeholder(tf.float64, [self.n_followers, self.dim])
        self.q_density = tf.placeholder(tf.float64, [self.n_followers])
        self.step_size = tf.placeholder(tf.float64, [])

        # Set seed
        # seed = 30
        
        # Register functions for debugging
        # k_A_A, sum_grad_A_k_A_A, A_Squared, h = self.construct_map()
        # k_A_B, sum_grad_A_k_A_B = self.apply_map()
        # A, B, n_q_density = self.svgd_update()
        # # self.q_density = self.density_update()

        # Register functions
        self.construct_leader_map()
        self.construct_follower_map()
        self.svgd_update()

    def construct_leader_map(self):
        # Calculate ||leader - leader'||^2/h, refer to leader as A as in SteinIS
        with tf.variable_scope('k_A_A'):
            x2_A_A_T = 2. * tf.matmul(self.A, tf.transpose(self.A)) #
            self.A_Squared = tf.reduce_sum(tf.square(self.A), keep_dims=True, axis=1)  #
            A_A_Distance_Squared = self.A_Squared - x2_A_A_T + tf.transpose(self.A_Squared) # 100 x 100
            # h_num = tf.square(median(tf.sqrt(A_A_Distance_Squared)))
            h_num = median(A_A_Distance_Squared)
            h_dem = 2. * tf.log(tf.to_double(self.n_leaders) + 1.)
            h = h_num / h_dem
            self.h = h
            self.k_A_A = tf.exp(-A_A_Distance_Squared / h)
        # Can't use vanilla tf.gradients as it sums dy/dx wrt to dx, want sum dy/dx wrt to dy, tf.map_fn is not available
        # tf.gradients also do not provide accurate gradients in this case
        with tf.variable_scope('sum_grad_A_k_A_A'):    
            self.sum_grad_A_k_A_A = tf.reduce_sum([tf.tile(tf.reshape(self.k_A_A[i, :], (-1, 1)), [1, self.dim]) * (-2 * (self.A[i] - self.A) / h) for i in range(self.n_leaders)], axis=0)
        return self.k_A_A, self.sum_grad_A_k_A_A, self.A_Squared, self.h

    def construct_follower_map(self):
        # Calculate ||leader - follower||^2/h, refer to follower as B as in SteinIS
        with tf.variable_scope('k_A_B'):
            x2_A_B_T = 2. * tf.matmul(self.A, tf.transpose(self.B))
            B_Squared = tf.reduce_sum(tf.square(self.B), keep_dims=True, axis=1)
            A_B_Distance_Squared = self.A_Squared - x2_A_B_T + tf.transpose(B_Squared)
            # h_num = tf.square(median(tf.sqrt(A_B_Distance_Squared)))
            # h_num = tf.sqrt(median(A_B_Distance_Squared))
            # h_dem = 2. * tf.log(tf.to_double(self.n_leaders) + 1.)
            # self.h = h_num / h_dem
            self.k_A_B = tf.exp(-A_B_Distance_Squared / self.h)
        with tf.variable_scope('sum_grad_A_k_A_B'):
            self.sum_grad_A_k_A_B = tf.reduce_sum([tf.tile(tf.reshape(self.k_A_B[i, :], (-1, 1)), [1, self.dim]) * (-2 * (self.A[i] - self.B) / self.h) for i in range(self.n_leaders)], axis=0)
        return self.k_A_B, self.sum_grad_A_k_A_B #, self.h

    def svgd_update(self):
        with tf.variable_scope('d_log_pA'):
            d_log_pA = self.gmm_model.d_log_px(self.A)
        with tf.variable_scope('n_A'):
            sum_d_log_pA_T_k_A_A = tf.matmul(self.k_A_A, d_log_pA)
            phi_A = (sum_d_log_pA_T_k_A_A + self.sum_grad_A_k_A_A) / self.n_leaders
            self.n_A = self.A + self.step_size * phi_A
        with tf.variable_scope('n_B'):
            sum_d_log_pA_T_k_A_B = tf.matmul(tf.transpose(self.k_A_B), d_log_pA)
            phi_B = (sum_d_log_pA_T_k_A_B + self.sum_grad_A_k_A_B) / self.n_leaders
            self.n_B = self.B + self.step_size * phi_B
        # See http://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf
        with tf.variable_scope('grad_B_phi_B'):
            sum_grad_B_grad_A_k_A_B = []
            for i in range(self.n_followers):
                x = (self.A - self.B[i]) / self.h
                term_1 = 2 * (tf.reduce_sum(self.k_A_B[:, i]) / self.h) * tf.eye(self.dim, dtype=tf.float64)
                term_2 = 4 * tf.matmul(tf.transpose(x), tf.matmul(tf.diag(self.k_A_B[:, i]), x))
                sum_grad_B_grad_A_k_A_B.append(term_1 - term_2)
            sum_d_log_pA_T_grad_B_k_A_B = tf.stack([tf.matmul(tf.transpose(d_log_pA), tf.matmul(tf.diag(self.k_A_B[:, i]), 2 * (self.A - self.B[i]) / self.h)) for i in range(self.n_followers)])
            # self.sum_d_log_pA_T_grad_A_k_A_B = sum_d_log_pA_T_grad_A_k_A_B
            grad_B_phi_B = (sum_d_log_pA_T_grad_B_k_A_B + tf.stack(sum_grad_B_grad_A_k_A_B)) / self.n_leaders
        with tf.variable_scope('density_update'):
            I = tf.eye(self.dim, dtype=tf.float64)
            inv_abs_det_I_grad_B_phi_B = tf.map_fn(lambda x: 1. / tf.abs(tf.matrix_determinant(I + self.step_size * x)), grad_B_phi_B)
            self.n_q_density = tf.multiply(self.q_density, inv_abs_det_I_grad_B_phi_B)
        return self.n_A, self.n_B, self.n_q_density


# output = '/home/sky/Downloads/stein_is'

# Initialise GMM
# mu = np.array([1., -1.]); sigma = np.sqrt(np.array([0.1, 0.05])); weights = np.array([1./3, 2./3]); dim=6
mu = np.array([[-.5], [.5], [-1.], [1.0], [-1.5], [1.5], [-2.0], [2.0], [-2.5], [2.5]])
sigma = 2 * np.ones(10)
weights = (1 / 10.0 * np.ones(10))
dim = 2
gmm = GMM(mu, sigma, weights, dim)

# Initialise leaders and followers
initial_mu = np.float64(0.)
initial_sigma = np.float64(2.)
n_leaders = 100
n_followers = 100

# Initialise model
model = SteinIS(gmm, dim, n_leaders, n_followers)

iterations = 800

step_size_alpha = np.float64(1.)
step_size_beta = np.float64(0.35)

start = time.time()
with tf.Session() as sess:
    B, q_density, A = sess.run(initialise_variables(initial_mu, initial_sigma, n_leaders, n_followers))

with tf.Session() as sess:
    for i in range(1, iterations+1):
        step_size = step_size_alpha * (1. + i) ** (-step_size_beta)
        # pdb.set_trace()
        A, B, q_density = sess.run([model.n_A, model.n_B, model.n_q_density], feed_dict={model.A: A, model.B: B, model.q_density: q_density, model.step_size: step_size})
        if i % 100 == 0:
            normalisation_constant = np.sum(sess.run(model.gmm_model.log_px(B)) / q_density) / n_followers
            print normalisation_constant

print 'Run took', time.time() - start

# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# run_metadata = tf.RunMetadata()
# tl = timeline.Timeline(run_metadata.step_stats)
# ctf = tl.generate_chrome_trace_format()
# with open(output + '/timeline.json', 'w') as f:
#     f.write(ctf)

# Look into using einsum https://www.tensorflow.org/api_docs/python/tf/einsum
# for i in range(n_followers):
#     if i == 0:
#         grad_B_phi_B = np.dot(w.T, x[i]).reshape((1, 2, 2))
#     else:
#         grad_B_phi_B = np.concatenate((grad_B_phi_B, np.dot(w.T, x[i]).reshape((1, 2, 2))), 0)grad_B_phi_B.shape
