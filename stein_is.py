import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow.contrib.distributions as ds
# import pdb


def median(x):
    x = tf.reshape(x, [-1])
    med = tf.floordiv(tf.shape(x)[0], 2)
    check_parity = tf.equal(tf.to_float(med), tf.divide(tf.to_float(tf.shape(x)[0]), 2.))
    def is_true():
        return tf.reduce_sum(tf.nn.top_k(x, med+1).values[-2:]) / 2.
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
        
    def log_px(self, x):
        # log_px = log(sum(exp(log(w_i) + log(p_i(x)))))
        log_px = []
        for i in range(weights.shape[0]):
            mu_, sigma_ = self.mu[i] * tf.ones(dim), self.sigma[i] * tf.ones(dim)
            mvn = ds.MultivariateNormalDiag(loc=mu_, scale_diag=sigma_)
            # Calculate log_px for each component
            log_px_i = tf.reduce_logsumexp(mvn.log_prob(x)) + tf.log(tf.to_float(weights[i]))
            log_px.append(log_px_i)
        return tf.reduce_logsumexp(log_px)
    
    def d_log_px(self, x):
        # d_log_px = 1 / exp(log(sum(exp(log(w_i) + log(p_i(x)))))) 
        #            * sum(exp(log(w_i) + log(p_i(x)) + log(-(x - mu)/sigma^2)))
        # Use symbolic differentiation instead
        log_px = self.log_px(x)
        return tf.gradients(log_px, [x])


class SteinIS(object):
    def __init__(self, gmm_model, mu, sigma, dim, n_leaders, n_followers, step_size_master=1., step_size_beta=0.35): # n_trials, step_size=0.01):
        # Required parameters
        self.gmm_model = gmm_model
        self.mu = mu
        self.sigma = sigma
        self.dim = dim
        self.n_leaders = n_leaders
        self.n_followers = n_followers
        # self.n_trials = n_trials
        self.step_size_master = step_size_master
        self.step_size_beta = step_size_beta
        self.eps = 1e-10
        
        # Set seed
        seed = 30
        
        # Intialisation
        self.B, self.B_density, self.A = self.initialise_variables()
        self.pB = self.gmm_model.log_px(self.B)
        
        # Register functions for debugging
        # self.k_A_A, self.sum_grad_A_k_A_A, self.A_Squared, self.h_0 = self.construct_map()
        # self.k_A_B, self.sum_grad_A_k_A_B, self.grad_A_grad_B_k_A_B, self.grad_B_k_A_B = self.apply_map()        
        # self.A, self.B, self.phi_B, self.grad_B_phi_B = self.svgd_update()
        # self.q_density = self.density_update()
        
        
    def initialise_variables(self):
        init_distribution = tf.contrib.distributions.MultivariateNormalDiag(self.mu * tf.ones(dim), self.sigma * tf.ones(dim))
        
        # followers = tf.reshape(init_distribution.sample(self.n_trials * self.n_followers, seed=123), [self.n_trials, self.n_followers, self.h_dim] 
        # leaders = tf.reshape(init_distribution.sample(self.n_trials * self.n_leaders, seed=123), [self.n_trials, self.n_leaders, self.h_dim] 
        
        followers = tf.reshape(init_distribution.sample(self.n_followers, seed=123), [self.n_followers, self.dim]) 
        q_density = init_distribution.log_prob(followers)
        leaders = tf.reshape(init_distribution.sample(self.n_leaders, seed=123), [self.n_leaders, self.dim])
                           
        return followers, q_density, leaders
                             
    def construct_map(self):
        # Calculate ||leader - leader'||^2/h_0, refer to leader as A as in SteinIS
        x2_A_A_T = tf.multiply(2., tf.matmul(self.A, tf.transpose(self.A)))
        A_Squared = tf.reduce_sum(tf.square(self.A), 1)
        A_Distance = tf.add(tf.subtract(A_Squared, x2_A_A_T), tf.transpose(A_Squared))   
        # h_0 = tf.divide(tf.add(median(A_Distance), self.eps), 2. * (tf.log(tf.cast(self.n_leaders, tf.float32)) + 1.))
        h_0 = tf.divide(median(A_Distance), 2. * (tf.log(tf.to_float(self.n_leaders)) + 1.))
        k_A_A = tf.exp(-tf.div(A_Distance, tf.square(h_0)))
        sum_grad_A_k_A_A = tf.reduce_sum(tf.gradients(k_A_A, [self.A]), 1)
        return k_A_A, sum_grad_A_k_A_A, A_Squared, h_0
    
    def apply_map(self):
        # Calculate ||leader - follower||^2/h_0, refer to follower as B as in SteinIS
        x2_A_B_T = tf.multiply(2., tf.matmul(self.A, tf.transpose(self.B)))
        B_Squared = tf.reduce_sum(tf.square(self.B), 1)
        A_B_Distance  = tf.add(tf.subtract(self.A_Squared, x2_A_B_T), B_Squared)
        k_A_B = tf.exp(-tf.div(A_B_Distance, tf.square(self.h_0)))
        sum_grad_A_k_A_B = tf.reduce_sum(tf.gradients(k_A_B, [self.A]), 1)
        return k_A_B, sum_grad_A_k_A_B 
                    
    def svgd_update(self):
        self.k_A_A, self.sum_grad_A_k_A_A, self.A_Squared, self.h_0 = self.construct_map()
        self.k_A_B, self.sum_grad_A_k_A_B = self.apply_map()
        self.d_log_pA = self.gmm_model.d_log_px(self.A)[0]
        sum_d_log_pA_T_k_A_A = tf.reduce_sum(tf.matmul(self.k_A_A, self.d_log_pA), 0)       
        phi_A = (1. / tf.to_float(self.n_leaders)) * tf.add(sum_d_log_pA_T_k_A_A, self.sum_grad_A_k_A_A)
        A = tf.add(self.A, self.step_size * phi_A)  
        sum_d_log_pA_T_k_A_B = tf.reduce_sum(tf.matmul(self.k_A_B, self.d_log_pA), 0)       
        phi_B = (1. / tf.to_float(self.n_leaders)) * tf.add(sum_d_log_pA_T_k_A_B, self.sum_grad_A_k_A_B)
        B = tf.add(self.B, self.step_size * phi_B) 
        grad_B_phi_B = tf.gradients(phi_B, [self.B])
        return A, B, phi_B, grad_B_phi_B[0]
    
    def density_update(self):
        I = tf.eye(self.dim)
        inv_abs_det_I_grad_B_phi = tf.map_fn(lambda x: 1./tf.abs(tf.matrix_determinant(tf.add(I, x))), self.grad_B_phi_B)
        return tf.multiply(self.B_density, inv_abs_det_I_grad_B_phi) 

    def main(self, iteration):
        for i in range(1, iteration+1):
            self.step_size = self.step_size_master * (1. + i) ** (-self.step_size_beta)
            self.A, self.B, self.phi_B, self.grad_B_phi_B = self.svgd_update()
            self.q_density = self.density_update()
            if i % 20 == 0:
                self.pB = self.gmm_model.log_px(self.B)
                self.importance_weights = tf.divide(self.q_density, self.pB)
                self.normalisation_constant = 1. / tf.to_float(self.n_followers) * tf.reduce_sum(self.importance_weights)
                print 'Iteration ', str(i), ' done'
        self.final_B = self.B
        return self.normalisation_constant
#         return self.final_B, self.importance_weights, self.normalisation_constant
#         return self.A, self.B

# Main
sess = tf.Session()

# GMM parameters
# # AIS
# mu = np.array([1., -1.])
# sigma = np.sqrt(np.array([0.1, 0.05]))
# weights = np.array([1./3, 2./3])
# dim = 6

# Stein IS
mu = np.array([[-.5], [.5], [-1.], [1.0], [-1.5], [1.5], [-2.0], [2.0], [-2.5], [2.5]]).astype(np.float32)
sigma = 2 * np.ones(10).astype(np.float32)
weights = (1/10.0 * np.ones(10)).astype(np.float32)
dim = 2

# Instantiate GMM
gmm = GMM(mu, sigma, weights, dim)

# Stein IS parameters
initial_mu = np.float32(0.)
initial_sigma = np.float32(2.)
n_leaders = 100
n_followers = 100
model = SteinIS(gmm, initial_mu, initial_sigma, dim, n_leaders, n_followers)

# Verify performance, know that partition function for specified GMM  = 1
sess.run([model.main(100)])





