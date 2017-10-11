from __future__ import print_function
from multiprocessing import Pool, Process, Queue, Event
from stein_is import initialise_variables, GMM, SteinIS

import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as ds
import time

import pdb


# class RunSteinIS(object):
#     def __init__(self, n_runs, initial_mu, initial_sigma, n_leaders, n_followers, iterations, step_size_alpha, step_size_beta):
#         # Collection
#         MSE = []
#         total_run_time = 0
        
#         # Initialisation
#         num_processes = 4
#         c_runs = 0
        
#         # Model parameters
#         initial_mu = initial_mu
#         initial_sigma = initial_sigma
#         n_leaders = n_leaders
#         n_followers = n_followers
        
#         # Hyperparameters
#         n_runs = n_runs
#         iterations = iterations
#         step_size_alpha = step_size_alpha
#         step_size_beta = step_size_beta

def stein_is_session(params):
    with tf.Session(config=tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1
    )
    )as sess: 
        [n_runs, initial_mu, initial_sigma, n_leaders, n_followers, iterations, step_size_alpha, step_size_beta, sess_num] = params
        print('Session', str(sess_num), 'has started')
        
        sess_MSE = []
        # sess_start = time.time()
        c_pool = 0    
        log_q_update = np.zeros(n_followers)

        # Initialise GMM
        mu = np.array([1., -1.]); sigma = np.array([0.1, 0.05]); weights = np.array([1. / 3, 2. / 3]); dim = 6
        # mu = np.array([1.]); sigma = np.sqrt(np.array([2.0])); weights = np.array([1.]); dim = 1
        # mu = np.array([[-.5], [.5], [-1.], [1.0], [-1.5], [1.5], [-2.0], [2.0], [-2.5], [2.5]]); sigma = np.sqrt(2) * np.ones(10); weights = (1 / 10.0 * np.ones(10)); dim = 2
        gmm = GMM(mu, sigma, weights, dim)

        # Initialise model
        model = SteinIS(gmm, dim, n_leaders, n_followers)
        # writer = tf.summary.FileWriter('Graphs', graph=tf.get_default_graph())

        for _ in range(n_runs):
            # run_start = time.time()
            B, q_density, A = sess.run(initialise_variables(initial_mu, initial_sigma, n_leaders, n_followers, dim))
            for i in range(1, iterations + 1):
                step_size = step_size_alpha * (1. + i) ** (-step_size_beta)
                # pdb.set_trace()
                A, B, log_q_update, k_A_A, k_A_B, log_pA, d_log_pA = sess.run([model.n_A, model.n_B, model.n_log_q_update, model.k_A_A, model.k_A_B, model.log_pA, model.d_log_pA], feed_dict={model.A: A, model.B: B, model.log_q_update: log_q_update, model.step_size: step_size})
                # pdb.set_trace()
                # A_ = A
                # B_ = B
                # log_q_update_ = log_q_update
                # k_A_A_ = k_A_A
                # k_A_B_ = k_A_B
                # d_log_pA_ = d_log_pA
                if i % 800 == 0:
                    normalisation_constant = np.sum(sess.run(tf.exp(model.gmm_model.log_px(B))) / (q_density * np.exp(-log_q_update))) / n_followers
                    sess_MSE.append((normalisation_constant - 0.000744) ** 2)
                # run_time = time.time() - run_start
            # print('Run', str(_), 'took', str(run_time))
    print('Session complete')
    return sess_MSE
# print(str(n_runs), 'runs took', str(time.time() - sess_start))


if __name__ == '__main__':
    # Model parameters
    initial_mu = np.float64(0.)
    initial_sigma = np.sqrt(np.float64(2.))
    n_leaders = 100
    n_followers = 100
    
    # Hyperparameters
    num_processes = 4
    n_runs = 3
    iterations = 800
    step_size_alpha = np.float64(.005)
    step_size_beta = np.float64(0.5)
    
    params = [[n_runs, initial_mu, initial_sigma, n_leaders, n_followers, iterations, step_size_alpha, step_size_beta, i] for i in range(num_processes)]    
    
    # Multiprocessing
    start_time = time.time()
    result = []
    pool = Pool(processes=num_processes)
    for i in range(num_processes):
        with tf.device('/gpu:%d' % i):
            result.append(pool.apply_async(stein_is_session, [params[i]]))
    MSE = np.array(([r.get() for r in result])).flatten()
    pool.close()
    # Alternative
    # result = pool.map_async(stein_is_session, params)
    print(time.time() - start_time)    
    
    # for gpu_id in range(num_processes):
    #     with tf.device('/gpu:%d' % gpu_id):
    #         with tf.name_scope('tower_%d' % gpu_id):
    #             pool.apply_async(stein_is_session)

    # # Save and output results
    np.save('a_' + str(step_size_alpha) + '_b_' + str(step_size_beta) + '_sig_' + str(initial_sigma) + '_' +  str(num_processes * n_runs) + '_AIS_MSE.npy', MSE)
    print(MSE)
    print(np.mean(MSE))


# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# run_metadata = tf.RunMetadata()
# tl = timeline.Timeline(run_metadata.step_stats)
# ctf = tl.generate_chrome_trace_format()
# with open(output + '/timeline.json', 'w') as f:
#     f.write(ctf)