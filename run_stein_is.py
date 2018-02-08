from multiprocessing import Pool, Process, Queue, Event
from stein_is import GMM, SteinIS

import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as ds
import time

import pdb


def stein_is_session(params):
    with tf.Session(config=tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1
    )
    )as sess:
        [n_runs, initial_mu, initial_sigma, n_leaders, n_followers, iterations, step_size_alpha, step_size_beta, sess_num] = params

        print 'Session ' + str(sess_num) + ' has started'

        sess_MSE = []
        sess_start = time.time()
        c_pool = 0
        # log_q_update = np.zeros(n_followers)

        # Initialise GMM
        # Simplest
        # mu = np.array([1.]); sigma = np.sqrt(np.array([2.0])); weights = np.array([1.]); dim = 1
        # Stein IS
        # mu = np.array([[-.5], [.5], [-1.], [1.0], [-1.5], [1.5], [-2.0], [2.0], [-2.5], [2.5]]); sigma = np.sqrt(2) * np.ones(10); weights = (1 / 10.0) * np.ones(10); dim = 2
        # Neal
        mu = np.array([1., -1.]); sigma = np.array([0.1, 0.05]); weights = np.array([1. / 3, 2. / 3]); dim = 6

        # Initialise model
        kernel = 'fisher'
        mm = GMM(mu, sigma, weights, dim)
        model = SteinIS(mm, dim, n_leaders, n_followers, initial_mu, initial_sigma, kernel)
        # writer = tf.summary.FileWriter('Graphs', graph=tf.get_default_graph())

        for run in range(n_runs):
            sess.run(tf.global_variables_initializer())
            run_start = time.time()
            # B, q_density, A = sess.run(initialise_variables(initial_mu, initial_sigma, n_leaders, n_followers, dim))
            for i in range(1, iterations + 1):
                step_size = step_size_alpha * (1. + i) ** (-step_size_beta)
                # Think about dx_log_pA and what it represents, why does it dominate the update term?
                # Can we do something like Adagrad where we maximise log_pA in a 'unit ball' regardless of dimensions?
                _, log_B = sess.run([model.updates, mm.mix.log_prob(model.B)], feed_dict={model.step_size: step_size})
                print log_B
                # _, A, log_pA, dx_log_pA = sess.run([model.updates, model.A, model.log_pA, model.dx_log_pA], feed_dict={model.step_size: step_size})
                # A, A_dw_log_px, A_dmu_log_px, A_dsigma2_log_px, A_dmu_log_px_ = sess.run([model.updates, model.A, model.A_dw_log_px, model.A_dmu_log_px, model.A_dsigma2_log_px, model.A_dmu_log_px_], feed_dict={model.step_size: step_size})
                # pdb.set_trace()
                if i % 800 == 0:
                    normalisation_constant = np.sum(sess.run(tf.exp(model.m_model.log_px(model.B)) / (model.q_density * tf.exp(-model.log_q_update)))) / n_followers
                    print normalisation_constant
                    sess_MSE.append((normalisation_constant - 0.000744) / 0.000744)
                    # sess_MSE.append((normalisation_constant - 1) ** 2)
            run_time = time.time() - run_start
            print 'Run ' + str(run) + ' took ' + str(run_time) + ' seconds'
    print 'Session complete'
    print str(n_runs) + ' runs took ' + str(time.time() - sess_start) + ' seconds'
    return sess_MSE

if __name__ == '__main__':
    # Model parameters
    initial_mu = np.float64(0.)
    initial_sigma = np.sqrt(np.float64(2.))
    n_leaders = 5
    n_followers = 10

    # Hyperparameters
    num_processes = 1
    n_runs = 3
    iterations = 800
    step_size_alpha = np.float64(.0001)
    step_size_beta = np.float64(0.5)

    params = [[n_runs, initial_mu, initial_sigma, n_leaders, n_followers, iterations, step_size_alpha, step_size_beta, i] for i in range(num_processes)]

    # # Multiprocessing
    # start_time = time.time()
    # result = []
    # pool = Pool(processes=num_processes)
    # for i in range(num_processes):
    #     # with tf.device('/gpu:%d' % i):
    #     result.append(pool.apply_async(stein_is_session, [params[i]]))
    # MSE = np.array(([r.get() for r in result])).flatten()
    # pool.close()
    # # Alternative
    # # result = pool.map_async(stein_is_session, params)
    # print(time.time() - start_time)

    # No multiprocessing
    MSE = stein_is_session(params[0])

    # # Save and output results
    # np.save('a_' + str(step_size_alpha) + '_b_' + str(step_size_beta) + '_sig_' + str(initial_sigma) + '_' +  str(num_processes * n_runs) + '_AIS_MSE.npy', MSE)
    print(MSE)
    print(np.mean(MSE))


# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# run_metadata = tf.RunMetadata()
# tl = timeline.Timeline(run_metadata.step_stats)
# ctf = tl.generate_chrome_trace_format()
# with open(output + '/timeline.json', 'w') as f:
#     f.write(ctf)
