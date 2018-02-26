from multiprocessing import Pool, Process, Queue, Event
from stein_is import GMM, SteinIS

import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as ds
import time

import pdb


# Check kernel outputs make sense: They do, but Fisher seems to be less discriminative than SE in adversarial conditions
# Check if Stein conidtions hold: Possibly, much smaller KSD when drawn from actual distribution

def stein_is_process(params):
    # Get params
    [kernel, target, n_runs, initial_mu, initial_sigma, n_leaders, n_followers, iterations, step_size_alpha, step_size_beta, proc_num] = params

    print 'Process ' + str(proc_num) + ' has started'

    proc_start = time.time()
    proc_MSE = []
    proc_MSE_scaled = []
    proc_NC = []
    proc_start = time.time()
    # log_q_update = np.zeros(n_followers)

    # Specify target distribution
    # Simplest
    # mu = np.array([1.]); sigma = np.sqrt(np.array([2.0])); weights = np.array([1.]); dim = 1
    if target == 'stein_is':
        # Stein IS
        mu = np.array([[-.5], [.5], [-1.], [1.0], [-1.5], [1.5], [-2.0], [2.0], [-2.5], [2.5]]); sigma = np.sqrt(2) * np.ones(10); weights = (1 / 10.0) * np.ones(10); dim = 2
    elif target == 'ais':
        # Neal
        mu = np.array([1., -1.]); sigma = np.array([0.1, 0.05]); weights = np.array([1. / 3, 2. / 3]); dim = 6
    elif target == '6d_gauss':
        # 6D Gaussian for sanity check
        mu = np.array([1.]); sigma = np.array([0.1]); weights = np.array([1.]); dim = 6

    # Initialise mixture model
    # model = SteinIS(mm, dim, n_leaders, n_followers, initial_mu, initial_sigma, kernel)
    # writer = tf.summary.FileWriter('Graphs', graph=tf.get_default_graph())

    for run in range(n_runs):
        tf.reset_default_graph()
        # Set seed
        with tf.Session(config=tf.ConfigProto(
                device_count={'CPU':1, 'GPU':0},
                allow_soft_placement=True,
                log_device_placement=False,
                inter_op_parallelism_threads=1,
                intra_op_parallelism_threads=1
                )
            )as sess:
            tf.set_random_seed((proc_num * n_runs) + run)
            # Intialise model after setting random seed to make deterministic
            mm = GMM(mu, sigma, weights, dim)
            model = SteinIS(mm, dim, n_leaders, n_followers, initial_mu, initial_sigma, kernel)
            sess.run(tf.global_variables_initializer())
            run_start = time.time()
            # B, q_density, A = sess.run(initialise_variables(initial_mu, initial_sigma, n_leaders, n_followers, dim))
            # print proc_num, run, '\n', sess.run(model.A)
            # pdb.set_trace()
            for i in range(1, iterations + 1):
                step_size = step_size_alpha * (1. + i) ** (-step_size_beta)
                # Why does dx_log_pA dominate the update term?
                # Can we do something like Adagrad where we maximise log_pA in a 'unit ball' regardless of points?
                _ = sess.run([model.updates], feed_dict={model.step_size: step_size})
                # if i == 1:
                #     pass
                # else:
                #     print log_B - log_B_old
                # log_B_old = log_B
                # _, A, log_pA, dx_log_pA = sess.run([model.updates, model.A, model.log_pA, model.dx_log_pA], feed_dict={model.step_size: step_size})
                # A, A_dw_log_px, A_dmu_log_px, A_dsigma2_log_px, A_dmu_log_px_ = sess.run([model.updates, model.A, model.A_dw_log_px, model.A_dmu_log_px, model.A_dsigma2_log_px, model.A_dmu_log_px_], feed_dict={model.step_size: step_size})
                # pdb.set_trace()
                if i % iterations == 0:
                    normalisation_constant = np.sum(sess.run(tf.exp(model.m_model.log_px(model.B)) / (model.q_density * tf.exp(-model.log_q_update)))) / n_followers
                    proc_NC.append(normalisation_constant)
                    if target == 'stein_is':
                        proc_MSE.append((normalisation_constant - 1) ** 2)
                    elif target == 'ais':
                        proc_MSE_scaled.append((np.abs(normalisation_constant - 0.000744) / 0.000744) ** 2)
                        proc_MSE.append((normalisation_constant - 0.000744) ** 2)
                    elif target == '6d_gauss':
                        proc_MSE_scaled.append((np.abs(normalisation_constant - 0.000248) / 0.000248) ** 2)
                        proc_MSE.append((normalisation_constant - 0.000248) ** 2)
            run_time = time.time() - run_start
            print 'Run ' + str(run) + ' took ' + str(run_time) + ' seconds'
    print 'Process complete'
    print str(n_runs) + ' runs took ' + str(time.time() - proc_start) + ' seconds'
    return proc_MSE, proc_MSE_scaled, proc_NC


if __name__ == '__main__':
    for j in [50, 100, 200]:
        p = {}
        # Model parameters
        p['initial_mu'] = np.float64(0.)
        p['initial_sigma'] = np.sqrt(np.float64(2.))
        p['n_leaders'] = 100
        p['n_followers'] = j

        # Hyperparameters
        p['kernel'] = 'fisher'
        p['target'] = 'stein_is'
        p['n_processes'] = 10
        p['n_runs'] = 10
        p['iterations'] = 800
        p['step_size_alpha'] = np.float64(.001)
        # p['step_size_alpha'] = np.float64(.00005)
        p['step_size_beta'] = np.float64(0.5)

        params = [[p['kernel'], p['target'], p['n_runs'], p['initial_mu'], p['initial_sigma'], p['n_leaders'], p['n_followers'], p['iterations'], p['step_size_alpha'], p['step_size_beta'], i] for i in range(p['n_processes'])]

        # Multiprocessing
        start_time = time.time()
        results = []
        pool = Pool(processes=p['n_processes'])
        for i in range(p['n_processes']):
            # with tf.device('/gpu:%d' % i):
            results.append(pool.apply_async(stein_is_process, [params[i]]))
        pool.close()
        # # Alternative
        # # results = pool.map_async(stein_is_session, params)
        # print(time.time() - start_time)

        for i in range(p['n_processes']):
            if i == 0:
                MSE, MSE_scaled, NC = results[i].get()
            else:
                MSE_t, MSE_scaled_t, NC_t = results[i].get()
                MSE, MSE_scaled, NC = MSE + MSE_t, MSE_scaled + MSE_scaled_t, NC + NC_t

        # # No multiprocessing
        # MSE, MSE_scaled, NC = stein_is_process(params[0])

        # Save and output results
        output = {}
        output['MSE'] = MSE
        output['MSE_scaled'] = MSE_scaled
        output['NC'] = NC
        output['params'] = p
        np.save(str(p['kernel']) + '_' + str(p['target']) + '_a_' + str(p['step_size_alpha']) + '_b_' + str(p['step_size_beta']) + '_l_' + str(p['n_leaders']) + '_f_' + str(p['n_followers']) + '_sig2_' + str(p['initial_sigma'] ** 2) + '_runs_' + str(p['n_runs'] * p['n_processes']) + '.npy', output)
        print 'Done! Total Time: ' + str(time.time() - start_time) + ' seconds'
        print np.mean(MSE), np.mean(MSE_scaled), np.mean(NC)


# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# run_metadata = tf.RunMetadata()
# tl = timeline.Timeline(run_metadata.step_stats)
# ctf = tl.generate_chrome_trace_format()
# with open(output + '/timeline.json', 'w') as f:
#     f.write(ctf)
