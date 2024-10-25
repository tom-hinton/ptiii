import os
import sys
import time
import numpy as np
import xarray as xr
from tqdm import tqdm

import scipy.stats as sc
import statsmodels.api as sm
import multiprocessing as mp
import sklearn.metrics.pairwise as skmp
import pickle
import logging
logging.captureWarnings(True)



def compute_exceeds(X, filepath, filename, 
                    ql = 0.99, n_jobs=1, theiler_len=0, p_value_exp=None,
                    exp_test_method='anderson', save_full=False, **kwargs):
    """
    Compute the predictability of a collection of states: X
    save_full: save the full matrix of exceedances (True) or only the indices (False)
    """
    ## Read kwargs
    metric = kwargs.get("metric")
    if metric is None: metric = "euclidean"
    
    ## check data format 
    if X.ndim != 2:	print("data is not in correct shape.")

    n_samples, n_features = X.shape  

    # initialize distances 
    dist = np.zeros((n_samples, n_samples))

    # compute pairwise distances (can take time)
    st = time.time()
    print(f'Computing pairwise_distances using {n_jobs} threads...')
    dist[:,:] = skmp.pairwise_distances(X, X, metric=metric, n_jobs=n_jobs)
    print(f'Elapsed time: {time.time()-st} seconds')

    # Theiler window
    for i in range(1, 1+theiler_len):       
        np.fill_diagonal(dist[:-i, i:], sys.float_info.max)  
        np.fill_diagonal(dist[i:, :-i], sys.float_info.max)
    np.fill_diagonal(dist, sys.float_info.max)
    if save_full:
        fname = f'{filepath}/{filename}_dist_{ql}_{theiler_len}.npy'
        np.save(fname, dist)
        print(f'Saved distance matrix to {fname}')
    else:
        print('Distance matrix not saved.')

    dist_log = -np.log(dist)
    # initialize quantile and neighbours index
    q     		 = np.zeros((n_samples)) + np.nan
    exceeds_bool = np.zeros((n_samples, n_samples), dtype = bool)

    ## calc quantile as with mquantiles with alphap=0.5 and betap=0.5
    # last 'time_lag' states do not have this index
    q = np.quantile(dist_log, ql, axis=1, interpolation='midpoint')

    # neighbors: 1; non-neighbors: 0
    # n_neigh = int(np.round(n_samples*(1-ql))) #
    exceeds_bool = dist_log > q.reshape(-1,1)
    n_neigh = np.sum(exceeds_bool, axis=1).min()
    exceeds_bool = _correct_n_neigh(exceeds_bool, dist_log, q, n_neigh)

    print(exceeds_bool.shape)
    exceeds_idx = np.argwhere(exceeds_bool).reshape(n_samples,n_neigh,2)[:,:,1]
    row_idx = np.arange(n_samples)[:,None]
    exceeds = dist_log[row_idx,exceeds_idx] - q[:, None]

    if p_value_exp is not None:
        exp_stat = np.zeros([n_samples,])
        for i in range(n_samples):
            exp_stat[i] = _exp_test(
                exceeds=exceeds[i,:], p_value=p_value_exp,
                exp_test=exp_test_method)
        
        # number of points for which we cannot reject the null hypothesis 
        # for which the exceedences are exponentially distributed
        n_exp = np.sum(exp_stat>=0)
        print('')
        print(f'H0: exceedances are exponentially distributed')
        print(f'H0 rejected for {n_samples-n_exp} out of {n_samples} points '\
                f'at a significance level of {p_value_exp}')
        print(f'Statistical test used: {exp_test_method}')
    else:
        exp_stat = []

    # save matrix
    fname = f'{filepath}/{filename}_exceeds_idx_{ql}_{theiler_len}.npy'
    np.save(fname, exceeds_idx)
    fname = f'{filepath}/{filename}_exceeds_{ql}_{theiler_len}.npy'
    np.save(fname, exceeds)
    fname = f'{filepath}/{filename}_exp_stat_{ql}_{theiler_len}.npy'
    np.save(fname, exp_stat)

    return dist, exceeds, exceeds_idx, exceeds_bool, exp_stat



def compute_d1(exceeds, filepath, filename, ql=0.99, theiler_len=0):
    """
    Compute the local dimension d1. 
    """
    d1 = 1 / np.mean(exceeds, axis=1)
    # save d1
    fname = f'{filepath}/{filename}_d1_{ql}_{theiler_len}.npy'
    np.save(fname, d1)
    return d1



def compute_theta(idx, filepath, filename, ql=0.99, theiler_len=0,
                    method='sueveges'):
    """
    Compute the extremal index \theta, also known as inverse persistence. 
    """
    def _calc_sueveges(idx, ql=0.99):
        q = 1 - ql
        # import pdb
        # pdb.set_trace()
        Ti = idx[:,1:] - idx[:,:-1]
        Si = Ti - 1
        Nc = np.sum(Si > 0, axis=1)
        K  = np.sum(q * Si, axis=1)
        N  = Ti.shape[1]
        theta = (K + N + Nc - np.sqrt((K + N + Nc)**2 - 8 * Nc * K)) / (2 * K)
        return theta
    
    def _calc_ferro(idx):
            Ti = idx[:,1:] - idx[:,:-1]
            theta = 2 * (np.sum(Ti, axis=1)**2) / ((Ti.shape[1] - 1) * \
                        np.sum(Ti**2, axis=1))
            theta2 = 2 * (np.sum(Ti - 1, axis=1)**2) / ((Ti.shape[1] - 1) * \
                        np.sum((Ti - 1) * (Ti - 2), axis=1))
            idx_Timax_larger_than_2 = np.max(Ti, axis=1) <= 2
            theta[idx_Timax_larger_than_2] = theta2[idx_Timax_larger_than_2]
            theta[theta>1] = 1
            return theta
    if method == 'sueveges':
        theta = _calc_sueveges(idx, ql)
    elif method == 'ferro':
        theta = _calc_ferro(idx)
    else:
        print(f'Method {method} to compute theta not recognized.')
        print('Using default Sueveges method.')
        theta = _calc_sueveges(idx, ql)
    
    # save theta
    fname = f'{filepath}/{filename}_theta_{ql}_{theiler_len}.npy'
    np.save(fname, theta)
    return theta



def compute_alphat(dist, exceeds_bool, filepath, filename, time_lag, 
                   ql=0.99, theiler_len=0, l=1):
    """
    Compute the lagged co-recurrence of the binary matrix 'neigh'.
    Requiring trajectories to stick to the reference trajectory. 
    'exceeds_idx' is the matrix of neighbour indices (i.e., exceedances).
    'time_lag' is the sorted list of time lag of interest
    """
    # find neighbour indices of the state of interest now
    exceeds_bool_now = exceeds_bool.copy()
    
    n_samples = exceeds_bool.shape[0]   # number of states
    n_neigh = np.sum(exceeds_bool[0,:]) # number of neighbours
    alphat_dict = {}

    for lag in tqdm(time_lag):
        # find forward recurrences shifting by lag exceeds_bool_now
        exceeds_bool_lag = np.zeros([n_samples-lag, n_samples], dtype=bool)
        exceeds_bool_lag[:, lag:] = exceeds_bool_now[:-lag, :-lag]

        # find forward-reference-state recurrences
        exceeds_bool_lag_ref = exceeds_bool_now[lag:, :]
        
        # compute the intersection of forward recurrences and
        # forward-reference-state recurrences and use a flag of -1 where there
        # is no intersection
        exceeds_bool_intersect = np.logical_and(exceeds_bool_lag_ref, exceeds_bool_lag)
        
        if l == 0:
            alphat_dict[lag] = np.sum(exceeds_bool_intersect, axis=1) / n_neigh
        else:
            dist_sum_in = np.nansum(np.where(exceeds_bool_intersect, dist[lag:, :], np.nan) ** l, axis=1) # Neighbors sticking
            dist_sum_all = np.nansum(np.where(exceeds_bool_lag, dist[lag:, :], np.nan) ** l, axis=1)      # All forward neighbors
            alphat_dict[lag] = dist_sum_in / dist_sum_all

    # save alphat_dict
    with open(f'{filepath}/{filename}_alphat_max{np.array(time_lag).max()}_{ql}_{theiler_len}_{l}.pkl', 'wb') as f:
        pickle.dump(alphat_dict, f)
            
        # save alphat_dict seperately
        # for key in alphat_dict.keys():
        #     fname = f'{filepath}/{filename}_alphat{key}_{ql}_{theiler_len}.txt'
        #     with open(fname, 'wb') as f:
        #         np.savetxt(f, alphat_dict[key])

    # return alphat_dict



def create_bool_from_idx(exceeds_idx):
    """
    Create a boolean mask matrix from an index matrix without using for loops.
    
    Parameters:
    - exceeds_idx (ndarray): A matrix where each row contains column indices
        to be marked as True.
    - num_cols (int): The number of columns in the boolean mask matrix.
    
    Returns:
    - mask_matrix (ndarray): The boolean mask matrix.
    """
    n_samples = exceeds_idx.shape[0]
    exceeds_bool = np.zeros((n_samples, n_samples), dtype=bool)
    row_indices = np.arange(n_samples)[:, None]
    exceeds_bool[row_indices, exceeds_idx] = True
    return exceeds_bool


def _exp_test(exceeds, p_value, exp_test):
    ''''
    Test if `exceeds` follow an exponential distribution.

    Parameters:
    - exceeds (1d array): exceedances array
    - p_value (float)   : significance level to test
    - exp_test (string) : 'anderson' (default) or 'chi2'

    Returns:
    - res_stat (float): The residual between the calculated statistic
    and the reference value used to check if H0 needs to be rejected or not.
    Positive value: do not reject H0 (i.e., we cannot reject the null hypothesis
    for which the data are exponentially distributed).
    
    '''
    if exp_test=='anderson':
        if p_value==0.15:
            ind_p_value_anderson = 0
        elif p_value==0.1:
            ind_p_value_anderson = 1
        elif p_value==0.05:
            ind_p_value_anderson = 2
        elif p_value==0.025:
            ind_p_value_anderson = 3
        elif p_value==0.01:
            ind_p_value_anderson = 4
        else:
            raise ValueError(
                'p_value must be one of the following values: ',
                '0.15'' 0.10, 0.05, 0.025, 0.01')

        ## perform anderson test
        anderson_stat, anderson_crit_val, anderson_sig_lev = \
            sc.anderson(exceeds, dist='expon')
        ref = anderson_crit_val[ind_p_value_anderson]
        # reject H0 if anderson_stat > ref
        # i.e., if res_stat < 0
        res_stat = 100*(ref - anderson_stat) / ref

    elif exp_test=='chi2':
        pplot = sm.ProbPlot(exceeds, sc.expon)
        xq = pplot.theoretical_quantiles
        yq = pplot.sample_quantiles
        p_fit = np.polyfit(xq, yq, 1)
        yfit = p_fit[0] * xq + p_fit[1]

        ## perform Chi-Square Goodness of Fit Test
        _, p_chi2 = sc.chisquare(f_obs=yq, f_exp=yfit)
        # reject H0 if p_value > p_chi2
        # i.e., if res_stat < 0
        res_stat = 100*(p_chi2 - p_value) / p_value
    return res_stat


def _correct_n_neigh(exceeds_bool, dist, q, n_neigh):
    '''
    Correct the matrix `exceeds_bool` in case some neighbours fall on the edge
    of the ball.
    '''
    exceeds_bool_geq = dist >= q.reshape(-1,1)
    print(f'Number selected: {n_neigh}')
    idx_edge = np.where(np.sum(exceeds_bool, axis=1)!=n_neigh)[0]
    for i in idx_edge:
        idx_add = np.where(exceeds_bool[i,:]!=exceeds_bool_geq[i,:])[0][0]
        exceeds_bool[i,idx_add] = True
    
    return exceeds_bool




### TODO autocorrelation and/or self mutual information for Theiler window

#######################
### Autocorrelation ###
#######################
# from statsmodels.tsa.stattools import acf

# acorr = acf(x, nlags=200)
# n_tau = np.shape(acorr)[0]

# plt.figure()
# plt.plot(range(n_tau), acorr, '-k')
# ind_acorr_min = np.where(np.diff(acorr)>=0)[0][0]
# ind_acorr0 = np.where(acorr<=0)[0][0]
# plt.plot(ind_acorr_min, acorr[ind_acorr_min],'.', color='orange',
#          markersize=markersize)
# plt.plot(ind_acorr0, acorr[ind_acorr0],'.', color='r', markersize=markersize)
# plt.xlabel('Time lag (steps)', fontsize=fontsize)
# plt.ylabel('Autocorrelation', fontsize=fontsize)

# print(ind_acorr_min)
# print(ind_acorr0)

###############################
### Self mutual information ###
###############################
# from sklearn.feature_selection import mutual_info_regression

# n_tau = 50

# def calc_smi_min(x, n_tau):
#     smi = np.zeros([n_tau,1])
#     smi[0] = mutual_info_regression(x[:,np.newaxis],x[:])
#     for i in range(1,n_tau):
#         smi[i,:] = mutual_info_regression(x[i:,np.newaxis],x[:-i])

#     ind_smi_min = np.where(np.diff(smi[:,0])>=0)[0][0]
#     return [smi, ind_smi_min]

# [smi, ind_smi_min] = calc_smi_min(x, n_tau)

# plt.figure()
# plt.plot(range(len(smi)), smi, '-k')
# plt.plot(ind_smi_min, smi[ind_smi_min],'.', color='orange',
#          markersize=markersize)
# plt.xlabel('Time lag (steps)', fontsize=fontsize)
# plt.ylabel('Self-MI', fontsize=fontsize)

# print(ind_smi_min)