import os
import sys
import time
import numpy as np
from tqdm import tqdm
import itertools as itt
import multiprocessing as mp
import sklearn.metrics.pairwise as skmp


# TODO
# How to determine if two neighbours are representative of the same states 
# in the phase space?
# We could, for example, estimate the mutual information between the path 
# to reach the current state and the state of the neighbour.



def _lag_corecurrence(neigh, time_lag):
    """
    Compute the lagged co-recurrence of the binary matrix 'neigh'.
    'time_lag' is the sorted list of time lag of interst
    """
    max_lag = np.array(time_lag).max()

    # define current neigh status
    neigh_now = neigh[:-max_lag, :]                   
    res_dict = {}
    for lag in tqdm(time_lag):
        neigh_lag_nei = np.zeros(neigh_now.shape, dtype=bool)
        # lagging
        neigh_lag_nei[:, lag:] = neigh_now[:, :-lag] 
        # find the analogs of future reference state
        if lag != max_lag:
             # lagging the neighborhood of the reference state
            neigh_lag_ref = neigh[lag:-(max_lag-lag), :]
        else: 
            neigh_lag_ref = neigh[lag:, :]  
        
        # count co-recurrence
        lag_recur = np.logical_and(neigh_lag_ref, neigh_lag_nei).sum(axis=1)
        lag_alpha = lag_recur / neigh_now.sum(axis=1)
        res_dict[lag] = lag_alpha

    # mean estimator
    return res_dict



def _lead_corecurrence(neigh, time_lag):
    """
    Backward in time application 
    """
    min_lag = np.array(time_lag).min()

    # define current neigh status
    neigh_now = neigh[-min_lag:, :]
    res_dict = {}
    print(f'{min_lag}')
    for lag in tqdm(time_lag):
        neigh_lead_nei = np.zeros(neigh_now.shape, dtype=bool)
        # lagging
        neigh_lead_nei[:, :lag] = neigh_now[:, -lag:] 
        # find the analogs of future reference state
        if lag != min_lag:
            # lagging the neighborhood of the reference state
            neigh_lag_ref = neigh[(-min_lag+lag):lag, :] 
        else: 
            neigh_lag_ref = neigh[:lag, :]  
        
        ## count co-recurrence
        lag_recur = np.logical_and(neigh_lag_ref, neigh_lead_nei).sum(axis=1)
        lag_alpha = lag_recur / neigh_now.sum(axis=1)
        res_dict[lag] = lag_alpha

    # mean estimator
    return res_dict



def _lag_corecurrence_continuous(neigh, time_lag):
    """
    Compute the lagged co-recurrence of the binary matrix 'neigh'.
    Requiring trajectories to stick to the reference trajectory. 
    'neigh' is the binary matrix.
    'time_lag' is the sorted list of time lag of interst
    """
    max_lag = np.array(time_lag).max()
    
    # define current neigh status
    neigh_now = neigh[:-max_lag, :]
    
    # track states that never leave the neighbourhood
    cur_recurrence = neigh_now.copy()
    res_dict = {}
    for lag in tqdm(time_lag):
        # find the future states of current analogs
        neigh_lag_nei = np.zeros(neigh_now.shape, dtype=bool)
        # lagging
        neigh_lag_nei[:, lag:] = neigh_now[:, :-lag]

        # find the analogs of future reference state
        if lag != max_lag:
            # lagging the neighborhood of the reference state
            neigh_lag_ref = neigh[lag:-(max_lag-lag), :] 
        else: 
            neigh_lag_ref = neigh[lag:, :]  

        ## compute the overlapping
        lag_overlap = np.logical_and(neigh_lag_ref, neigh_lag_nei)
        bool_matrix = np.zeros(neigh_now.shape, dtype=bool)
        bool_matrix[:,:-lag] = lag_overlap[:,lag:]
        cur_recurrence = np.logical_and(cur_recurrence, bool_matrix)
        
        if lag in time_lag:
            res_dict[lag] = \
                cur_recurrence.copy().sum(axis=1) / neigh_now.sum(axis=1)
            # Why another intersection to get cur_recurrence? Isn't enough to
            # get lag_overlap?
            #res_dict[lag] = np.sum(lag_overlap, axis=1) / neigh_now.sum(axis=1)
        
    return res_dict



def pairwise_matrix(X, filepath, filename, \
    ql = 0.99, n_jobs=1, theiler_len=0, **kwargs):
    """
    Compute the predictability of a collection of states: X
    """
    ## Read kwargs
    metric = kwargs.get("metric")
    if metric is None: metric = "euclidean"
    cross_metric = kwargs.get("cross_metric")
    if cross_metric is None: cross_metric = lambda x,y : np.sqrt(x**2+y**2)
    return_shape = kwargs.get("return_shape")
    return_dist  = kwargs.get("return_dist")
    return_neigh = kwargs.get("return_neigh")
    if return_shape is None: return_shape = False
    if return_dist  is None: return_dist  = False
    if return_neigh is None: return_neigh = False
    
    ## check data format 
    if X.ndim != 2:	print("data is not in correct shape.")

    n_sampleX, n_features = X.shape  

    # initialize distances 
    dist = np.zeros((n_sampleX, n_sampleX))

    # compute pairwise distances (can take time)
    st = time.time()
    print(f'Computing pairwise_distances using {n_jobs} threads...')
    dist[:,:] = skmp.pairwise_distances(X, X, metric=metric, n_jobs=n_jobs)
    print(f'Elapsed time: {time.time()-st} seconds')

    # normalize distances (why?)
    dist[:,:] /= np.linalg.norm(dist[:,:], axis=1).reshape(-1, 1)
    
    # Theiler window
    for i in range(1, 1+theiler_len):       
        np.fill_diagonal(dist[:-i, i:], sys.float_info.max)  
        np.fill_diagonal(dist[i:, :-i], sys.float_info.max)
    dist[~(dist > 0)] = sys.float_info.max
    dist = -np.log(dist)

    # initialize quantile and neighbours index
    q     = np.zeros((n_sampleX)) + np.nan
    neigh = np.zeros((n_sampleX, n_sampleX), dtype = bool)

    # last 'time_lag' states do not have this index
    q = np.quantile(dist, ql, axis=1)   

    # neighbors: 1; non-neighbors: 0
    neigh = dist > q.reshape(-1,1)		

    #np.save(f'./pairwise_matrix/pw_analogs_{name}_{ql}_{theiler_len}.npy', neigh)
    with open(f'{filepath}/{filename}_pairwise_{ql}_{theiler_len}.npy','wb') as f:
        np.save(f, neigh)

    return neigh



