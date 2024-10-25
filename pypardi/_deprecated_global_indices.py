import os
import sys
import time
import warnings
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors



def _calc_tangent_map(
	X, t_step=1, n_neighbors=20, eps_over_L0=0.05,
	eps_over_L_fact=1.2, verbose=False):
	"""
 	This function calculates an approximation
	of the tangent map in the phase space.
	"""
	eps_over_L = eps_over_L0

	# Find number of epochs Nt and dimension m of X
	t = X[:,0]
	Y = X[:,1:]
	Nt, m, _ = Y.shape

	# Find horizontal extent of the attractor
	L = np.max(Y[:,-1]) - np.min(Y[:,-1])
	flag_calc_map = True
	while flag_calc_map == True:

		# Set epsilon threshold
		eps = eps_over_L * L
		if verbose == True: print("eps_over_L = %f" %(eps_over_L))

		# Find first n_neighbors nearest neighbors
		# to each element X[tt,:]. The number of
		# neighbors is n_neighbors + 1 (because
		# the n_neighbors distances are calculated
		# also from the point itself and the distance
		# 0 needs to be esxcluded).
		nbrs = NearestNeighbors(
			n_neighbors=n_neighbors+1,
			algorithm='ball_tree').fit(Y[:-t_step,:,0])

		# Find the distances and the indeces
		# of the nearest neighbors
		distances, indices = nbrs.kneighbors(Y[:-t_step,:,0])

		# Find where the distances of the neighbours
		# are larger than the eps threshold
		ii = np.where(distances > eps)
		if len(ii[0]) > 0:
			eps_over_L = eps_over_L * eps_over_L_fact
		else:
			flag_calc_map = False

	# If n_neighbors_min is lower than the minimum
	# number of points required to estimate the
	# tangent map (i.e., lower than the dimension
	# of X), use as minimum number of neighbors
	# the minimum number necessary to calculate
	# the tangent map
	if n_neighbors < m: n_neighbors = m

	# Initialize the tangent map matrix A at each
	# epoch tt (if at time tt only n<n_neighbors
	# neighbors have a distance smaller than
	# eps, then retain only n neighbors).
	A = np.empty((Nt - t_step,m,m))

	# For every time step...
	for tt in tqdm(np.arange(Nt - t_step), desc="tangent map [tt]"):

		# The point under exam is X[tt,:]
		x0_nn = Y[tt,:,0]

		# and it moves in X[tt+t_step,:] after t_step
		x0_nn1 = Y[tt+t_step,:,0]

		# Create the variables containing the neighbors
		# at time tt (xneigh_nn) and their evolution
		# after t_step (xneigh_nn1)
		xneigh_nn  = Y[indices[tt],:,0]
		xneigh_nn1 = Y[indices[tt]+t_step,:,0]

		# Calculate the distances of the neighbors
		# from the point under exam (exclude the
		# first element of xneigh_nn because it
		# is equal to x0_nn)
		y = xneigh_nn[1:] - x0_nn

		# Calculate the distances of the neighbors' evolution from
		# the evolution in time of the point under exam (exclude the
		# first element of xneigh_nn1 because it is equal to x0_nn1)
		z = xneigh_nn1[1:] - x0_nn1
  
		# Calculate the tangent map A at time
		# tt using the pseudo-inverse of y.T
		A[tt,:,:] = np.dot(z.T, np.linalg.pinv(y.T))

	# Return the tangent map A
	return A, eps_over_L



def compute_lyapunov_spectrum(
	X, dt=1, t_step=1, n_neighbors=20, eps_over_L0=0.05, 
 	eps_over_L_fact=1.2, sampling=['rand',100], n=1000, method="SS85", 
  	verbose=False, flag_calc_tangent_map=True, A=None):
	"""
	Calculate Lyapunov spectrum.
 
	Parameters
	----------
		* X: Input data. Format: Nt x m, with Nt  
  			number of epochs, m number of time series.
		* t_step: time step. The default is 1.
		* n_neighbors_min: Minimum number of neighbors 
  			to use to calculate the tangent map. 
   			The default is 0, which uses m for n_neighbors_min.
		* n_neighbors_max: The default is 20.
		* eps_over_L0: Starting value for the distance to look up 
			for neighbors, expressed as a fraction of the attractor
			size L. The default is 0.05.
		* eps_over_L_fact: Factor to increase the size of the neighborhood
			if not enough neighbors were found. The default is 1.2.
		* sampling: to decide which points to use for the Lyapunov 
  			spectrum estimation. Options:
	   		- ['all', None]: Use all the possible trajectories.
	   		- ['begin', int]: Start from the beginning of the
	   		                time series and take a new trajectory
							after int steps.
	   		- ['mid', int]: Start from the middle of the time
	   		              	series and take a new trajectory
							after int steps.
	   		- ['rand', None]: Start from allowed random times.
	Returns
	-------
		- A: Tangent map approximation.
		- eps_over_L: Final value for the distance to look up for neighbors,
			expressed as a fraction of the attractor size L, such
			that there are at least n_neighbors for the calculation
			of the tangent map at each epoch.
	"""
	# Find number of epochs Nt and dimension m of X
	Nt, m, _ = X[:,1:].shape

	if method == "SS85":
		# Find horizontal extent of the attractor
		L = np.max(X[:,-1]) - np.min(X[:,-1])
		if flag_calc_tangent_map == True:
			if verbose == True:
				tic = time.time()
				print("")
				print("Calculating tangent map: ", end='')

			A, eps_over_L = _calc_tangent_map(
				X, t_step=t_step,
				n_neighbors=n_neighbors,
				eps_over_L0=eps_over_L0,
				eps_over_L_fact=eps_over_L_fact
			)

			if verbose == True:
				print("eps_over_L = %f  %.2f s" %(eps_over_L, time.time()-tic))
		else:
			if A == None:
				raise ValueError('Tangent map is missing.')

		nbrs = NearestNeighbors(
			n_neighbors = n_neighbors + 1,
			algorithm='ball_tree').fit(X[:-t_step,:,0])
		distances, indices = nbrs.kneighbors(X[:,:,0])
		print(f"{A.shape=:}")
		if sampling[0] == 'all':
			if sampling[1] == None:
				ts = np.arange(0, Nt-n*t_step, 1)
			else:
				raise ValueError(
					'When sampling[0] is ''all'', sampling[1] must be None')
		elif sampling[0] == 'mid':
			ts = np.arange(int(Nt/2), Nt - n * t_step,sampling[1])
		elif sampling[0] == 'begin':
			ts = np.arange(0,Nt-n*t_step,sampling[1])
		elif sampling[0] == 'rand':
			Nles_statistic = sampling[1]
			ts = np.sort(np.array([int(ii) for ii in \
					np.floor(np.random.rand(Nles_statistic) * \
						(Nt-n*t_step))])
					)
		else:
			raise ValueError('sampling[0] not valid.')

		Nles_statistic = len(ts)
		logR = np.zeros((Nles_statistic,n,m))
		les = np.empty((Nles_statistic,n,m))
		kk = -1
		if verbose == True:
			print("")
			print("Calculating Lyapunov spectrum")
		les_mean = np.zeros((n,m))
		les_std  = np.zeros((n,m))
		for t0 in tqdm(ts):
			kk += 1
			ind2follow = indices[t0,:]
			distances_ind2follow = distances[t0,:]
			ind2rm = np.where(ind2follow+(t0+n*t_step)>Nt)[0]
			ind2follow = np.delete(ind2follow, ind2rm)
			distances_ind2follow = np.delete(distances_ind2follow, ind2rm)
			jj2rm = distances_ind2follow>(eps_over_L * L)
			ind2follow = np.delete(ind2follow, jj2rm)
			ind2follow = ind2follow[1:]
			e = np.eye(m)
			for nn in np.arange(n):
				ii = t0 + nn * t_step
				Aii = A[ii,:,:]
				NA = 0
				NR = 0
				if np.sum(np.abs(Aii)) == 0.0:
					logR[kk,nn:,:] = np.nan
					les[kk,nn:,:] = np.nan
					if NA != 0:
						NA = nn
				else:
					Ae = np.dot(Aii,e)
					Q, R = np.linalg.qr(Ae)
					if nn > 0:
						logR[kk,nn,:] = np.log(np.abs(np.diag(R)))
						if np.abs(np.sum(logR[kk,nn,:])) == np.inf:
							les[kk,nn,:] = np.nan
							if NR != 0: NR = nn
						else:
							les[kk,nn,:] = (1 / (nn * t_step * dt)) * \
								np.sum(logR[kk,1:nn,:], axis=0)
					e = Q
				if NA != 0: les[kk,NA:,:] = np.nan
				if NR != 0: les[kk,NR:,:] = np.nan

				# Temporary ignore RunTimeWarning when performing
				# np.nanmean and np.nanstd on arrays containing only nan
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", category=RuntimeWarning)
					les_mean[nn,:] = np.nanmean(les[:,nn,:], axis=0)
					les_std [nn,:] = np.nanstd (les[:,nn,:], axis=0)
		if verbose==True:
			print("\nles: ", end='')
			print(les_mean[-1,:])
			print("les std: ", end='')
			print(les_std[-1,:])
	else:
		raise ValueError("Select a valid method")
	# get results
	les_mean_last = les_mean[-1,:]
	les_std_last = les_std[-1,:]
	H_last = np.sum(les_mean_last[les_mean_last>0])
	results = {
		'les_mean': les_mean_last,
		'les_std' : les_std_last,
		'H'       : H_last,
	}
	return results