import os
import sys
import numpy as np
import xarray as xr
np.random.seed(0)
# custom libraries
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD,"../pypardi"))
import _deprecated_global_indices as _dgi



def _test_global_lyapunov__deprecated():
	# read data
	fname = os.path.join(CFD, "data", "data_lorenz.txt")
	data = np.loadtxt(fname)
	# for compatibility with deprecated library
	data = data.reshape(data.shape + (1,))
	## compute local indices
	global_indices = _dgi.compute_lyapunov_spectrum(
    	data, eps_over_L0=0.001, n=100, sampling=['mid',20])
	## results
	tol = 1e-10
	assert(global_indices['les_mean' ].shape == (3,))
	assert(global_indices['les_std'  ].shape == (3,))
	## estimated les in unit time step
	assert(global_indices['les_mean'][0]<0.019933916452632927+tol) & \
		  (global_indices['les_mean'][0]>0.019933916452632927-tol)
	assert(global_indices['les_mean'][1]<-0.00530222807902822+tol) & \
		  (global_indices['les_mean'][1]>-0.00530222807902822-tol)
	assert(global_indices['les_mean'][2]<-0.06518322109157328+tol) & \
		  (global_indices['les_mean'][2]>-0.06518322109157328-tol)
	## estimated les_std in unit time step
	assert(global_indices['les_std'  ][0]<0.015366383211936412+tol) & \
		  (global_indices['les_std'  ][0]>0.015366383211936412-tol)
	assert(global_indices['les_std'  ][1]<0.014515555314841774+tol) & \
		  (global_indices['les_std'  ][1]>0.014515555314841774-tol)
	assert(global_indices['les_std'  ][2]<0.018226627606356818+tol) & \
		  (global_indices['les_std'  ][2]>0.018226627606356818-tol)



if __name__ == '__main__':
	_test_global_lyapunov__deprecated()
