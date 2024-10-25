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
import local_indices as li



# set parameters
filepath = "data"
filename = "data_lorenz"
ql = 0.98
# load data
fname = os.path.join(CFD, filepath, filename+'.txt')
Y = np.loadtxt(fname)
t = Y[:,0]
X = Y[:,1:]



def test_local_d1():
	# compute exceeds
	exceeds, exceeds_idx, exceeds_bool = li.compute_exceeds(
		X, filepath=filepath, filename=filename, ql=ql, n_jobs=12,
		theiler_len=0)
	# compute d1
	d1 = li.compute_d1(exceeds, filepath, filename, ql=ql, theiler_len=0)
	## results
	# print(f"{d1.shape=:}")
	# print(f"{d1[0]=:}")
	# print(f"{d1[1022]=:}")
	# print(f"{np.min(d1)=:}")
	# print(f"{np.max(d1)=:}")
	# print(f"{np.mean(d1)=:}")
	tol = 1e-10
	assert(d1.shape == (2001,))
	## indices at idx 0 and 1022
	assert(d1[0   ]<0.958205640476+tol) & (d1[0]>0.9582056404761-tol)
	assert(d1[1022]<2.197065464775+tol) & (d1[1022]>2.1490820801-tol)
	## min, max, mean indices
	assert(np.min (d1)<0.805646834908+tol) & (np.min (d1)>0.805646834908-tol)
	assert(np.max (d1)<4.116311739464+tol) & (np.max (d1)>4.116311739464-tol)
	assert(np.mean(d1)<2.155555563283+tol) & (np.mean(d1)>2.155555563283-tol)



def test_local_theta():
	# compute exceeds
	exceeds, exceeds_idx, exceeds_bool = li.compute_exceeds(
		X, filepath=filepath, filename=filename, ql=ql, n_jobs=12,
		theiler_len=0)
	# compute theta
	theta = li.compute_theta(
	 	exceeds_idx, filepath, filename, ql=ql, theiler_len=0)
	## results
	# print(f"{theta.shape=:}")
	# print(f"{theta[0]=:}")
	# print(f"{theta[1022]=:}")
	# print(f"{np.min(theta)=:}")
	# print(f"{np.max(theta)=:}")
	# print(f"{np.mean(theta)=:}")
	tol = 1e-10
	## test shape
	assert(theta.shape == (2001,))
	## indices at idx 0 and 1022
	assert(theta[0   ]<0.053261231515+tol) & (theta[0   ]>0.053261231515-tol)
	assert(theta[1022]<0.273748344891+tol) & (theta[1022]>0.273748344891-tol)
	## min, max, mean indices
	assert(np.min (theta)<0.053261231515+tol) & \
     	  (np.min (theta)>0.053261231515-tol)
	assert(np.max (theta)<0.332636980882+tol) & \
     	  (np.max (theta)>0.332636980882-tol)
	assert(np.mean(theta)<0.241106981243+tol) & \
     	  (np.mean(theta)>0.241106981243-tol)



def test_local_alphat_compute_exceeds():
	# compute exceeds
	exceeds, exceeds_idx, exceeds_bool = li.compute_exceeds(
		X, filepath=filepath, filename=filename, ql=ql, n_jobs=12,
		theiler_len=0)
	# compute alphat for time_lags = [11,22,33]
	time_lag = [11, 22, 33]
	alphat = li.compute_alphat(
	 	exceeds_bool, filepath, filename, 
   		time_lag=time_lag, ql=ql, theiler_len=0)
	## results
	# print(f"{alphat[11].shape=:}")
	# print(f"{alphat[11][0]=:}")
	# print(f"{alphat[11][1022]=:}")
	# print(f"{np.min(alphat[11])=:}")
	# print(f"{np.max(alphat[11])=:}")
	# print(f"{np.mean(alphat[11])=:}")
	# print(f"{alphat[22].shape=:}")
	# print(f"{alphat[22][0]=:}")
	# print(f"{alphat[22][1022]=:}")
	# print(f"{np.min(alphat[22])=:}")
	# print(f"{np.max(alphat[22])=:}")
	# print(f"{np.mean(alphat[22])=:}")
	# print(f"{alphat[33].shape=:}")
	# print(f"{alphat[33][0]=:}")
	# print(f"{alphat[33][1022]=:}")
	# print(f"{np.min(alphat[33])=:}")
	# print(f"{np.max(alphat[33])=:}")
	# print(f"{np.mean(alphat[33])=:}")
	tol = 1e-10
	## test shape
	assert(alphat[11].shape == (1990,))
	assert(alphat[22].shape == (1979,))
	assert(alphat[33].shape == (1968,))
	## indices at idx 0 and 1022
	assert(alphat[11][0   ]<0.375+tol) & (alphat[11][0   ]>0.375-tol)
	assert(alphat[11][1022]<0.800+tol) & (alphat[11][1022]>0.800-tol)
	assert(alphat[22][0   ]<0.150+tol) & (alphat[22][0   ]>0.150-tol)
	assert(alphat[22][1022]<0.800+tol) & (alphat[22][1022]>0.800-tol)
	assert(alphat[33][0   ]<0.075+tol) & (alphat[33][0   ]>0.075-tol)
	assert(alphat[33][1022]<0.850+tol) & (alphat[33][1022]>0.850-tol)
	## min, max, mean indices
	assert(np.min (alphat[11])<0.200+tol) & \
     	  (np.min (alphat[11])>0.200-tol)
	assert(np.max (alphat[11])<1.000+tol) & \
     	  (np.max (alphat[11])>1.000-tol)
	assert(np.mean(alphat[11])<0.799095477386+tol) & \
     	  (np.mean(alphat[11])>0.799095477386-tol)
	assert(np.min (alphat[22])<0.150+tol) & \
     	  (np.min (alphat[22])>0.150-tol)
	assert(np.max (alphat[22])<1.000+tol) & \
     	  (np.max (alphat[22])>1.000-tol)
	assert(np.mean(alphat[22])<0.703840323395+tol) & \
     	  (np.mean(alphat[22])>0.703840323395-tol)
	assert(np.min (alphat[33])<0.075+tol) & \
     	  (np.min (alphat[33])>0.075-tol)
	assert(np.max (alphat[33])<0.975+tol) & \
     	  (np.max (alphat[33])>0.975-tol)
	assert(np.mean(alphat[33])<0.646100101626+tol) & \
     	  (np.mean(alphat[33])>0.646100101626-tol)
    
    
    
def test_local_alphat_load_exceeds():
	# load exceeds
	theiler_len = 0
	fname_exceeds = f"data_lorenz_exceeds_idx_{ql}_{theiler_len}_test"
	fname = os.path.join(CFD, filepath, fname_exceeds+'.npy')
	exceeds_idx = np.load(fname)
	# convert to boolean available exceeds
	exceeds_bool = li.create_bool_from_idx(exceeds_idx)
	# compute alphat for time_lags = [11,22,33]
	time_lag = [11, 22, 33]
	alphat = li.compute_alphat(
	 	exceeds_bool, filepath, filename, 
   		time_lag=time_lag, ql=ql, theiler_len=0)
	## results
	# print(f"{alphat[11].shape=:}")
	# print(f"{alphat[11][0]=:}")
	# print(f"{alphat[11][1022]=:}")
	# print(f"{np.min(alphat[11])=:}")
	# print(f"{np.max(alphat[11])=:}")
	# print(f"{np.mean(alphat[11])=:}")
	# print(f"{alphat[22].shape=:}")
	# print(f"{alphat[22][0]=:}")
	# print(f"{alphat[22][1022]=:}")
	# print(f"{np.min(alphat[22])=:}")
	# print(f"{np.max(alphat[22])=:}")
	# print(f"{np.mean(alphat[22])=:}")
	# print(f"{alphat[33].shape=:}")
	# print(f"{alphat[33][0]=:}")
	# print(f"{alphat[33][1022]=:}")
	# print(f"{np.min(alphat[33])=:}")
	# print(f"{np.max(alphat[33])=:}")
	# print(f"{np.mean(alphat[33])=:}")
	tol = 1e-10
	## test shape
	assert(alphat[11].shape == (1990,))
	assert(alphat[22].shape == (1979,))
	assert(alphat[33].shape == (1968,))
	## indices at idx 0 and 1022
	assert(alphat[11][0   ]<0.375+tol) & (alphat[11][0   ]>0.375-tol)
	assert(alphat[11][1022]<0.800+tol) & (alphat[11][1022]>0.800-tol)
	assert(alphat[22][0   ]<0.150+tol) & (alphat[22][0   ]>0.150-tol)
	assert(alphat[22][1022]<0.800+tol) & (alphat[22][1022]>0.800-tol)
	assert(alphat[33][0   ]<0.075+tol) & (alphat[33][0   ]>0.075-tol)
	assert(alphat[33][1022]<0.850+tol) & (alphat[33][1022]>0.850-tol)
	## min, max, mean indices
	assert(np.min (alphat[11])<0.200+tol) & \
     	  (np.min (alphat[11])>0.200-tol)
	assert(np.max (alphat[11])<1.000+tol) & \
     	  (np.max (alphat[11])>1.000-tol)
	assert(np.mean(alphat[11])<0.799095477386+tol) & \
     	  (np.mean(alphat[11])>0.799095477386-tol)
	assert(np.min (alphat[22])<0.150+tol) & \
     	  (np.min (alphat[22])>0.150-tol)
	assert(np.max (alphat[22])<1.000+tol) & \
     	  (np.max (alphat[22])>1.000-tol)
	assert(np.mean(alphat[22])<0.703840323395+tol) & \
     	  (np.mean(alphat[22])>0.703840323395-tol)
	assert(np.min (alphat[33])<0.075+tol) & \
     	  (np.min (alphat[33])>0.075-tol)
	assert(np.max (alphat[33])<0.975+tol) & \
     	  (np.max (alphat[33])>0.975-tol)
	assert(np.mean(alphat[33])<0.646100101626+tol) & \
     	  (np.mean(alphat[33])>0.646100101626-tol)
 
 



if __name__ == '__main__':
	test_local_d1()
	test_local_theta()
	test_local_alphat_compute_exceeds()
	test_local_alphat_load_exceeds()
