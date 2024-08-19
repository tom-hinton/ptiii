"""
Created on Wed Aug 25 12:52:46 2021

@author: vinco
"""

import pdb

def get_params(case_study):
	if case_study=='b693':
		parameters = {
				't0' : 7150.0,
				'tend' : 7350.0,
				'num_headers' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 350,
				'peaks_height' : 0.6
				}
	elif case_study=='b694':
		parameters = {
				't0' : 6330.0,
				'tend' : 6530.0,
				'num_headers' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 300,
				'peaks_height' : 0.6
				}
	elif case_study=='b695':
		parameters = {
				't0' : 7900.0,
				'tend' : 8100.0,
				'num_headers' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 400,
				'peaks_height' : 0.6
				}
	elif case_study=='b696':
		parameters = {
				't0' : 7300.0,
				'tend' : 7500,
				'num_headers' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 300,
				'peaks_height' : 0.6
				}
	elif case_study=='b697':
		parameters = {
				't0' : 9150.0,
				'tend' : 9350.0,
				'num_headers' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 350,
				'peaks_height' : 0.5
				}
	elif case_study=='b698':
		parameters = {
				't0' : 11100.0,
				'tend' : 11300.0,
				'num_headers' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 350,
				'peaks_height' : 0.6
				}
	elif case_study=='b721':
		parameters = {
				't0' : 8500.0,
				'tend' : 8700.0,
				'num_headers' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 390,
				'peaks_height' : 0.7
				}
	elif case_study=='b722':
		parameters = {
				't0' : 7900.0,
				'tend' : 8100.0,
				'num_headers' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 300,
				'peaks_height' : 0.6
				}
	elif case_study=='b724':
		parameters = {
				't0' : 8400.0,
				'tend' : 8600.0,
				'num_headers' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 300,
				'peaks_height' : 0.6
				}
	elif case_study=='b725':
		parameters = {
				't0' : 8550.0,
				'tend' : 8750.0,
				'num_headers' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 350,
				'peaks_height' : 0.7
				}
	elif case_study=='b726':
		parameters = {
				't0' : 9450.0,
				'tend' : 9650.0,
				'num_headers' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 300,
				'peaks_height' : 0.6
				}
	elif case_study=='b727':
		parameters = {
				't0' : 7100.0,
				'tend' : 7300.0,
				'num_headers' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 390,
				'peaks_height' : 0.7
				}
	elif case_study=='b728':
		parameters = {
				't0' : 7677.5,
				'tend' : 7877.5,
				'num_headers' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 400,
				'peaks_height' : 0.6
				}
	elif case_study=='i417':
		parameters = {
				't0' : 3650.0,
				'tend' : 3850.0,
				'num_headers' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 390,
				'peaks_height' : 0.8
				}
	else:
		raise Exception('Paramaters of case study unknown')
	return parameters