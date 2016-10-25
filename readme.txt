Supernova Likelihood maximisation

Besides this readme, six files exist in the currect directory:
	-	JLA.npy + JLA.tsv
	JLA supernova parameters as found on http://supernovae.in2p3.fr/sdss_snls_jla/ReadMe.html, converted to python format, and in tab-seperated values.

	-	SNJLA.py
	The fitting script. Uses scipy minimisation functions. Uses only basic numpy/scipy functions.
	Run by writing (in this directory): 
	python SNJLA.py

	This starts from pre determined points and minimises the likelihood function. Results in scipy format are output.

	-	covmat directory
	Stores all the covariance matrices in numpy format.

	-	Interpolation.npy
	Interpolation between points in (Omega_m,Omega_Lambda) space for rapid likelihood evaluation.

	-	findinterpolation.np
	Mathematica script to compute Interpolation.npy
