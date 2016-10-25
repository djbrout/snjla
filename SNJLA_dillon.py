import numpy as np
from scipy import interpolate, linalg, optimize


# Numbers here are hardcoded for the JLA compilation
# The interpolation.npy is only for JLA redshifts

c = 299792.458 # km/s
H0 = 70 #(km/s) / Mpc

N=740 ; # Number of SNe

# Spline interpolation of luminosity distance
# Interpolation.npy is a table calculated in Mathematica
# The grid size can be seen from here: .01 between calculated points (in OM-OL space).
# Only calculated for OM in [0,1.5], OL in [-.5,1.5]
print 'loading Interpolation'
interp = np.load( 'Interpolation.npy' )
print 'splining'
tempInt = [] ;
for i in range(N):
	tempInt.append(interpolate.RectBivariateSpline( np.arange(0,1.51,.01), np.arange(-.50,1.51,.01) , interp[i]))

def dL( OM, OL ): # Returns in same order as always - c/H0 multiplied on after, in mu
	return np.hstack( [tempdL(OM,OL) for tempdL in tempInt] );
def MU( OM, OL ):
	return 5*np.log10( c/H0 * dL(OM,OL) ) + 25

#Import JLA data
#cols are z,m,x,c,cluster mass, survey
print 'loading jla.npy'
Z = np.load( 'JLA.npy' ) ;

#### FULL LIKELIHOOD ####
print 'loading full likelihood'
COVd = np.load( 'covmat/stat.npy' ) # Constructing data covariance matrix w/ sys.
# sigmaz and sigmalens are constructed as described in the JLA paper
# all others are taken from their .tar and converted to python format
for i in [ "cal", "model", "bias", "dust", "pecvel", "sigmaz", "sigmalens", "nonia" ]:
#Notice the lack of "host" covariances - we don't include the mass-step correction.
	COVd += np.load( 'covmat/'+i+'.npy' ); print 'loaded covmat/'+i+'.npy'

def COV( A , B , VM, VX, VC , RV=0): # Total covariance matrix
	block3 = np.array( [[VM + VX*A**2 + VC*B**2,    -VX*A, VC*B],
			     [-VX*A , VX, 0],
			     [ VC*B ,  0, VC]] )
	ATCOVlA = linalg.block_diag( *[ block3 for i in range(N) ] ) ;
	
	if RV==0:
		return np.array( COVd + ATCOVlA );
	elif RV==1:
		return np.array( COVd );
	elif RV==2:
		return np.array( ATCOVlA );

def RES( OM, OL , A , B , M0, X0, C0 ): #Total residual, \hat Z - Y_0*A
	Y0A = np.array([ M0-A*X0+B*C0, X0, C0 ]) 
	mu = MU(OM, OL)[0] ;
	return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - Y0A ) for i in range(N) ] )  


def m2loglike(pars , RV = 0):
	if RV != 0 and RV != 1 and RV != 2:
		raise ValueError('Inappropriate RV value')
	else:
		cov = COV( *[ pars[i] for i in [2,5,9,4,7] ] )
	try:
		chol_fac = linalg.cho_factor(cov, overwrite_a = True, lower = True ) 
	except ValueError: # If contains infinity
		return 13995*10.**20
	except:# If not positive definite
		return 13993*10.**20 
	res = RES( *[ pars[i] for i in [0,1,2,5,8,3,6] ] )

#Dont throw away the logPI part.
	part_log = 3*N*np.log(2*np.pi) + np.sum( np.log( np.diag( chol_fac[0] ) ) ) * 2
	part_exp = np.dot( res, linalg.cho_solve( chol_fac, res) )

	if pars[0]<0 or pars[0]>1.5 or pars[1]<-.50 or pars[1]>1.5 \
		    or pars[4]<0 or pars[7]<0 or pars[9]<0:
		part_exp += 100* np.sum(np.array([ _**2 for _ in pars ]))
	# if outside valid region, give penalty
        if RV==0:
            #print 'rv0'
            m2loglike = part_log + part_exp
            #print 'inside',part_log, part_exp,m2loglike
            if np.isnan(m2loglike):
                print 'isnan'
                return 13995*10.**20
            elif np.isinf(m2loglike):
                print 'isinf'
                return 13995*10.**20
            else:
                return m2loglike 
        elif RV==1: 
            print 'rv1'
            return part_exp 
        elif RV==2:
            print 'rv2'
            return part_log 

# Constraint fucntions for fits (constraint is func == 0)

def m2CONSflat( pars ):
	return pars[0] + pars[1] - 1

def m2CONSempt( pars ):
	return pars[0]**2 + pars[1]**2


def m2CONSzm( pars ):
	return pars[0]**2

def m2CONSEdS( pars ):
	return (pars[0]-1)**2 + pars[1]**2

def m2CONSacc( pars ):
	return pars[0]/2. - pars[1]

#### CONSTRAINED CHI2 ####

def COV_C( A , B , VM ):
	block1 = np.array( [1 , A , -B] ) ;
	AJLA = linalg.block_diag( *[ block1 for i in range(N) ] );
	return np.dot( AJLA, np.dot( COVd, AJLA.transpose() ) ) + np.eye(N) * VM;

def RES_C( OM, OL, A ,B , M0 ):
	mu = MU(OM,OL)[0] ;
	return Z[:,1] - M0 + A * Z[:,2] - B * Z[:,3] - mu

# INPUT HERE IS REDUCED: pars = [ om, ol, a, b, m0] , VM seperate

def chi2_C( pars, VM ):
	if pars[0]<0 or pars[0]>1.5 or pars[1]<-.50 or pars[1]>1.5 \
		    or VM<0:
		return 14994*10.**20
	cov = COV_C( pars[2], pars[3] , VM )
	chol_fac = linalg.cho_factor( cov, overwrite_a = True, lower = True )
	
	res = RES_C( *pars )
	
	part_exp = np.dot( res , linalg.cho_solve( chol_fac, res) )
	return part_exp


bounds = ( (0,1.5),(-0.5,1.5),
	   (None,None),(None,None),(0,None),
	   (None,None),(None,None),(0,None),
	   (None,None),(0,None) )

# Results already found

pre_found_best = np.array([  3.40658319e-01,   5.68558786e-01,   1.34469382e-01,
			      3.84466029e-02,   8.67848219e-01,   3.05861386e+00,
			      -1.59939791e-02,   5.04364259e-03,  -1.90515806e+01,
			               1.17007078e-02])

import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator

import mcmc as m

mcmc_obj = m.mh(m2loglike,[0])

param_step_stds = np.sqrt(abs(pre_found_best))/400.
mcmc_obj.run_d_mc(pre_found_best,param_step_stds,numsteps=25000,burn_in=1000)

p, p_uncertainty, chains, accepted_fraction, loglike_history = mcmc_obj.get_params()

## chains = size(numsteps,length(p0))

labels = [r"$\Omega_M$",r"$\Omega_{\Lambda}$",r"$ \alpha $",r"$x_0$",r"$ \sigma^2_{x_0} $",r"$ \beta $",r"$c_0$",r"$ \sigma^2_{c_0} $",r"$M_0$",r"$ \sigma^2_{M_0}$"]
fig, axesx = pl.subplots(5,2, sharex=True, figsize=(12, 14))
axes = axesx.ravel()
for i in range(len(chains[0,:])):
    axes[i].plot(chains[:, i], color="k")
    axes[i].yaxis.set_major_locator(MaxNLocator(5))
    axes[i].set_ylabel(labels[i],size=20)

axes[-2].set_xlabel("step number")
axes[-1].set_xlabel("step number")

fig.tight_layout(h_pad=0.0)
pl.savefig('chains.png')

import corner
fig = corner.corner(chains,labels=labels,quantiles=[0.68, 0.95, 0.997])
pl.savefig('corner.png')

def correlation_matrix(df):
    from matplotlib import cm as cm
    fig = pl.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    pl.title('Covariance')
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    cbar = fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])

from pandas import DataFrame, read_csv
df = DataFrame(data=chains,columns=labels)
correlation_matrix(df)
pl.savefig('covar.png')
