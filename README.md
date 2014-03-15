lgcp
====

nonparametric modeling of point processes

Models a spatio temporal point process as a log Gaussian Cox Process. 
A fixed, fine grid is used to approximate the intensity functions, which 
are then inferred via MCMC.  


1. Specifying prior covariance functions over space and time 
   - space: smoother
   - time: fit spectral density for a stationary kernel
           to find interesting periodicities




