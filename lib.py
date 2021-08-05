#/usr/bin/env python
# encoding: utf-8
import numpy as np
from scipy import linalg

def bounds(lats, lons, cycles, lat_min, lat_max, lon_min, lon_max, cycle_min, cycle_max):
    """
    Given bounds calculates indices of points and cycles within the bounds
    
    lats, lons, cycles are array-like objects of latitude, longitude, and cycle, respectively.
    lat_min, lon_min, cycle_min are respective minimum values
    lat_max, lon_max, cycle_max are respective maximum values
    """
    #first for latitudes
    lat_idx = np.array([i[0] for i in enumerate(lats) if i[1] <= lat_max and i[1] >= lat_min])
    #then for longitudes
    lon_idx = np.array([i[0] for i in enumerate(lons) if i[1] <= lon_max and i[1] >= lon_min])
    #join the lists
    point_idx = np.array([i[1] for i in enumerate(lat_idx) if i[1] in lon_idx])
    #point_idx2 = np.array([i[1] for i in enumerate(lon_idx) if i[1] in lat_idx])
    #assert(len(point_idx) == len(point_idx2))
    #then for cycles
    cycle_idx = np.array([i[0] for i in enumerate(cycles) if i[1] <= cycle_max and i[1] >= cycle_min])
    
    return point_idx, cycle_idx

"""
The functions were developed by CTOH team at LEGOS, in particular by F.Birol and F.Leger,
who courteously provided the script. 
"""

def genweights(p, q, dt):
    """Given p and q, return the vector of cn's: the optimal weighting
    coefficients and the noise reduction factor of h.

    p is the number of points before the point of interest (always negative)
    q is the number of points after the point of interest (always positive)
    dt is the sampling period (defaults to 1s)

    Written by Brian Powell (2004) University of Colorado, Boulder
    """
    p = max(p, -p)
    q = max(q, -q)

    #check inputs
    if (-p > q): 
        raise RuntimeError("genweights : P must be lesser than q")

    #Build matrices
    N = abs(p) + abs(q)
    T = N + 1
    A = np.matrix(np.zeros((T,T)))
    A[T-1,:] = np.append(np.ones(N), 0.)
    sn = np.arange(-p, q+1)
    sn = sn.compress(sn != 0)
    for i in np.arange(len(sn)):
        A[i,:] = np.append(((1./sn)*(-sn[i]/2.)),sn[i]**2.*dt**2./4.) #Eq.11 (PL)
        A[i,i] = -1.

    B = np.zeros(T)
    B[N] = 1.0

    #Compute the coefficients
    cn=linalg.solve(A,B)
    cn = cn[0:N] #Check the indices

    #Compute the error 
    error = np.sqrt(np.sum(cn.transpose()/(sn*dt))**2. + np.sum((cn.transpose()/(sn*dt))**2.))

    return cn, error

def geost1D(x,y,m,ssh):
    """% Alongtrack Geostrophic speed calculation 1D

        INPuT:
                x: longitude vector
                y: latitude vector
                m: derivative window length (taille de la fenetre de calcul des derivees)
                ssh: alongtrack ssh vector
        OuTPuT:
                u: speed

    """
    g=9.81        # Gravity
    f0=2*7.29e-5  # Coriolis f=f0*sin(lat)
    deg2rad=np.pi/180.
    Rdeg2rad = 111100. # Radius of Earth in m * deg2rad

    (cn, _) = genweights(m,m,1)
    n=len(y)
    assert(len(x)==len(y) and len(x)==len(ssh))
    u=np.nan*np.zeros(n)
    for i in range(m, n-m):
        f=f0*np.sin(y[i]*deg2rad)
        u[i]=0.
        # calculate ui-
        for j in range(1,m+1): 
            # Distance calculation by equirectangular approximation
            dlon=(x[i]-x[i-j])*np.cos(y[i]*deg2rad) # * deg2rad
            dlat=y[i]-y[i-j] # * deg2rad
            dist=dlat*dlat+dlon*dlon
            dist=np.sqrt(dist)*Rdeg2rad # deg2rad is finally taken into account
            dh=ssh[i]-ssh[i-j]
            u[i]+=-(g*dh)/(f*dist)*cn[j-1]
        # calcul ui+
        for j in range(1,m+1): 
            dlat=y[i+j]-y[i]
            dlon=(x[i+j]-x[i])*np.cos(y[i]*np.pi/180)
            dist=dlat*dlat+dlon*dlon
            dist=np.sqrt(dist)*111100
            dh=ssh[i+j]-ssh[i]
            u[i]+=-(g*dh)/(f*dist)*cn[j+m-1]

    return u




