#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:05:04 2020

@author: davidvishny
"""

### This script is used for reading the simulation output data and generating the figures shown in the manuscript ###

from scipy.stats import t as stud
from statistics import stdev
import numpy as np
import numba
from numba import jit
from numba import njit
import scipy as sp 
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import netCDF4 as nc
from scipy.fftpack import fftfreq
#import spectrum
import scipy.signal.windows as windows
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib import cm
import matplotlib.colors as colors




AM_NC = 'ctrl_take_3_1.nc'
daily_data= Dataset(AM_NC)
u_zon_av_nd = daily_data.variables['u_zon_av'][:]
v_zon_av_nd = daily_data.variables['v_zon_av'][:]
momflux_zon_av_nd = daily_data.variables['momflux_zon_av'][:]
#uw_eddy = daily_data.variables['uw_tot_zon_av'][:] - daily_data.variables['uw_barbar_zon_av'][:]

temp_zon_av_nd = daily_data.variables['temp_zon_av'][:]
eke_zon_av_nd = daily_data.variables['eke_zon_av'][:]
lat = daily_data.variables['lat'][:]
press = daily_data.variables['press'][:]
vT_tot_zon_av_nd = daily_data.variables['vT_tot_zon_av'][:]
vT_barbar_zon_av_nd = daily_data.variables['vT_barbar_zon_av'][:]



varlist = daily_data.variables.keys() 
daily_data.close()

AM_NC = 'SAM_CW_a1_retest_1.nc'
daily_data= Dataset(AM_NC)
u_zon_av_forc_a1_nd = daily_data.variables['u_zon_av'][:]
v_zon_av_forc_a1_nd = daily_data.variables['v_zon_av'][:]
momflux_zon_av_forc_a1_nd = daily_data.variables['momflux_zon_av'][:]
temp_zon_av_forc_a1_nd = daily_data.variables['temp_zon_av'][:]
eke_zon_av_forc_a1_nd = daily_data.variables['eke_zon_av'][:]
vT_tot_zon_av_forc_a1_nd = daily_data.variables['vT_tot_zon_av'][:]
vT_barbar_zon_av_forc_a1_nd = daily_data.variables['vT_barbar_zon_av'][:]
#uw_eddy_forc_a1 = daily_data.variables['uw_tot_zon_av'][:] - daily_data.variables['uw_barbar_zon_av'][:]
daily_data.close()

AM_NC = 'SAM_CW_a-1_retest_1.nc'
daily_data= Dataset(AM_NC)
u_zon_av_forc_a2_nd = daily_data.variables['u_zon_av'][:]
v_zon_av_forc_a2_nd = daily_data.variables['v_zon_av'][:]
momflux_zon_av_forc_a2_nd = daily_data.variables['momflux_zon_av'][:]
temp_zon_av_forc_a2_nd = daily_data.variables['temp_zon_av'][:]
eke_zon_av_forc_a2_nd = daily_data.variables['eke_zon_av'][:]
vT_tot_zon_av_forc_a2_nd = daily_data.variables['vT_tot_zon_av'][:]
vT_barbar_zon_av_forc_a2_nd = daily_data.variables['vT_barbar_zon_av'][:]
#uw_barbar_zon_av_forc_a2_nd = daily_data.variables['uw_barbar_zon_av'][:]
daily_data.close()


AM_NC = 'SAM_CW_a3_retest_1.nc'
daily_data= Dataset(AM_NC)
u_zon_av_forc_a3_nd = daily_data.variables['u_zon_av'][:]
v_zon_av_forc_a3_nd = daily_data.variables['v_zon_av'][:]
momflux_zon_av_forc_a3_nd = daily_data.variables['momflux_zon_av'][:]
temp_zon_av_forc_a3_nd = daily_data.variables['temp_zon_av'][:]
eke_zon_av_forc_a3_nd = daily_data.variables['eke_zon_av'][:]
vT_tot_zon_av_forc_a3_nd = daily_data.variables['vT_tot_zon_av'][:]
vT_barbar_zon_av_forc_a3_nd = daily_data.variables['vT_barbar_zon_av'][:]
#uw_barbar_zon_av_forc_a3_nd = daily_data.variables['uw_barbar_zon_av'][:]
daily_data.close()


AM_NC = 'SAM_CW_a-3_retest_1.nc'
daily_data= Dataset(AM_NC)
u_zon_av_forc_a4_nd = daily_data.variables['u_zon_av'][:]
v_zon_av_forc_a4_nd = daily_data.variables['v_zon_av'][:]
momflux_zon_av_forc_a4_nd = daily_data.variables['momflux_zon_av'][:]
temp_zon_av_forc_a4_nd = daily_data.variables['temp_zon_av'][:]
eke_zon_av_forc_a4_nd = daily_data.variables['eke_zon_av'][:]
vT_tot_zon_av_forc_a4_nd = daily_data.variables['vT_tot_zon_av'][:]
vT_barbar_zon_av_forc_a4_nd = daily_data.variables['vT_barbar_zon_av'][:]
#uw_barbar_zon_av_forc_a4_nd = daily_data.variables['uw_barbar_zon_av'][:]

daily_data.close()



num_legs = 19

for b in range(num_legs):
    AM_NC  = 'ctrl_take_3_' + str(b+2) + '.nc'
    daily_data_2 = Dataset(AM_NC)
    momflux_zon_av_2= daily_data_2.variables['momflux_zon_av'][:]
    v_zon_av_2 = daily_data_2.variables['v_zon_av'][:]
    eke_zon_av_2 = daily_data_2.variables['eke_zon_av'][:]
    momflux_zon_av_nd = np.append(momflux_zon_av_nd,momflux_zon_av_2,axis=0)
    v_zon_av_nd = np.append(v_zon_av_nd,v_zon_av_2,axis=0)
    u_zon_av_2 = daily_data_2.variables['u_zon_av'][:]
    u_zon_av_nd = np.append(u_zon_av_nd,u_zon_av_2,axis=0)
    eke_zon_av_nd = np.append(eke_zon_av_nd,eke_zon_av_2,axis=0)
    temp_zon_av_2 = daily_data_2.variables['temp_zon_av'][:]
    temp_zon_av_nd = np.append(temp_zon_av_nd,temp_zon_av_2,axis=0)
    vT_tot_zon_av_2 = daily_data_2.variables['vT_tot_zon_av'][:]
    vT_tot_zon_av_nd = np.append(vT_tot_zon_av_nd,vT_tot_zon_av_2,axis=0)
    vT_barbar_zon_av_2 = daily_data_2.variables['vT_barbar_zon_av'][:]
    vT_barbar_zon_av_nd = np.append(vT_barbar_zon_av_nd,vT_barbar_zon_av_2,axis=0)
    daily_data_2.close()
    


for b in range(num_legs):
    AM_NC  = 'SAM_CW_a1_retest_' + str(b+2) + '.nc'
    daily_data_2 = Dataset(AM_NC)
    momflux_zon_av_2= daily_data_2.variables['momflux_zon_av'][:]
   
    v_zon_av_2 = daily_data_2.variables['v_zon_av'][:]
    eke_zon_av_2 = daily_data_2.variables['eke_zon_av'][:]
    momflux_zon_av_forc_a1_nd = np.append(momflux_zon_av_forc_a1_nd,momflux_zon_av_2,axis=0)
   
    v_zon_av_forc_a1_nd = np.append(v_zon_av_forc_a1_nd,v_zon_av_2,axis=0)
    eke_zon_av_forc_a1_nd = np.append(eke_zon_av_forc_a1_nd,eke_zon_av_2,axis=0)
    u_zon_av_2 = daily_data_2.variables['u_zon_av'][:]
    u_zon_av_forc_a1_nd = np.append(u_zon_av_forc_a1_nd,u_zon_av_2,axis=0)
    temp_zon_av_2 = daily_data_2.variables['temp_zon_av'][:]
    temp_zon_av_forc_a1_nd = np.append(temp_zon_av_forc_a1_nd,temp_zon_av_2,axis=0)
    vT_tot_zon_av_2 = daily_data_2.variables['vT_tot_zon_av'][:]
    vT_tot_zon_av_forc_a1_nd = np.append(vT_tot_zon_av_forc_a1_nd,vT_tot_zon_av_2,axis=0)
    vT_barbar_zon_av_2 = daily_data_2.variables['vT_barbar_zon_av'][:]
    vT_barbar_zon_av_forc_a1_nd = np.append(vT_barbar_zon_av_forc_a1_nd,vT_barbar_zon_av_2,axis=0)
    daily_data_2.close()
    
for b in range(num_legs):
    AM_NC  = 'SAM_CW_a-1_retest_' + str(b+2) + '.nc'
    daily_data_2 = Dataset(AM_NC)
    momflux_zon_av_2= daily_data_2.variables['momflux_zon_av'][:]
    
    v_zon_av_2 = daily_data_2.variables['v_zon_av'][:]
    eke_zon_av_2 = daily_data_2.variables['eke_zon_av'][:]
    momflux_zon_av_forc_a2_nd = np.append(momflux_zon_av_forc_a2_nd,momflux_zon_av_2,axis=0)
   
    v_zon_av_forc_a2_nd = np.append(v_zon_av_forc_a2_nd,v_zon_av_2,axis=0)
    eke_zon_av_forc_a2_nd = np.append(eke_zon_av_forc_a2_nd,eke_zon_av_2,axis=0)
    u_zon_av_2 = daily_data_2.variables['u_zon_av'][:]
    u_zon_av_forc_a2_nd = np.append(u_zon_av_forc_a2_nd,u_zon_av_2,axis=0)
    temp_zon_av_2 = daily_data_2.variables['temp_zon_av'][:]
    temp_zon_av_forc_a2_nd = np.append(temp_zon_av_forc_a2_nd,temp_zon_av_2,axis=0)
    vT_tot_zon_av_2 = daily_data_2.variables['vT_tot_zon_av'][:]
    vT_tot_zon_av_forc_a2_nd = np.append(vT_tot_zon_av_forc_a2_nd,vT_tot_zon_av_2,axis=0)
    vT_barbar_zon_av_2 = daily_data_2.variables['vT_barbar_zon_av'][:]
    vT_barbar_zon_av_forc_a2_nd = np.append(vT_barbar_zon_av_forc_a2_nd,vT_barbar_zon_av_2,axis=0)
    daily_data_2.close()
    
    
for b in range(num_legs):
    AM_NC  = 'SAM_CW_a3_retest_' + str(b+2) + '.nc'
    daily_data_2 = Dataset(AM_NC)
    momflux_zon_av_2= daily_data_2.variables['momflux_zon_av'][:]
    v_zon_av_2 = daily_data_2.variables['v_zon_av'][:]
    eke_zon_av_2 = daily_data_2.variables['eke_zon_av'][:]
    momflux_zon_av_forc_a3_nd = np.append(momflux_zon_av_forc_a3_nd,momflux_zon_av_2,axis=0)
   
    v_zon_av_forc_a3_nd = np.append(v_zon_av_forc_a3_nd,v_zon_av_2,axis=0)
    eke_zon_av_forc_a3_nd = np.append(eke_zon_av_forc_a3_nd,eke_zon_av_2,axis=0)
    u_zon_av_2 = daily_data_2.variables['u_zon_av'][:]
    u_zon_av_forc_a3_nd = np.append(u_zon_av_forc_a3_nd,u_zon_av_2,axis=0)
    temp_zon_av_2 = daily_data_2.variables['temp_zon_av'][:]
    temp_zon_av_forc_a3_nd = np.append(temp_zon_av_forc_a3_nd,temp_zon_av_2,axis=0)
    vT_tot_zon_av_2 = daily_data_2.variables['vT_tot_zon_av'][:]
    vT_tot_zon_av_forc_a3_nd = np.append(vT_tot_zon_av_forc_a3_nd,vT_tot_zon_av_2,axis=0)
    vT_barbar_zon_av_2 = daily_data_2.variables['vT_barbar_zon_av'][:]
    vT_barbar_zon_av_forc_a3_nd = np.append(vT_barbar_zon_av_forc_a3_nd,vT_barbar_zon_av_2,axis=0)
    daily_data_2.close()

for b in range(num_legs):
    AM_NC  = 'SAM_CW_a-3_retest_' + str(b+2) + '.nc'
    daily_data_2 = Dataset(AM_NC)
    momflux_zon_av_2= daily_data_2.variables['momflux_zon_av'][:]
   
    v_zon_av_2 = daily_data_2.variables['v_zon_av'][:]
    eke_zon_av_2 = daily_data_2.variables['eke_zon_av'][:]
    momflux_zon_av_forc_a4_nd = np.append(momflux_zon_av_forc_a4_nd,momflux_zon_av_2,axis=0)
    
    v_zon_av_forc_a4_nd = np.append(v_zon_av_forc_a4_nd,v_zon_av_2,axis=0)
    eke_zon_av_forc_a4_nd = np.append(eke_zon_av_forc_a4_nd,eke_zon_av_2,axis=0)
    u_zon_av_2 = daily_data_2.variables['u_zon_av'][:]
    u_zon_av_forc_a4_nd = np.append(u_zon_av_forc_a4_nd,u_zon_av_2,axis=0)
    temp_zon_av_2 = daily_data_2.variables['temp_zon_av'][:]
    temp_zon_av_forc_a4_nd = np.append(temp_zon_av_forc_a4_nd,temp_zon_av_2,axis=0)
    vT_tot_zon_av_2 = daily_data_2.variables['vT_tot_zon_av'][:]
    vT_tot_zon_av_forc_a4_nd = np.append(vT_tot_zon_av_forc_a4_nd,vT_tot_zon_av_2,axis=0)
    vT_barbar_zon_av_2 = daily_data_2.variables['vT_barbar_zon_av'][:]
    vT_barbar_zon_av_forc_a4_nd = np.append(vT_barbar_zon_av_forc_a4_nd,vT_barbar_zon_av_2,axis=0)
    daily_data_2.close()

tlen_nd = len(u_zon_av_nd)


u_zon_av_vertsum_nd = np.mean(u_zon_av_nd,axis=1)
u_zon_av_vertsum_forc_a1_nd = np.mean(u_zon_av_forc_a1_nd,axis=1)
u_zon_av_vertsum_forc_a2_nd = np.mean(u_zon_av_forc_a2_nd,axis=1)
u_zon_av_vertsum_forc_a3_nd = np.mean(u_zon_av_forc_a3_nd,axis=1)
u_zon_av_vertsum_forc_a4_nd = np.mean(u_zon_av_forc_a4_nd,axis=1)




v_zon_av_vertsum_nd = np.mean(v_zon_av_nd,axis=1)
v_zon_av_vertsum_forc_a1_nd = np.mean(v_zon_av_forc_a1_nd,axis=1)
v_zon_av_vertsum_forc_a2_nd = np.mean(v_zon_av_forc_a2_nd,axis=1)
v_zon_av_vertsum_forc_a3_nd = np.mean(v_zon_av_forc_a3_nd,axis=1)
v_zon_av_vertsum_forc_a4_nd = np.mean(v_zon_av_forc_a4_nd,axis=1)


vorflux_vertsum_nd = np.zeros((tlen_nd,len(lat)),dtype='d')
vorflux_vertsum_forc_a1_nd = np.zeros((tlen_nd,len(lat)),dtype='d')
vorflux_vertsum_forc_a2_nd = np.zeros((tlen_nd,len(lat)),dtype='d')
vorflux_vertsum_forc_a3_nd = np.zeros((tlen_nd,len(lat)),dtype='d')
vorflux_vertsum_forc_a4_nd = np.zeros((tlen_nd,len(lat)),dtype='d')

vertflux_div = np.zeros((tlen_nd,len(press),len(lat)),dtype='d')
vertflux_div_forc_a1 = np.zeros((tlen_nd,len(press),len(lat)),dtype='d')

 
g = 9.81
T_star = 255. #ref temp. for  buoyancy
kappa = 2./7
HS_m = 8500.       

#### Make ctrl_clim.nc ####
temp_zon_av_clima = np.mean(temp_zon_av_nd,axis=0)
temp_zon_av_clima_forc_a1 = np.mean(temp_zon_av_forc_a1_nd,axis=0)
temp_zon_av_clima_forc_a2 = np.mean(temp_zon_av_forc_a2_nd,axis=0)
temp_zon_av_clima_forc_a3 = np.mean(temp_zon_av_forc_a3_nd,axis=0)
temp_zon_av_clima_forc_a4 = np.mean(temp_zon_av_forc_a4_nd,axis=0)


u_zon_av_clima = np.mean(u_zon_av_nd,axis=0)
u_zon_av_clima_forc_a1 = np.mean(u_zon_av_forc_a1_nd,axis=0)
u_zon_av_clima_forc_a2 = np.mean(u_zon_av_forc_a2_nd,axis=0)
u_zon_av_clima_forc_a3 = np.mean(u_zon_av_forc_a3_nd,axis=0)
u_zon_av_clima_forc_a4 = np.mean(u_zon_av_forc_a4_nd,axis=0)




################### Functions ################################

def f1(x):
    return f"{x:.1f}"
   

def Cov(X,Y):
    X_m = np.mean(X)
    Y_m = np.mean(Y)
    Sum  = np.dot(X-X_m,Y-Y_m)
    return Sum/len(X)

def Corr(X,Y):
    X = X - np.mean(X)
    Y = Y - np.mean(Y)
    return np.dot(X,Y)/(np.linalg.norm(X)*np.linalg.norm(Y))

def nearest_coord(val,arr):
    diff = abs(arr - val)
    diff_min = np.inf
    for i in range(len(arr)):
        if diff[i] < diff_min:
            diff_min = diff[i]
            coord = i
    return coord



def EOF1(X): 
    col = len(X[0])
    M = np.zeros((col,col), dtype = 'd')
    for i in range(col):
        for j in range(col):
                Mi = X[:,i]
                Mj = X[:,j]
                M[i,j] = Cov(Mi,Mj)
    [vals, vecs] = np.linalg.eig(M)
    return vecs[:,0]
   
   


def cross_corr(X,Y):
    X = X - np.mean(X)
    Y = Y - np.mean(Y)
    spec_X = sp.fft(X)
    spec_Y_star = np.conjugate(sp.fft(Y))
    cross_spec = spec_X * spec_Y_star
    A = sp.ifft(cross_spec)
    return A#/(np.linalg.norm(X)*np.linalg.norm(Y))


def cumav(X):
    n = len(X)
    C = np.zeros(len(X),dtype='d')
    C[0] = X[0]
    for k in range(n-1):
        C[k+1]  = k/(k+1.)*C[k] + X[k+1]/(k+1)
    return C


def maxind(X):
    M = -np.inf
    for j in range(len(X)):
        if X[j] > M:
            max_ind = j
            M = X[j]
    return max_ind

def specfilt_Butterworth(X,wc,o): #o > 0 -> high-pass, o < 0 -> low-pass
    freqs  = fftfreq(len(X), d=1.0)
    if np.size(X[0]) > 1:
        XT = np.transpose(X)
        ft_XT = sp.fft.fft(XT)
        for k in range(len(XT)): #space
            for n in range(np.size(XT[0])): #time
                ft_XT[k,n] = ft_XT[k,n]/(np.sqrt(1+(wc/freqs[n])**(2*o)))
        return np.transpose(sp.fft.ifft(ft_XT))
    else:
        ft_X = sp.fft.fft(X)
        for n in range(len(X)): #time
             ft_X[n] = ft_X[n]/(np.sqrt(1+(wc/freqs[n])**(2*o)))
             
        return sp.fft.ifft(ft_X)

def Lag(X,n):
    L = len(X)
    Y = np.zeros(len(X),dtype='d')
    for i in range(len(X)):
        Y[i] = X[(i+n)%L]
    return Y

def lag_dot(X,Y,tau):
    return np.dot(X, Lag(Y,tau))


def damp_rate(p): #Calculate damping rate in day^-1
    k_f = 1. 
    sig = p/press[-1] #p in hPa
    sig_b = 0.7
    return k_f * max(0,(sig-sig_b)/(1.-sig_b))

def rad_damp_rate(p,l): #Calculate damping rate in day^-1
    k_a = 1./40
    k_s =  1./4
    sig = p/press[-1] #p in hPa
    sig_b = 0.7
    #return k_a + (k_s-k_a)*max(0,(sig-sig_b)/(1.-sig_b)) * np.cos(l*np.pi/180)**4
    return k_a + (k_s-k_a)*max(0,(sig-sig_b)/(1.-sig_b)) #test meridional dependence

def ACF_full(X):
    Y = X - np.mean(X)
    spec = sp.fft.fft(Y)
    power_spec = np.abs(spec)**2
    A = sp.fft.ifft(power_spec)
    return A/A[0]

def CCF_full(X,Y):
    Z1 = X - np.mean(X)
    Z2 = Y - np.mean(Y)
    spec1 = sp.fft.fft(Z1)
    spec2 = sp.fft.fft(Z2)
    cross_spec = spec2*np.conj(spec1)
    C = sp.fft.ifft(cross_spec)
    return C

def std_ACF(N,tau,t):
    C1 = 1 + np.exp(-2/tau)
    C2 = 1 - np.exp(-2*t/tau)
    C3 = 1 - np.exp(-2/tau)
    D = -2*t*np.exp(-2*t/tau)
    return np.sqrt(1/N*(C1*C2/C3 + D)) #95% at ~1.6 sigma

def unwind(arr): #Convert 2D arr to 1D arr
    return arr.reshape(len(arr[:,0])*len(arr[0,:]))

def rewind(arr):
    return arr.reshape(len(press),len(lat))

def nandot(A,B):
    Prod = A*B
    return np.nansum(Prod)


del_p = 100 * (press[1] - press[0])
D_p = np.zeros((len(press),len(press)),dtype='d')
for j in range(len(press)):
    if j > 0 and j < len(press) -1 :
        D_p[j,j-1] = -1.
        D_p[j,j+1] = 1.
D_p *= 1./(2*del_p)

del_y = np.pi/180*(lat[1] - lat[0])*6.37e6
D_y = np.zeros((len(lat),len(lat)),dtype='d')
D2_y = np.zeros((len(lat),len(lat)),dtype='d')
for j in range(len(lat)):
    if j > 0 and j < len(lat) -1 :
        D_y[j,j-1] = -1.
        D_y[j,j+1] = 1.
        
        D2_y[j,j-1] = 1.
        D2_y[j,j] = -2.
        D2_y[j,j+1] = 1.
        
D_y *= 1./(2*del_y)
D2_y *= 1./(del_y)**2


D_z = -100.*press[:,np.newaxis]/HS_m * D_p 


def d_y(X): #FD meridional derivative
 
  return np.matmul(D_y,X)




def d2_y(X): #FD meridional derivative
 
 return np.matmul(D2_y,X)

def d_P(X): #FD pressure derivative
 return np.matmul(D_p,X)

def d_z(X): #FD z derivative
 return np.matmul(D_z,X)


D_t = np.zeros((tlen_nd,tlen_nd),dtype='d')
for j in range(tlen_nd):
    if  j < tlen_nd -1 :
        D_t[j,j] = -1.
        D_t[j,j+1] = 1. 
def d_t(X): #FD time derivative
 return np.matmul(D_t,X)

def trap_int(X):  #trapezoidal integration
    Y = np.zeros((len(X)-1),dtype='d')
    for i in range(len(X)-1):
        Y[i] = (X[i]+X[i+1])/2.
    return np.cumsum(Y)

   
def near_ind(x,arr):
    diff = np.Infinity
    ind = 0
    for i in range(len(arr)):
        if abs(arr[i] - x) < diff:
            ind = i
            diff = abs(arr[i] - x)
    return ind

def linterp1D(arr, num_btw): #num_btw =  num. points added b/w original ones
    old_len = len(arr)
    new_len = old_len + (old_len-1)*num_btw
    new_arr = np.zeros(new_len, dtype='d')
    for i in range(old_len - 1):
        for j in range(num_btw+1):
          new_arr[i + j] = arr[i] + j*(arr[i+1]- arr[i])/(num_btw+1)
    return new_arr

# def t_CI(M,S,n,conf):
#     I = np.array(stud.interval(conf,n-1))
#     return [M - S/np.sqrt(n)*I[1], M - S/np.sqrt(n)*I[0]]

def t_err(S,n,conf):
    I = np.array(stud.interval(conf,n-1))
    return S/np.sqrt(n)*(I[1] - I[0])/2

GP14_test = t_err(0.94,10,0.95)

def adaptive_weights(x, Pk, nx, V ):
    sig2 = np.matrix(x) * np.matrix(x).H / nx # power
    P = ( Pk[0, :] + Pk[1, :] ) / 2 #initial spectrum estimate
    P1 = np.zeros( nx )
    tol = .0005 * sig2 / nx
    a = np.array( sig2 * (1. - V) )
    while sum( abs( P - P1 ) / nx) > tol:
        be = P[:, np.newaxis] / ( P[:, np.newaxis] * V[np.newaxis, :] + a[np.newaxis, :] ) # weights
        wk = (be ** 2) * V[np.newaxis, :] # new spectral estimate
        wk = np.swapaxes( wk, 1, 2 )
        P1 = sum( wk[0, :, :] * Pk ) / np.sum( wk[0, :, :], axis = 0 )
        Ptemp = P1[:]
        P1 = P
        P = Ptemp
    return P, be, wk  

def adaptive_weight_P(x, Pk, nx, V ):
    sig2 = np.matrix(x) * np.matrix(x).H / nx # power
    P = ( Pk[0, :] + Pk[1, :] ) / 2 #initial spectrum estimate
    P1 = np.zeros( nx )
    tol = .0005 * sig2 / nx
    a = np.array( sig2 * (1. - V) )
    while sum( abs( P - P1 ) / nx) > tol:
        b = P[:, np.newaxis] / ( P[:, np.newaxis] * V[np.newaxis, :] + a[np.newaxis, :] ) # weights
        wk = (b ** 2) * V[np.newaxis, :] # new spectral estimate
        wk = np.swapaxes( wk, 1, 2 )
        P1 = sum( wk[0, :, :] * Pk ) / np.sum( wk[0, :, :], axis = 0 )
        Ptemp = P1[:]
        P1 = P
        P = Ptemp
    return P 


def adaptive_weight_b(x, Pk, nx, V ):
      sig2 = np.matrix(x) * np.matrix(x).H / nx # power
      P = ( Pk[0, :] + Pk[1, :] ) / 2 #initial spectrum estimate
      P1 = np.zeros( nx )
      tol = .0005 * sig2 / nx
      a = np.array( sig2 * (1. - V) )
      while sum( abs( P - P1 ) / nx) > tol:
          b = P[:, np.newaxis] / ( P[:, np.newaxis] * V[np.newaxis, :] + a[np.newaxis, :] ) # weights
          wk = (b ** 2) * V[np.newaxis, :] # new spectral estimate
          wk = np.swapaxes( wk, 1, 2 )
          P1 = sum( wk[0, :, :] * Pk ) / np.sum( wk[0, :, :], axis = 0 )
          Ptemp = P1[:]
          P1 = P
          P = Ptemp
      return b

def adaptive_weight_wk(x, Pk, nx, V ):
      sig2 = np.matrix(x) * np.matrix(x).H / nx # power
      P = ( Pk[0, :] + Pk[1, :] ) / 2 #initial spectrum estimate
      P1 = np.zeros( nx )
      tol = .0005 * sig2 / nx
      a = np.array( sig2 * (1. - V) )
      while sum( abs( P - P1 ) / nx) > tol:
          b = P[:, np.newaxis] / ( P[:, np.newaxis] * V[np.newaxis, :] + a[np.newaxis, :] ) # weights
          wk = (b ** 2) * V[np.newaxis, :] # new spectral estimate
          wk = np.swapaxes( wk, 1, 2 )
          P1 = sum( wk[0, :, :] * Pk ) / np.sum( wk[0, :, :], axis = 0 )
          Ptemp = P1[:]
          P1 = P
          P = Ptemp
      return wk      



def multi_taper_ps( x, nfft, dt = 1, nw = 3 ):
  nx = len(x)
  k = min( round(2 * nw), nx )
  k = int( max( k - 1, 1) ) #number of windows
  w = float(nw) / float(dt*nx) #half-bandwidth of the dpss
  s = np.arange(0., 1. / dt, 1. / nfft / dt )
  #E, V = spectrum.dpss( nx, nw, k )
  E,V = windows.dpss(nx,nw,k,return_ratios=True)
  E = np.transpose(E)
  V = np.transpose(V)
  Pk = np.zeros( ( k, nx ) )
  for i in range( k ):
      fx = np.fft.fft( E[:, i] * x[:], nfft)
      Pk[i, :] = abs( np.fft.fft( E[:, i] * x[:], nfft) ) ** 2
      #Iteration to determine adaptive weights:
      
 
  P = adaptive_weight_P( x, Pk, nfft, V )
  be = adaptive_weight_b( x, Pk, nfft, V )
  #wk = adaptive_weight_wk( x, Pk, nfft, V )
  v = (2. * np.sum( (be** 2) * V[np.newaxis, :], axis = 2 ) ** 2 ) / (np.sum( (be ** 4) * ( V[np.newaxis, :] ** 2 ), axis = 2 ) )
  
  #cut records
  fin_ind = round((nfft + 1) / 2 + 1) #DV changes
  P = P[:fin_ind]
  s = s[:fin_ind]
  v = v[0, :fin_ind]
  
  #Chi-squared 95% confidence interval
  ci = np.zeros( ( 2, len( v ) ) )
  ci[0, :] = 1. / (1. - 2. / (9. * v ) - 1.96 * np.sqrt( 2. / ( 9 * v ) ) ) ** 3
  ci[1, :] = 1. / (1. - 2. / (9. * v ) + 1.96 * np.sqrt( 2. / ( 9 * v ) ) ) ** 3
  
  
  return P, s, ci

####################################################################

#Li and Thompson 2015: SAM ind is PC1 of [u] over all levels and lats on 1000-200 hPa, 70-20 S
# (k=19 to k=39), (j=13 to j=49)
####################################################################



momflux_vertsum = np.mean(momflux_zon_av_nd.data,axis = 1)      
momflux_vertsum_forc_a1_nd = np.mean(momflux_zon_av_forc_a1_nd.data,axis = 1)
momflux_vertsum_forc_a2_nd = np.mean(momflux_zon_av_forc_a2_nd.data,axis = 1)
momflux_vertsum_forc_a3_nd = np.mean(momflux_zon_av_forc_a3_nd.data,axis = 1)
momflux_vertsum_forc_a4_nd = np.mean(momflux_zon_av_forc_a4_nd.data,axis = 1)




pref = 2./6.37e6*np.tan(lat*np.pi/180)
#pref = 1./6.37e6*np.tan(lat*np.pi/180) #why does this work?!
#pref = 0

###Numba function###

@jit
def D_y_t(arr):
    diff_arr = arr
    for t in range(len(arr)):
        diff_arr[t,:] = -d_y(arr[t,:]) + pref*arr[t,:]
    return diff_arr

vorflux_vertsum_nd = D_y_t(momflux_vertsum)
vorflux_vertsum_forc_a1_nd =  D_y_t(momflux_vertsum_forc_a1_nd)
vorflux_vertsum_forc_a2_nd =  D_y_t(momflux_vertsum_forc_a2_nd)
vorflux_vertsum_forc_a3_nd =  D_y_t(momflux_vertsum_forc_a3_nd)
vorflux_vertsum_forc_a4_nd =  D_y_t(momflux_vertsum_forc_a4_nd)

#=============================================================================
for t in range(tlen_nd):
      vorflux_vertsum_nd[t,:] = -d_y(momflux_vertsum[t,:]) + \
    pref*momflux_vertsum[t,:]
      vorflux_vertsum_forc_a1_nd[t,:] = -d_y(momflux_vertsum_forc_a1_nd[t,:]) + \
    pref*momflux_vertsum_forc_a1_nd[t,:]
      vorflux_vertsum_forc_a2_nd[t,:] = -d_y(momflux_vertsum_forc_a2_nd[t,:]) + \
    pref*momflux_vertsum_forc_a2_nd[t,:]
      vorflux_vertsum_forc_a3_nd[t,:] = -d_y(momflux_vertsum_forc_a3_nd[t,:]) + \
    pref*momflux_vertsum_forc_a3_nd[t,:]
      vorflux_vertsum_forc_a4_nd[t,:] = -d_y(momflux_vertsum_forc_a4_nd[t,:]) + \
      pref*momflux_vertsum_forc_a4_nd[t,:]

#=============================================================================
     

#Miscellaneous: colorblind-friendly colors

CB_blue = (0./255, 107./255, 164./255)
CB_orange = (255./255, 128./255, 14./255)
CB_lightblue = (95./255, 158./255, 209./255)
CB_red = (200./255, 82./255, 0./255)

##Interpolating heating field
forc_NC = 'CW_SAM_heating.nc'
forc_data = Dataset(forc_NC)
heat_lat = forc_data.variables['lat'][:]
heat_latb = forc_data.variables['latb'][:]
heat_rate = forc_data.variables['local_heating'][0,:,:,0]
heat_press = forc_data.variables['pfull'][:]

heat_lat_interp = linterp1D(heat_lat, 1)
heat_rate_interp = np.zeros((len(heat_press), len(heat_lat_interp)), dtype='d')
heat_rate_work = np.zeros((len(press), len(lat)), dtype='d')
for p in range(len(heat_press)):
    heat_rate_interp[p,:] = linterp1D(heat_rate[p,:],1)

for p in range(len(press)):
  for l in range(len(lat)):
    L = near_ind(lat[l],heat_lat_interp)
    P = near_ind(press[p],heat_press)

    heat_rate_work[p,l] = heat_rate_interp[P,L]

#########

Damp_rates = np.zeros(len(press),dtype='d')
#Damp_rates_ext = np.zeros(len(press_ext),dtype='d')

for p in range(len(press)):
    Damp_rates[p] = damp_rate(press[p])
# for p in range(len(press_ext)):
#     Damp_rates_ext[p] = damp_rate(press_ext[p])
 
damp_nd = Damp_rates[np.newaxis,:,np.newaxis]*u_zon_av_nd  
damp_forc_a1_nd = Damp_rates[np.newaxis,:,np.newaxis]*u_zon_av_forc_a1_nd
damp_forc_a2_nd = Damp_rates[np.newaxis,:,np.newaxis]*u_zon_av_forc_a2_nd
damp_vertsum_nd = np.mean(damp_nd,axis=1)
damp_vertsum_forc_a1_nd = np.mean(damp_forc_a1_nd,axis=1)
damp_vertsum_forc_a2_nd = np.mean(damp_forc_a2_nd,axis=1)
damp_forc_a3_nd = Damp_rates[np.newaxis,:,np.newaxis]*u_zon_av_forc_a3_nd
damp_forc_a4_nd = Damp_rates[np.newaxis,:,np.newaxis]*u_zon_av_forc_a4_nd
damp_vertsum_forc_a3_nd = np.mean(damp_forc_a3_nd,axis=1)
damp_vertsum_forc_a4_nd = np.mean(damp_forc_a4_nd,axis=1)


####Analysis of forcings######
Taus = np.zeros(81,dtype='d')
Lags = np.cumsum(np.array([1]*tlen_nd))
SAM_PC1 = np.zeros(tlen_nd,dtype='d')
SAM_PC1_forc_a1 = np.zeros(tlen_nd,dtype='d')
SAM_PC1_forc_a2 = np.zeros(tlen_nd,dtype='d')
SAM_PC1_forc_a3 = np.zeros(tlen_nd,dtype='d')
SAM_PC1_forc_a4 = np.zeros(tlen_nd,dtype='d')


u_wtd = u_zon_av_vertsum_nd * np.sqrt((np.cos(lat*np.pi/180)))[np.newaxis,:]
u_wtd_forc_a1 = u_zon_av_vertsum_forc_a1_nd * np.sqrt((np.cos(lat*np.pi/180)))[np.newaxis,:]
u_wtd_forc_a2 = u_zon_av_vertsum_forc_a2_nd * np.sqrt((np.cos(lat*np.pi/180)))[np.newaxis,:]
u_wtd_forc_a3 = u_zon_av_vertsum_forc_a3_nd * np.sqrt((np.cos(lat*np.pi/180)))[np.newaxis,:]
u_wtd_forc_a4 = u_zon_av_vertsum_forc_a4_nd * np.sqrt((np.cos(lat*np.pi/180)))[np.newaxis,:]


SAM_EOF = EOF1(u_wtd)
SAM_EOF_a1 = EOF1(u_wtd_forc_a1)
SAM_EOF_a2 = EOF1(u_wtd_forc_a2)
SAM_EOF_a3 = EOF1(u_wtd_forc_a3)
SAM_EOF_a4 = EOF1(u_wtd_forc_a4)



if SAM_EOF[14] < 0:
    SAM_EOF= -SAM_EOF
if SAM_EOF_a1[14] < 0:
    SAM_EOF_a1 = -SAM_EOF_a1
if SAM_EOF_a2[14] < 0:
    SAM_EOF_a2= -SAM_EOF_a2
if SAM_EOF_a3[14] < 0:
    SAM_EOF_a3= -SAM_EOF_a3
if SAM_EOF_a4[14] < 0:
    SAM_EOF_a4= -SAM_EOF_a4


###Numba function?###

@jit(nopython=True)
def dot_t(arr_2D,arr):
    arr_dot = np.zeros(len(arr_2D),dtype='d')
    for t in range(len(arr_2D)):
        arr_dot[t] = np.sum(arr_2D[t,:]*arr)
    return arr_dot



SAM_PC1 = dot_t(u_zon_av_vertsum_nd.data,SAM_EOF)
SAM_PC1_forc_a1 = dot_t(u_zon_av_vertsum_forc_a1_nd.data,SAM_EOF_a1)
SAM_PC1_forc_a2 = dot_t(u_zon_av_vertsum_forc_a2_nd.data,SAM_EOF_a2)
SAM_PC1_forc_a3 = dot_t(u_zon_av_vertsum_forc_a3_nd.data,SAM_EOF_a3)
SAM_PC1_forc_a4 = dot_t(u_zon_av_vertsum_forc_a4_nd.data,SAM_EOF_a4)

for t in range(tlen_nd):
    SAM_PC1[t] = np.dot(u_zon_av_vertsum_nd[t,:],SAM_EOF)
    SAM_PC1_forc_a1[t] = np.dot(u_zon_av_vertsum_forc_a1_nd[t,:],SAM_EOF_a1)
    SAM_PC1_forc_a2[t] = np.dot(u_zon_av_vertsum_forc_a2_nd[t,:],SAM_EOF_a2)
    SAM_PC1_forc_a3[t] = np.dot(u_zon_av_vertsum_forc_a3_nd[t,:],SAM_EOF_a3)
    SAM_PC1_forc_a4[t] = np.dot(u_zon_av_vertsum_forc_a4_nd[t,:],SAM_EOF_a4)
   


  
Lags = np.zeros(tlen_nd,dtype='d')
for tau in range(tlen_nd):
    Lags[tau] = tau  

S = np.std(SAM_PC1)
SAM_PC1 =  (SAM_PC1 - np.mean(SAM_PC1))/S
SAM_PC1_forc_a1 =  (SAM_PC1_forc_a1 - np.mean(SAM_PC1_forc_a1))/S
SAM_PC1_forc_a2 =  (SAM_PC1_forc_a2 - np.mean(SAM_PC1_forc_a2))/S
SAM_PC1_forc_a3 =  (SAM_PC1_forc_a3 - np.mean(SAM_PC1_forc_a3))/S
SAM_PC1_forc_a4 =  (SAM_PC1_forc_a4 - np.mean(SAM_PC1_forc_a4))/S


# u_zon_av_dot_ind = np.zeros((len(press),len(lat)),dtype='d')
# u_zon_av_dot_ind_a1 = np.zeros((len(press),len(lat)),dtype='d')
# u_zon_av_dot_ind_a2 = np.zeros((len(press),len(lat)),dtype='d')
# u_zon_av_dot_ind_a3 = np.zeros((len(press),len(lat)),dtype='d')
# u_zon_av_dot_ind_a4 = np.zeros((len(press),len(lat)),dtype='d')



# for p in range(len(press)):
#     for l in range(len(lat)):
#         u_zon_av_dot_ind[p,l] = np.dot(u_zon_av_nd[:,p,l],SAM_PC1)/ np.dot(SAM_PC1,SAM_PC1)
#         u_zon_av_dot_ind_a1[p,l] = np.dot(u_zon_av_forc_a1_nd[:,p,l],SAM_PC1_forc_a1)/ np.dot(SAM_PC1_forc_a1,SAM_PC1_forc_a1)
#         u_zon_av_dot_ind_a2[p,l] = np.dot(u_zon_av_forc_a2_nd[:,p,l],SAM_PC1_forc_a2)/ np.dot(SAM_PC1_forc_a2,SAM_PC1_forc_a2)
#         u_zon_av_dot_ind_a3[p,l] = np.dot(u_zon_av_forc_a3_nd[:,p,l],SAM_PC1_forc_a3)/ np.dot(SAM_PC1_forc_a3,SAM_PC1_forc_a3)
#         u_zon_av_dot_ind_a4[p,l] = np.dot(u_zon_av_forc_a4_nd[:,p,l],SAM_PC1_forc_a4)/ np.dot(SAM_PC1_forc_a4,SAM_PC1_forc_a4)
        
        
        
##Buoyancy,heat and EP flux analysis ###

# vorflux_zon_av_nd = np.zeros(np.shape(momflux_zon_av_nd),dtype='d')
# vorflux_zon_av_forc_a1_nd = np.zeros(np.shape(momflux_zon_av_nd),dtype='d')
# vorflux_zon_av_forc_a2_nd = np.zeros(np.shape(momflux_zon_av_nd),dtype='d')
# vorflux_zon_av_forc_a3_nd = np.zeros(np.shape(momflux_zon_av_nd),dtype='d')
# vorflux_zon_av_forc_a4_nd = np.zeros(np.shape(momflux_zon_av_nd),dtype='d')

# vorflux_zon_av_nd[:,:,0:-1] = -np.diff(momflux_zon_av_nd[:,:,:],axis=2)/del_y + pref[0:-1]*momflux_zon_av_nd[:,:,0:-1]
# vorflux_zon_av_forc_a1_nd[:,:,0:-1] = -np.diff(momflux_zon_av_forc_a1_nd[:,:,:],axis=2)/del_y + pref[0:-1]*momflux_zon_av_forc_a1_nd[:,:,0:-1]
# vorflux_zon_av_forc_a2_nd[:,:,0:-1] = -np.diff(momflux_zon_av_forc_a2_nd[:,:,:],axis=2)/del_y + pref[0:-1]*momflux_zon_av_forc_a2_nd[:,:,0:-1]
# vorflux_zon_av_forc_a3_nd[:,:,0:-1] = -np.diff(momflux_zon_av_forc_a3_nd[:,:,:],axis=2)/del_y + pref[0:-1]*momflux_zon_av_forc_a3_nd[:,:,0:-1]
# vorflux_zon_av_forc_a4_nd[:,:,0:-1] = -np.diff(momflux_zon_av_forc_a4_nd[:,:,:],axis=2)/del_y + pref[0:-1]*momflux_zon_av_forc_a4_nd[:,:,0:-1]

#Gradient method (i.e. 2nd order finite difference) actually looks noisier!
# vorflux_zon_av_nd = -np.gradient(momflux_zon_av_nd,axis=2)/del_y + pref*momflux_zon_av_nd
# vorflux_zon_av_forc_a1_nd = -np.gradient(momflux_zon_av_forc_a1_nd,axis=2)/del_y + pref*momflux_zon_av_forc_a1_nd
# vorflux_zon_av_forc_a2_nd = -np.gradient(momflux_zon_av_forc_a2_nd,axis=2)/del_y + pref*momflux_zon_av_forc_a2_nd
# vorflux_zon_av_forc_a3_nd = -np.gradient(momflux_zon_av_forc_a3_nd,axis=2)/del_y + pref*momflux_zon_av_forc_a3_nd
# vorflux_zon_av_forc_a4_nd = -np.gradient(momflux_zon_av_forc_a4_nd,axis=2)/del_y + pref*momflux_zon_av_forc_a4_nd


        

        
buoy = g/T_star*(temp_zon_av_nd*(press[-1]/press[:,np.newaxis])**(kappa) - T_star)
buoy_a1 = g/T_star*(temp_zon_av_forc_a1_nd*(press[-1]/press[:,np.newaxis])**(kappa) - T_star)
buoy_a2 = g/T_star*(temp_zon_av_forc_a2_nd*(press[-1]/press[:,np.newaxis])**(kappa) - T_star)
buoy_a3 = g/T_star*(temp_zon_av_forc_a3_nd*(press[-1]/press[:,np.newaxis])**(kappa) - T_star)
buoy_a4 = g/T_star*(temp_zon_av_forc_a4_nd*(press[-1]/press[:,np.newaxis])**(kappa) - T_star)


# vorflux_ccf = np.zeros((tlen_nd,len(press),len(lat)),dtype='d')
# vorflux_ccf_a1 = np.zeros((tlen_nd,len(press),len(lat)),dtype='d')
# vorflux_ccf_a2 = np.zeros((tlen_nd,len(press),len(lat)),dtype='d')
# vorflux_ccf_a3 = np.zeros((tlen_nd,len(press),len(lat)),dtype='d')

# for l in range(len(lat)):
#     for p in range(len(press)):
  
#       vorflux_ccf[:,p,l] = CCF_full(SAM_PC1,vorflux_zon_av_nd[:,p,l])/CCF_full(SAM_PC1,SAM_PC1)
#       vorflux_ccf_a1[:,p,l] = CCF_full(SAM_PC1_forc_a1,vorflux_zon_av_forc_a1_nd[:,p,l])/CCF_full(SAM_PC1_forc_a1,SAM_PC1_forc_a1)
     # vorflux_ccf_a2[:,p,l] = CCF_full(SAM_PC1_forc_a2,vorflux_a2[:,p,l])/CCF_full(SAM_PC1_forc_a2,SAM_PC1_forc_a2)
     # vorflux_ccf_a3[:,p,l] = CCF_full(SAM_PC1_forc_a3,vorflux_a3[:,p,l])/CCF_full(SAM_PC1_forc_a3,SAM_PC1_forc_a3)
 


# ###Define function with numba to do this faster###
D_zz = np.array(D_z.data,dtype='d')
latlen = len(lat)
#@jit(nopython=True)
@jit
def D_z_t_l(arr):
    arr_diff = np.zeros(np.shape(arr),dtype='d')
    for t in range(len(arr)):
        for l in range(latlen):
            arr_diff[t,:,l] = np.matmul(D_zz,arr[t,:,l])
    return arr_diff
buoyy = np.array(buoy.data,dtype='d')
buoyy_a1 = np.array(buoy_a1.data,dtype='d')
buoyy_a2 = np.array(buoy_a2.data,dtype='d')
buoyy_a3 = np.array(buoy_a3.data,dtype='d')
buoyy_a4 = np.array(buoy_a4.data,dtype='d')

N_sq = D_z_t_l(buoyy)
N_sq_a1 = D_z_t_l(buoyy_a1)
N_sq_a2 = D_z_t_l(buoyy_a2)
N_sq_a3 = D_z_t_l(buoyy_a3)
N_sq_a4 = D_z_t_l(buoyy_a4)



###Define function with numba to do this faster###
presslen = len(press)
D_yy = np.array(D_y.data,dtype='d')
#@jit(nopython=True)
@jit
def D_y_p(arr):
    arr_diff = arr
    for t in range(len(arr)):
      for p in range(presslen):
        arr_diff[t,p,:] = np.matmul(D_y,arr[t,p,:])
    return arr_diff
temp_zon_av_ndd = np.array(temp_zon_av_nd.data,dtype='d')
temp_zon_av_forc_a1_ndd = np.array(temp_zon_av_forc_a1_nd.data,dtype='d')
temp_zon_av_forc_a2_ndd = np.array(temp_zon_av_forc_a2_nd.data,dtype='d')
temp_zon_av_forc_a3_ndd = np.array(temp_zon_av_forc_a3_nd.data,dtype='d')
temp_zon_av_forc_a4_ndd = np.array(temp_zon_av_forc_a4_nd.data,dtype='d')

dT_dy = D_y_p(temp_zon_av_ndd)
dT_dy_a1 = D_y_p(temp_zon_av_forc_a1_ndd)
dT_dy_a2 = D_y_p(temp_zon_av_forc_a2_ndd)
dT_dy_a3 = D_y_p(temp_zon_av_forc_a3_ndd)
dT_dy_a4 = D_y_p(temp_zon_av_forc_a4_ndd)




 
###########Baroclinicity calculations#########


d_P = np.zeros((len(lat)*len(press),len(lat)*len(press)),dtype='d')
d_l = np.zeros((len(lat)*len(press),len(lat)*len(press)),dtype='d')
press_arr = np.zeros((len(press),len(lat)), dtype='d')
baroclin = np.zeros((tlen_nd,len(press),len(lat)),dtype='d')
baroclin_forc_a1 = np.zeros((tlen_nd,len(press),len(lat)),dtype='d')
baroclin_forc_a2 = np.zeros((tlen_nd,len(press),len(lat)),dtype='d')
baroclin_forc_a3 = np.zeros((tlen_nd,len(press),len(lat)),dtype='d')
baroclin_forc_a4 = np.zeros((tlen_nd,len(press),len(lat)),dtype='d')



for p in range(len(press)):
    for l in range(len(lat)):
        press_arr[p,l] = press[p]
press_arr_uw = unwind(press_arr)

a = 6.37e6
del_p = (press[1] - press[0])*100
del_l = np.pi/180*(lat[1] - lat[0])

HS_m = 8500.
g = 9.81
T_star = 255.
Pre = 2./7 * 1./24. * 1./9.8e-3 * 0.75 * (6.37e6*30*np.pi/180)**2

for j in range(len(lat)*len(press)): #Derivative operators on unwound arrays
    if j < len(lat)*(len(press)-1) and j > len(lat): #pressure
      d_P[j,j + len(lat)] = 1.
      d_P[j,j - len(lat)] = -1.

    if j % len(lat) != len(lat) -1 and j % len(lat) != 0: #latitude
        d_l[j,j+1] = 1.
        d_l[j,j-1] = -1.

d_P *= 1./(2*del_p)
d_l *= 1./(2*del_l)
# d_z = -press_arr_uw[:,np.newaxis]/HS_m * d_P 
# d_y = 1./a * d_l



        
Omega = 7.27e-5
fs = 2*Omega*np.sin(lat*np.pi/180)
###

momflux_vertsum_nd = np.mean(momflux_zon_av_nd,axis = 1)
momflux_vertsum_forc_a1_nd = np.mean(momflux_zon_av_forc_a1_nd,axis = 1)
momflux_vertsum_forc_a2_nd = np.mean(momflux_zon_av_forc_a2_nd,axis = 1)
momflux_vertsum_forc_a3_nd = np.mean(momflux_zon_av_forc_a3_nd,axis = 1)
momflux_vertsum_forc_a4_nd = np.mean(momflux_zon_av_forc_a4_nd,axis = 1)

u_zon_av_reg_SAM = np.zeros((len(press),len(lat)),dtype='d')
u_zon_av_reg_SAM_a1 = np.zeros((len(press),len(lat)),dtype='d')
u_zon_av_reg_SAM_a2 = np.zeros((len(press),len(lat)),dtype='d')
u_zon_av_reg_SAM_a3 = np.zeros((len(press),len(lat)),dtype='d')
u_zon_av_reg_SAM_a4 = np.zeros((len(press),len(lat)),dtype='d')


temp_zon_av_reg_SAM = np.zeros((len(press),len(lat)),dtype='d')
temp_zon_av_reg_SAM_a1 = np.zeros((len(press),len(lat)),dtype='d')
temp_zon_av_reg_SAM_a2 = np.zeros((len(press),len(lat)),dtype='d')
temp_zon_av_reg_SAM_a3 = np.zeros((len(press),len(lat)),dtype='d')
temp_zon_av_reg_SAM_a4 = np.zeros((len(press),len(lat)),dtype='d')

dT_dy_reg_SAM = np.zeros((len(press),len(lat)),dtype='d')
dT_dy_reg_SAM_a1 = np.zeros((len(press),len(lat)),dtype='d')
dT_dy_reg_SAM_a2 = np.zeros((len(press),len(lat)),dtype='d')
dT_dy_reg_SAM_a3 = np.zeros((len(press),len(lat)),dtype='d')
dT_dy_reg_SAM_a4 = np.zeros((len(press),len(lat)),dtype='d')

N_sq_reg_SAM = np.zeros((len(press),len(lat)),dtype='d')
N_sq_reg_SAM_a1 = np.zeros((len(press),len(lat)),dtype='d')
N_sq_reg_SAM_a2 = np.zeros((len(press),len(lat)),dtype='d')
N_sq_reg_SAM_a3 = np.zeros((len(press),len(lat)),dtype='d')
N_sq_reg_SAM_a4 = np.zeros((len(press),len(lat)),dtype='d')


vorflux_zon_av_reg_SAM = np.zeros((len(press),len(lat)),dtype='d')
vorflux_zon_av_reg_SAM_a1 = np.zeros((len(press),len(lat)),dtype='d')
# vorflux_zon_av_reg_SAM_a2 = np.zeros((len(press),len(lat)),dtype='d')
# vorflux_zon_av_reg_SAM_a3 = np.zeros((len(press),len(lat)),dtype='d')
# vorflux_zon_av_reg_SAM_a4 = np.zeros((len(press),len(lat)),dtype='d')


#norm_sq_PC1 = CCF_full(SAM_PC1[0:tlen_nd-1],SAM_PC1[0:tlen_nd-1])
norm_sq_PC1 = CCF_full(SAM_PC1,SAM_PC1)
norm_sq_PC1_a1 = CCF_full(SAM_PC1_forc_a1,SAM_PC1_forc_a1)
norm_sq_PC1_a2 = CCF_full(SAM_PC1_forc_a2,SAM_PC1_forc_a2)
norm_sq_PC1_a3 = CCF_full(SAM_PC1_forc_a3,SAM_PC1_forc_a3)
norm_sq_PC1_a4 = CCF_full(SAM_PC1_forc_a4,SAM_PC1_forc_a4)

    
for l in range(len(lat)):
    for p in range(len(press)):
#                 u_zon_av_reg_SAM[p,l] = CCF_full(SAM_PC1,u_zon_av_nd[:,p,l])[0]/CCF_full(SAM_PC1,SAM_PC1)[0]
#                 u_zon_av_reg_SAM_a1[p,l] = CCF_full(SAM_PC1_forc_a1,u_zon_av_forc_a1_nd[:,p,l])[0]/CCF_full(SAM_PC1_forc_a1, SAM_PC1_forc_a1)[0]
#                 u_zon_av_reg_SAM_a2[p,l] = CCF_full(SAM_PC1_forc_a2,u_zon_av_forc_a2_nd[:,p,l])[0]/CCF_full(SAM_PC1_forc_a2, SAM_PC1_forc_a2)[0]
#                 u_zon_av_reg_SAM_a3[p,l] = CCF_full(SAM_PC1_forc_a3,u_zon_av_forc_a3_nd[:,p,l])[0]/CCF_full(SAM_PC1_forc_a3, SAM_PC1_forc_a3)[0]
#                 u_zon_av_reg_SAM_a4[p,l] = CCF_full(SAM_PC1_forc_a4,u_zon_av_forc_a4_nd[:,p,l])[0]/CCF_full(SAM_PC1_forc_a4, SAM_PC1_forc_a4)[0]
            

                temp_zon_av_reg_SAM[p,l] = CCF_full(SAM_PC1,temp_zon_av_nd[:,p,l])[0]/CCF_full(SAM_PC1,SAM_PC1)[0]
                temp_zon_av_reg_SAM_a1[p,l] = CCF_full(SAM_PC1_forc_a1,temp_zon_av_forc_a1_nd[:,p,l])[0]/CCF_full(SAM_PC1_forc_a1, SAM_PC1_forc_a1)[0]
#                 temp_zon_av_reg_SAM_a2[p,l] = CCF_full(SAM_PC1_forc_a2,temp_zon_av_forc_a2_nd[:,p,l])[0]/CCF_full(SAM_PC1_forc_a2, SAM_PC1_forc_a2)[0]
#                 temp_zon_av_reg_SAM_a3[p,l] = CCF_full(SAM_PC1_forc_a3,temp_zon_av_forc_a3_nd[:,p,l])[0]/CCF_full(SAM_PC1_forc_a3, SAM_PC1_forc_a3)[0]
#                 temp_zon_av_reg_SAM_a4[p,l] = CCF_full(SAM_PC1_forc_a4,temp_zon_av_forc_a4_nd[:,p,l])[0]/CCF_full(SAM_PC1_forc_a4, SAM_PC1_forc_a4)[0]
                
#                 dT_dy_reg_SAM[p,l] = 1e6*CCF_full(SAM_PC1,dT_dy[:,p,l])[0]/CCF_full(SAM_PC1,SAM_PC1)[0]
#                 dT_dy_reg_SAM_a1[p,l] = 1e6*CCF_full(SAM_PC1_forc_a1,dT_dy_a1[:,p,l])[0]/CCF_full(SAM_PC1_forc_a1, SAM_PC1_forc_a1)[0]
#                 dT_dy_reg_SAM_a2[p,l] = 1e6*CCF_full(SAM_PC1_forc_a2,dT_dy_a2[:,p,l])[0]/CCF_full(SAM_PC1_forc_a2, SAM_PC1_forc_a2)[0]
#                 dT_dy_reg_SAM_a3[p,l] = 1e6*CCF_full(SAM_PC1_forc_a3,dT_dy_a3[:,p,l])[0]/CCF_full(SAM_PC1_forc_a3, SAM_PC1_forc_a3)[0]
#                 dT_dy_reg_SAM_a4[p,l] = 1e6*CCF_full(SAM_PC1_forc_a4,dT_dy_a4[:,p,l])[0]/CCF_full(SAM_PC1_forc_a4, SAM_PC1_forc_a4)[0]
                
#                 N_sq_reg_SAM[p,l] = CCF_full(SAM_PC1,N_sq[:,p,l])[0]/CCF_full(SAM_PC1,SAM_PC1)[0]
#                 N_sq_reg_SAM_a1[p,l] = CCF_full(SAM_PC1_forc_a1,N_sq_a1[:,p,l])[0]/CCF_full(SAM_PC1_forc_a1, SAM_PC1_forc_a1)[0]
#                 N_sq_reg_SAM_a2[p,l] = CCF_full(SAM_PC1_forc_a2,N_sq_a2[:,p,l])[0]/CCF_full(SAM_PC1_forc_a2, SAM_PC1_forc_a2)[0]
#                 N_sq_reg_SAM_a3[p,l] = CCF_full(SAM_PC1_forc_a3,N_sq_a3[:,p,l])[0]/CCF_full(SAM_PC1_forc_a3, SAM_PC1_forc_a3)[0]
#                 N_sq_reg_SAM_a4[p,l] = CCF_full(SAM_PC1_forc_a4,N_sq_a4[:,p,l])[0]/CCF_full(SAM_PC1_forc_a4, SAM_PC1_forc_a4)[0]
                
            
                   
#####


col_a1 = (31/255., 119/255., 180/255., 0.6)
col_a2 = (255/255., 152/255., 150/255., 0.7)
col_a3 = (31/255., 119/255., 180/255., 1.0)
col_a4 = (214/255., 39/255., 40/255., 0.7)


levs=33
RdBu_r = cm.get_cmap('RdBu_r',levs)
newcolors = RdBu_r(np.linspace(0, 1, levs))
mdpnt = round(levs/2)
whtwdth = 2
newcolors[mdpnt-whtwdth:mdpnt+whtwdth+1] = [0., 0., 0., 0.]
newcmap = ListedColormap(newcolors)

PuOr= cm.get_cmap('PuOr_r',levs)
newcolors = PuOr(np.linspace(0, 1, levs))
mdpnt = round(levs/2)
whtwdth = 2
newcolors[mdpnt-whtwdth:mdpnt+whtwdth+1] = [0., 0., 0., 0.]
newcmap2 = ListedColormap(newcolors)

levs=73
RdBu_r = cm.get_cmap('RdBu_r',levs)
newcolors = RdBu_r(np.linspace(0, 1, levs))
mdpnt = round(levs/2)
whtwdth = 2
newcolors[mdpnt-whtwdth:mdpnt+whtwdth+1] = [0., 0., 0., 0.]
newcmap3 = ListedColormap(newcolors)

levs=73
PuOr= cm.get_cmap('PuOr_r',levs)
newcolors = PuOr(np.linspace(0, 1, levs))
mdpnt = round(levs/2)
whtwdth = 2
newcolors[mdpnt-whtwdth:mdpnt+whtwdth+1] = [0., 0., 0., 0.]
newcmap4 = ListedColormap(newcolors)



#### Supplementary 4-panel figs: T, dT/dy (resid.), dT/dy ( no resid.), u, N_sq, ###

# fig = plt.figure( figsize = (23, 19) ) #GRL proportions?
# fig.subplots_adjust(left =0.1, bottom = 0.1, top = 0.95, hspace = 0.25, wspace = 0.2, right = 0.85)
# bigsize=47
# smallsize=40


# num_levs=103
# base_var = u_zon_av_reg_SAM_a4 - u_zon_av_reg_SAM
# Var = u_zon_av_reg_SAM_a1 - u_zon_av_reg_SAM
# bound = 1.081*max(abs(base_var.min()),abs(base_var.max()))
# levs_u_zon_av = np.zeros(num_levs,dtype='d')
# for l in range(num_levs):
#     levs_u_zon_av[l] = round(-bound + 2*bound*l/(num_levs-1),2)


# a1_ax = plt.subplot(2,2,1)
# col = a1_ax.contourf(lat,press,Var, levels=levs_u_zon_av,cmap=newcmap4)
# a1_ax.contour(lat,press, u_zon_av_reg_SAM, colors='black',levels=6)
# #plt.xlabel('Latitude ',fontsize=bigsize)
# plt.ylabel('Pressure [hPa]',fontsize=bigsize)
# plt.gca().invert_yaxis()
# plt.title("a=1",fontsize=bigsize)
# plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
# plt.yticks([200, 400, 600, 800],fontsize = smallsize)
# a1_ax.tick_params(axis='x',length=10, width = 3)
# a1_ax.tick_params(axis='y',length=10, width = 3)
# plt.xticks(color='w',fontsize=1)


# Var = u_zon_av_reg_SAM_a2 - u_zon_av_reg_SAM

# a2_ax = plt.subplot(2,2,2)
# col = a2_ax.contourf(lat,press,Var, levels=levs_u_zon_av ,cmap=newcmap4)
# a2_ax.contour(lat,press, u_zon_av_reg_SAM, colors='black',levels=6)
# #plt.xlabel('Latitude ',fontsize=bigsize)
# #plt.ylabel('Pressure [hPa]',fontsize=bigsize)
# plt.gca().invert_yaxis()
# plt.title("a=-1",fontsize=bigsize)
# plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
# plt.yticks([200, 400, 600, 800],fontsize = smallsize)
# plt.yticks(color='w',fontsize=1)
# a2_ax.tick_params(axis='x',length=10, width = 3)
# a2_ax.tick_params(axis='y',length=10, width = 3)
# plt.xticks(color='w',fontsize=1)


# Var = u_zon_av_reg_SAM_a3 - u_zon_av_reg_SAM

# a3_ax = plt.subplot(2,2,3)
# col = a3_ax.contourf(lat,press,Var, levels=levs_u_zon_av,cmap=newcmap4)
# a3_ax.contour(lat,press, u_zon_av_reg_SAM, colors='black',levels=6)
# plt.xlabel('Latitude ',fontsize=bigsize)
# plt.ylabel('Pressure [hPa]',fontsize=bigsize)
# plt.gca().invert_yaxis()
# plt.title("a=3",fontsize=bigsize)
# plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
# plt.yticks([200, 400, 600, 800],fontsize = smallsize)
# a3_ax.tick_params(axis='x',length=10, width = 3)
# a3_ax.tick_params(axis='y',length=10, width = 3)


# Var = u_zon_av_reg_SAM_a4 - u_zon_av_reg_SAM

# a4_ax = plt.subplot(2,2,4)
# col = a4_ax.contourf(lat,press,Var, levels=levs_u_zon_av, cmap=newcmap4)
# a4_ax.contour(lat,press, u_zon_av_reg_SAM, colors='black',levels=6)
# plt.xlabel('Latitude ',fontsize=bigsize)
# plt.title("a=-3",fontsize=bigsize)
# plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
# plt.yticks([200, 400, 600, 800],fontsize = smallsize)
# plt.yticks(color='w',fontsize=1)
# #plt.ylabel('Pressure [hPa]',fontsize=bigsize)
# plt.gca().invert_yaxis()
# a4_ax.tick_params(axis='x',length=10, width = 3)
# a4_ax.tick_params(axis='y',length=10, width = 3)
# plt.yticks(color='w',fontsize=1)
# cbar = fig.add_axes([0.86, 0.1, 0.02, 0.85]) 
# cb = fig.colorbar(col, cax=cbar)
# cb.ax.tick_params(labelsize=smallsize)
# cb.set_label('[$10^{-3}$ K km $^{-1}$]',fontsize=smallsize)
# #cb.set_ticks([-1.2,-0.9,-0.6,-0.3,0.0,0.3,0.6,0.9,1.2])
# plt.savefig('ACRE_suppfig6.png')
# plt.savefig('ACRE_suppfig6.pdf')

#####Autocorr accounting ######



vorflux_dot_eof = np.zeros(tlen_nd,dtype='d')
vorflux_dot_eof_forc_a1 = np.zeros(tlen_nd,dtype='d')
vorflux_dot_eof_forc_a2 = np.zeros(tlen_nd,dtype='d')
vorflux_dot_eof_forc_a3 = np.zeros(tlen_nd,dtype='d')
vorflux_dot_eof_forc_a4 = np.zeros(tlen_nd,dtype='d')

damp_dot_eof = np.zeros(tlen_nd,dtype='d')
damp_dot_eof_forc_a1 = np.zeros(tlen_nd,dtype='d')
damp_dot_eof_forc_a2 = np.zeros(tlen_nd,dtype='d')
damp_dot_eof_forc_a3 = np.zeros(tlen_nd,dtype='d')
damp_dot_eof_forc_a4 = np.zeros(tlen_nd,dtype='d')


fv_dot_eof = np.zeros(tlen_nd,dtype='d')
fv_dot_eof_forc_a1 = np.zeros(tlen_nd,dtype='d')
fv_dot_eof_forc_a2 = np.zeros(tlen_nd,dtype='d')
fv_dot_eof_forc_a3 = np.zeros(tlen_nd,dtype='d')
fv_dot_eof_forc_a4 = np.zeros(tlen_nd,dtype='d')


f = 2*7.27e-5*np.sin(lat*np.pi/180.)


vorflux_dot_eof = dot_t(vorflux_vertsum_nd,SAM_EOF)  
vorflux_dot_eof_forc_a1 = dot_t(vorflux_vertsum_forc_a1_nd,SAM_EOF_a1)
vorflux_dot_eof_forc_a2 = dot_t(vorflux_vertsum_forc_a2_nd,SAM_EOF_a2)
vorflux_dot_eof_forc_a3 = dot_t(vorflux_vertsum_forc_a3_nd,SAM_EOF_a3)
vorflux_dot_eof_forc_a4 = dot_t(vorflux_vertsum_forc_a4_nd,SAM_EOF_a4)

fv_dot_eof = dot_t(f[np.newaxis,:]*v_zon_av_vertsum_nd,SAM_EOF)  
fv_dot_eof_forc_a1 = dot_t(f[np.newaxis,:]*v_zon_av_vertsum_forc_a1_nd,SAM_EOF_a1)
fv_dot_eof_forc_a2 = dot_t(f[np.newaxis,:]*v_zon_av_vertsum_forc_a2_nd,SAM_EOF_a2)
fv_dot_eof_forc_a3 = dot_t(f[np.newaxis,:]*v_zon_av_vertsum_forc_a3_nd,SAM_EOF_a3)
fv_dot_eof_forc_a4 = dot_t(f[np.newaxis,:]*v_zon_av_vertsum_forc_a4_nd,SAM_EOF_a4)

damp_dot_eof = dot_t(damp_vertsum_nd.data,SAM_EOF)  
damp_dot_eof_forc_a1 = dot_t(damp_vertsum_forc_a1_nd.data,SAM_EOF_a1)
damp_dot_eof_forc_a2 = dot_t(damp_vertsum_forc_a2_nd.data,SAM_EOF_a2)
damp_dot_eof_forc_a3 = dot_t(damp_vertsum_forc_a3_nd.data,SAM_EOF_a3)
damp_dot_eof_forc_a4 = dot_t(damp_vertsum_forc_a4_nd.data,SAM_EOF_a4)


vorflux_dot_eof = vorflux_dot_eof - np.mean(vorflux_dot_eof)
vorflux_dot_eof_norm = vorflux_dot_eof/S * 86400
vorflux_dot_eof_forc_a1 = vorflux_dot_eof_forc_a1 - np.mean(vorflux_dot_eof_forc_a1)
vorflux_dot_eof_forc_a2 = vorflux_dot_eof_forc_a2 - np.mean(vorflux_dot_eof_forc_a2)
vorflux_dot_eof_norm_forc_a1 = vorflux_dot_eof_forc_a1/S * 86400
vorflux_dot_eof_norm_forc_a2 = vorflux_dot_eof_forc_a2/S * 86400
vorflux_dot_eof_forc_a3 = vorflux_dot_eof_forc_a3 - np.mean(vorflux_dot_eof_forc_a3)
vorflux_dot_eof_forc_a4 = vorflux_dot_eof_forc_a4 - np.mean(vorflux_dot_eof_forc_a4)
vorflux_dot_eof_norm_forc_a3 = vorflux_dot_eof_forc_a3/S * 86400
vorflux_dot_eof_norm_forc_a4 = vorflux_dot_eof_forc_a4/S * 86400

damp_dot_eof = damp_dot_eof - np.mean(damp_dot_eof)
damp_dot_eof_norm = damp_dot_eof/S
damp_dot_eof_forc_a1 = damp_dot_eof_forc_a1 - np.mean(damp_dot_eof_forc_a1)
damp_dot_eof_forc_a2 = damp_dot_eof_forc_a2 - np.mean(damp_dot_eof_forc_a2)
damp_dot_eof_norm_forc_a1 = damp_dot_eof_forc_a1/S
damp_dot_eof_norm_forc_a2 = damp_dot_eof_forc_a2/S
damp_dot_eof_forc_a3 = damp_dot_eof_forc_a3 - np.mean(damp_dot_eof_forc_a3)
damp_dot_eof_forc_a4 = damp_dot_eof_forc_a4 - np.mean(damp_dot_eof_forc_a4)
damp_dot_eof_norm_forc_a3 = damp_dot_eof_forc_a3/S
damp_dot_eof_norm_forc_a4 = damp_dot_eof_forc_a4/S


fv_dot_eof = fv_dot_eof - np.mean(fv_dot_eof)
fv_dot_eof_norm = fv_dot_eof/S * 86400
fv_dot_eof_forc_a1 = fv_dot_eof_forc_a1 - np.mean(fv_dot_eof_forc_a1)
fv_dot_eof_forc_a2 = fv_dot_eof_forc_a2 - np.mean(fv_dot_eof_forc_a2)
fv_dot_eof_norm_forc_a1 = fv_dot_eof_forc_a1/S * 86400
fv_dot_eof_norm_forc_a2 = fv_dot_eof_forc_a2/S * 86400
fv_dot_eof_forc_a3 = fv_dot_eof_forc_a3 - np.mean(fv_dot_eof_forc_a3)
fv_dot_eof_forc_a4 = fv_dot_eof_forc_a4 - np.mean(fv_dot_eof_forc_a4)
fv_dot_eof_norm_forc_a3 = fv_dot_eof_forc_a3/S * 86400
fv_dot_eof_norm_forc_a4 = fv_dot_eof_forc_a4/S * 86400


# SAM_PC1_filt = specfilt_Butterworth(SAM_PC1,1/50, -3)
# SAM_PC1_filt_a1 = specfilt_Butterworth(SAM_PC1_forc_a1,1/50, -3)
# SAM_PC1_filt_a2 = specfilt_Butterworth(SAM_PC1_forc_a2,1/50, -3)
# SAM_PC1_filt_a3 = specfilt_Butterworth(SAM_PC1_forc_a3,1/50, -3)
# SAM_PC1_filt_a4 = specfilt_Butterworth(SAM_PC1_forc_a4,1/50, -3)



##### Cumulative average of simplified SAM index ####

# SAM_simp = u_zon_av_nd[:,16,14] - u_zon_av_nd[:,16,29]
# SAM_simp_a1 = u_zon_av_forc_a1_nd[:,16,14] - u_zon_av_forc_a1_nd[:,16,29]
# SAM_simp_a2 = u_zon_av_forc_a2_nd[:,16,14] - u_zon_av_forc_a2_nd[:,16,29]
# SAM_simp_a3 = u_zon_av_forc_a3_nd[:,16,14] - u_zon_av_forc_a3_nd[:,16,29]
# SAM_simp_a4 = u_zon_av_forc_a4_nd[:,16,14] - u_zon_av_forc_a4_nd[:,16,29]

# S = np.std(SAM_simp)
# M = np.mean(SAM_simp)
# SAM_simp =  (SAM_simp - np.mean(SAM_simp))/S
# SAM_simp_a1 =  (SAM_simp_a1 - M)/S
# SAM_simp_a2 =  (SAM_simp_a2 - M)/S
# SAM_simp_a3 =  (SAM_simp_a3 - M)/S
# SAM_simp_a4 =  (SAM_simp_a4 - M)/S



# sm_SAM_simp = cumav(SAM_simp)
# sm_SAM_simp_a1 = cumav(SAM_simp_a1)
# sm_SAM_simp_a2 = cumav(SAM_simp_a2)
# sm_SAM_simp_a3 = cumav(SAM_simp_a3)
# sm_SAM_simp_a4 = cumav(SAM_simp_a4)
# zero_line = np.zeros(tlen_nd,dtype='d')

# fig,ax = plt.subplots()
# ax.plot(sm_SAM_simp,color='black',label='control')
# ax.plot(sm_SAM_simp_a1,color=CB_orange,label='ACRE (a=1)')
# ax.plot(sm_SAM_simp_a2,color=CB_lightblue,label='ACRE (a=-1)')
# ax.plot(sm_SAM_simp_a3,color=CB_red,label='ACRE (a=3)')
# ax.plot(sm_SAM_simp_a4,color=CB_blue,label='ACRE (a=-3)')
# ax.plot(zero_line,color='black',linestyle='dashed')
# ax.legend(frameon=False,loc='upper right')
# plt.xlabel('Time (days)')
# plt.ylabel('SAM index')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.ylim(-0.5,0.5)
# plt.savefig('SAM_index_vs_time_test.png')






######### Damping & Baroclin Feedback ##########

WCs = np.array([1./10,1./20,1./30,1./50.,1./100,1./150,1./200,1./300,1./400,1./600])
WCs = np.array([1./1,1/5.,1/10.,1./20,1./30.,1/10.,1/100.,1/100.,1./300])


for i in range(len(WCs)):
  WC = WCs[i]
  O = -7
  
    
#####Spectral Analysis, Key figure 2 #######

Freqs = np.fft.fftfreq(tlen_nd,d=1.0)
W = 16
tau_inv = 0.105



AC_SAM = ACF_full(SAM_PC1) 
AC_SAM_forc_a1 = ACF_full(SAM_PC1_forc_a1)
AC_SAM_forc_a2 = ACF_full(SAM_PC1_forc_a2) 
AC_SAM_forc_a3 = ACF_full(SAM_PC1_forc_a3) 
AC_SAM_forc_a4 = ACF_full(SAM_PC1_forc_a4) 


# eddy_noise_pspec = sp.signal.welch(eddy_noise)
# eddy_noise_a1_pspec = sp.signal.welch(eddy_noise_a1)
# eddy_noise_a2_pspec = sp.signal.welch(eddy_noise_a2)
# eddy_noise_a3_pspec = sp.signal.welch(eddy_noise_a3)
# eddy_noise_a4_pspec = sp.signal.welch(eddy_noise_a4)

W=45
# mt_freqs = multi_taper_ps(eddy_noise[0:-1],tlen_nd-1,nw=W)[1]
# eddy_noise_pspec = multi_taper_ps(eddy_noise[0:-1],tlen_nd-1,nw=W)[0]
# eddy_noise_a1_pspec = multi_taper_ps(eddy_noise_a1[0:-1],tlen_nd-1,nw=W)[0]
# eddy_noise_a2_pspec = multi_taper_ps(eddy_noise_a2[0:-1],tlen_nd-1,nw=W)[0]
# eddy_noise_a3_pspec = multi_taper_ps(eddy_noise_a3[0:-1],tlen_nd-1,nw=W)[0]
# eddy_noise_a4_pspec = multi_taper_ps(eddy_noise_a4[0:-1],tlen_nd-1,nw=W)[0]


# eddy_noise_pspec = np.abs(sp.fft.fft(eddy_noise))**2
# eddy_noise_a1_pspec = np.abs(sp.fft.fft(eddy_noise_a1))**2
# eddy_noise_a2_pspec = np.abs(sp.fft.fft(eddy_noise_a2))**2
# eddy_noise_a3_pspec = np.abs(sp.fft.fft(eddy_noise_a3))**2
# eddy_noise_a4_pspec = np.abs(sp.fft.fft(eddy_noise_a4))**2

# offline_pspec = eddy_noise_pspec/((2*np.pi*Freqs)**2 + tau_inv**2)
# offline_pspec_a1 = eddy_noise_a1_pspec/((2*np.pi*Freqs)**2 + tau_inv**2)
# offline_pspec_a2 = eddy_noise_a2_pspec/((2*np.pi*Freqs)**2 + tau_inv**2)
# offline_pspec_a3 = eddy_noise_a3_pspec/((2*np.pi*Freqs)**2 + tau_inv**2)
# offline_pspec_a4 = eddy_noise_a4_pspec/((2*np.pi*Freqs)**2 + tau_inv**2)
# offline_AC = sp.fft.ifft(offline_pspec)
# offline_AC_a1 = sp.fft.ifft(offline_pspec_a1)
# offline_AC_a2 = sp.fft.ifft(offline_pspec_a2)
# offline_AC_a3 = sp.fft.ifft(offline_pspec_a3)
# offline_AC_a4 = sp.fft.ifft(offline_pspec_a4)

# offline_AC = offline_AC/offline_AC[0]
# offline_AC_a1 = offline_AC_a1/offline_AC_a1[0]
# offline_AC_a2 = offline_AC_a2/offline_AC_a2[0]
# offline_AC_a3 = offline_AC_a3/offline_AC_a3[0]
# offline_AC_a4 = offline_AC_a4/offline_AC_a4[0]

# fig, ax = plt.subplots()
# ax.plot(Lags[l1:l2], AC_SAM_forc_a2[l1:l2],color= CB_lightblue,label='True AC')
# ax.plot(Lags[l1:l2], offline_AC_a2[l1:l2],linestyle='dashed', color= CB_lightblue,label='Offline model (1/tau = 0.105 $days^{-1}$)')
# plt.xlabel('Lag (days)')
# ax.legend(frameon=False)
# plt.ylabel('Autocorrelation')


#SAM_PC1_pspec = sp.signal.welch(SAM_PC1)
# SAM_PC1_a1_pspec = sp.signal.welch(SAM_PC1_forc_a1)
# SAM_PC1_a2_pspec = sp.signal.welch(SAM_PC1_forc_a2)
# SAM_PC1_a3_pspec = sp.signal.welch(SAM_PC1_forc_a3)
# SAM_PC1_a4_pspec = sp.signal.welch(SAM_PC1_forc_a4)



# tau_ctrl = (-np.log(AC_SAM[25] / AC_SAM[15])/10)**(-1)
# eddy_forc_pspec = SAM_PC1_pspec[1]*((2*np.pi*SAM_PC1_pspec[0])**2 + tau_ctrl**(-2))
# eddy_forc_cent = np.dot(eddy_forc_pspec[0:],SAM_PC1_pspec[0][0:])/np.sum(eddy_forc_pspec[0:])



f1 = 0
f2= -15000



#####  Autocorrelation accounting ####



###Uncertainty quantification###
samp = 200
seglen = round(tlen_nd/samp)
dtau_a1_ss = np.zeros(samp,dtype='d')
dtau_a2_ss = np.zeros(samp,dtype='d')
dtau_a3_ss = np.zeros(samp,dtype='d')
dtau_a4_ss = np.zeros(samp,dtype='d')
dtau_a0_r_a1_ss = np.zeros(samp,dtype='d')
dtau_a0_r_a2_ss = np.zeros(samp,dtype='d')
dtau_a0_r_a3_ss = np.zeros(samp,dtype='d')
dtau_a0_r_a4_ss = np.zeros(samp,dtype='d')
dtau_a0_b_a1_ss = np.zeros(samp,dtype='d')
dtau_a0_b_a2_ss = np.zeros(samp,dtype='d')
dtau_a0_b_a3_ss = np.zeros(samp,dtype='d')
dtau_a0_b_a4_ss = np.zeros(samp,dtype='d')
dtau_a0_m_a1_ss = np.zeros(samp,dtype='d')
dtau_a0_m_a2_ss = np.zeros(samp,dtype='d')
dtau_a0_m_a3_ss = np.zeros(samp,dtype='d')
dtau_a0_m_a4_ss = np.zeros(samp,dtype='d')
dtau_a0_brm_a1_ss = np.zeros(samp,dtype='d')
dtau_a0_brm_a2_ss = np.zeros(samp,dtype='d')
dtau_a0_brm_a3_ss = np.zeros(samp,dtype='d')
dtau_a0_brm_a4_ss = np.zeros(samp,dtype='d')

dtau_a0_b_low_a1_ss = np.zeros(samp,dtype='d')
dtau_a0_b_low_a2_ss = np.zeros(samp,dtype='d')
dtau_a0_b_low_a3_ss = np.zeros(samp,dtype='d')
dtau_a0_b_low_a4_ss = np.zeros(samp,dtype='d')
dtau_a0_b_high_a1_ss = np.zeros(samp,dtype='d')
dtau_a0_b_high_a2_ss = np.zeros(samp,dtype='d')
dtau_a0_b_high_a3_ss = np.zeros(samp,dtype='d')
dtau_a0_b_high_a4_ss = np.zeros(samp,dtype='d')

for s in range(samp):       
          i1 = round(tlen_nd/samp)*s
          i2 = i1 + round(tlen_nd/samp)
          
          SAM_PC1_s = SAM_PC1[i1:i2]
          SAM_PC1_a1_s = SAM_PC1_forc_a1[i1:i2]
          SAM_PC1_a2_s = SAM_PC1_forc_a2[i1:i2]
          SAM_PC1_a3_s = SAM_PC1_forc_a3[i1:i2]
          SAM_PC1_a4_s = SAM_PC1_forc_a4[i1:i2]
          

          
          
          vorflux_dot_eof_norm_s = vorflux_dot_eof_norm[i1:i2]
          vorflux_dot_eof_norm_a1_s = vorflux_dot_eof_norm_forc_a1[i1:i2]
          vorflux_dot_eof_norm_a2_s = vorflux_dot_eof_norm_forc_a2[i1:i2]
          vorflux_dot_eof_norm_a3_s = vorflux_dot_eof_norm_forc_a3[i1:i2]
          vorflux_dot_eof_norm_a4_s = vorflux_dot_eof_norm_forc_a4[i1:i2]
          

          
          damp_dot_eof_norm_s = damp_dot_eof_norm[i1:i2]
          damp_dot_eof_norm_a1_s = damp_dot_eof_norm_forc_a1[i1:i2]
          damp_dot_eof_norm_a2_s = damp_dot_eof_norm_forc_a2[i1:i2]
          damp_dot_eof_norm_a3_s = damp_dot_eof_norm_forc_a3[i1:i2]
          damp_dot_eof_norm_a4_s = damp_dot_eof_norm_forc_a4[i1:i2]
          
          fv_dot_eof_norm_s = fv_dot_eof_norm[i1:i2]
          fv_dot_eof_norm_a1_s = fv_dot_eof_norm_forc_a1[i1:i2]
          fv_dot_eof_norm_a2_s = fv_dot_eof_norm_forc_a2[i1:i2]
          fv_dot_eof_norm_a3_s = fv_dot_eof_norm_forc_a3[i1:i2]
          fv_dot_eof_norm_a4_s = fv_dot_eof_norm_forc_a4[i1:i2]
          
          eddy_feed_s = CCF_full(SAM_PC1_s,vorflux_dot_eof_norm_s)/CCF_full(SAM_PC1_s,SAM_PC1_s)
          eddy_feed_a1_s = CCF_full(SAM_PC1_a1_s,vorflux_dot_eof_norm_a1_s)/CCF_full(SAM_PC1_a1_s,SAM_PC1_a1_s)
          eddy_feed_a2_s = CCF_full(SAM_PC1_a2_s,vorflux_dot_eof_norm_a2_s)/CCF_full(SAM_PC1_a2_s,SAM_PC1_a2_s)
          eddy_feed_a3_s = CCF_full(SAM_PC1_a3_s,vorflux_dot_eof_norm_a3_s)/CCF_full(SAM_PC1_a3_s,SAM_PC1_a3_s)
          eddy_feed_a4_s = CCF_full(SAM_PC1_a4_s,vorflux_dot_eof_norm_a4_s)/CCF_full(SAM_PC1_a4_s,SAM_PC1_a4_s)
          

          fv_feed_s = CCF_full(SAM_PC1_s,fv_dot_eof_norm_s)/CCF_full(SAM_PC1_s,SAM_PC1_s)
          fv_feed_a1_s = CCF_full(SAM_PC1_a1_s,fv_dot_eof_norm_a1_s)/CCF_full(SAM_PC1_a1_s,SAM_PC1_a1_s)
          fv_feed_a2_s = CCF_full(SAM_PC1_a2_s,fv_dot_eof_norm_a2_s)/CCF_full(SAM_PC1_a2_s,SAM_PC1_a2_s)
          fv_feed_a3_s = CCF_full(SAM_PC1_a3_s,fv_dot_eof_norm_a3_s)/CCF_full(SAM_PC1_a3_s,SAM_PC1_a3_s)
          fv_feed_a4_s = CCF_full(SAM_PC1_a4_s,fv_dot_eof_norm_a4_s)/CCF_full(SAM_PC1_a4_s,SAM_PC1_a4_s)

          damp_feed_s = -CCF_full(SAM_PC1_s,damp_dot_eof_norm_s)/CCF_full(SAM_PC1_s,SAM_PC1_s)
          damp_feed_a1_s = -CCF_full(SAM_PC1_a1_s,damp_dot_eof_norm_a1_s)/CCF_full(SAM_PC1_a1_s,SAM_PC1_a1_s)
          damp_feed_a2_s = -CCF_full(SAM_PC1_a2_s,damp_dot_eof_norm_a2_s)/CCF_full(SAM_PC1_a2_s,SAM_PC1_a2_s)
          damp_feed_a3_s = -CCF_full(SAM_PC1_a3_s,damp_dot_eof_norm_a3_s)/CCF_full(SAM_PC1_a3_s,SAM_PC1_a3_s)
          damp_feed_a4_s = -CCF_full(SAM_PC1_a4_s,damp_dot_eof_norm_a4_s)/CCF_full(SAM_PC1_a4_s,SAM_PC1_a4_s)
          
#           #Drop subscript s where possible
          AC_SAM_s = ACF_full(SAM_PC1_s) 
          AC_SAM_a1_s = ACF_full(SAM_PC1_a1_s)
          AC_SAM_a2_s = ACF_full(SAM_PC1_a2_s) 
          AC_SAM_a3_s = ACF_full(SAM_PC1_a3_s) 
          AC_SAM_a4_s = ACF_full(SAM_PC1_a4_s)
          
          AC_SAM_ZG_pre_s = np.diff(np.log(ACF_full(SAM_PC1_s) ))
          AC_SAM_ZG_s = np.zeros(round(tlen_nd/samp),dtype='d')
          AC_SAM_ZG_s[1:] = AC_SAM_ZG_pre_s
          
          

          

          tau_a0_s = nearest_coord(1/np.e,AC_SAM_s[0:70])
          tau_a1_s = nearest_coord(1/np.e,AC_SAM_a1_s[0:70])
          tau_a2_s = nearest_coord(1/np.e,AC_SAM_a2_s[0:70])
          tau_a3_s = nearest_coord(1/np.e,AC_SAM_a3_s[0:70])
          tau_a4_s = nearest_coord(1/np.e,AC_SAM_a4_s[0:70])
              ##
          dtau_a1_s = tau_a1_s - tau_a0_s
          dtau_a2_s = tau_a2_s - tau_a0_s
          dtau_a3_s = tau_a3_s - tau_a0_s
          dtau_a4_s = tau_a4_s - tau_a0_s
              ##
          dtau_a1_ss[s] = dtau_a1_s
          dtau_a2_ss[s] = dtau_a2_s
          dtau_a3_ss[s] = dtau_a3_s
          dtau_a4_ss[s] = dtau_a4_s
          
          

          SAM_ZG_a0_b_a1_s = AC_SAM_ZG_s + (eddy_feed_a1_s - eddy_feed_s)
          SAM_ZG_a0_b_a2_s = AC_SAM_ZG_s + (eddy_feed_a2_s - eddy_feed_s) 
          SAM_ZG_a0_b_a3_s = AC_SAM_ZG_s + (eddy_feed_a3_s - eddy_feed_s) 
          SAM_ZG_a0_b_a4_s = AC_SAM_ZG_s + (eddy_feed_a4_s - eddy_feed_s) 


          AC_SAM_a0_b_a1_s = np.exp(np.cumsum(SAM_ZG_a0_b_a1_s))
          AC_SAM_a0_b_a2_s = np.exp(np.cumsum(SAM_ZG_a0_b_a2_s))
          AC_SAM_a0_b_a3_s = np.exp(np.cumsum(SAM_ZG_a0_b_a3_s))
          AC_SAM_a0_b_a4_s = np.exp(np.cumsum(SAM_ZG_a0_b_a4_s))
              ##
          dtau_a0_b_a1_s = nearest_coord(1/np.e,AC_SAM_a0_b_a1_s[0:70])-tau_a0_s
          dtau_a0_b_a2_s = nearest_coord(1/np.e,AC_SAM_a0_b_a2_s[0:70])-tau_a0_s
          dtau_a0_b_a3_s = nearest_coord(1/np.e,AC_SAM_a0_b_a3_s[0:70])-tau_a0_s
          dtau_a0_b_a4_s = nearest_coord(1/np.e,AC_SAM_a0_b_a4_s[0:70])-tau_a0_s
              ##
          dtau_a0_b_a1_ss[s] = dtau_a0_b_a1_s
          dtau_a0_b_a2_ss[s] = dtau_a0_b_a2_s
          dtau_a0_b_a3_ss[s] = dtau_a0_b_a3_s
          dtau_a0_b_a4_ss[s] = dtau_a0_b_a4_s



           
          SAM_ZG_a0_r_a1_s = AC_SAM_ZG_s +  (damp_feed_a1_s - damp_feed_s) #+ (fv_feed_a1 - fv_feed) 
          SAM_ZG_a0_r_a2_s = AC_SAM_ZG_s + (damp_feed_a2_s - damp_feed_s) #+ (fv_feed_a2 - fv_feed) 
          SAM_ZG_a0_r_a3_s = AC_SAM_ZG_s + (damp_feed_a3_s - damp_feed_s) #+ (fv_feed_a3 - fv_feed) 
          SAM_ZG_a0_r_a4_s = AC_SAM_ZG_s + (damp_feed_a4_s - damp_feed_s) #+ (fv_feed_a4 - fv_feed) 


          AC_SAM_a0_r_a1_s = np.exp(np.cumsum(SAM_ZG_a0_r_a1_s))
          AC_SAM_a0_r_a2_s = np.exp(np.cumsum(SAM_ZG_a0_r_a2_s))
          AC_SAM_a0_r_a3_s = np.exp(np.cumsum(SAM_ZG_a0_r_a3_s))
          AC_SAM_a0_r_a4_s = np.exp(np.cumsum(SAM_ZG_a0_r_a4_s))
          ##
          dtau_a0_r_a1_s = nearest_coord(1/np.e,AC_SAM_a0_r_a1_s[0:70])-tau_a0_s
          dtau_a0_r_a2_s = nearest_coord(1/np.e,AC_SAM_a0_r_a2_s[0:70])-tau_a0_s
          dtau_a0_r_a3_s = nearest_coord(1/np.e,AC_SAM_a0_r_a3_s[0:70])-tau_a0_s
          dtau_a0_r_a4_s = nearest_coord(1/np.e,AC_SAM_a0_r_a4_s[0:70])-tau_a0_s
          ##
          dtau_a0_r_a1_ss[s] = dtau_a0_r_a1_s
          dtau_a0_r_a2_ss[s] = dtau_a0_r_a2_s
          dtau_a0_r_a3_ss[s] = dtau_a0_r_a3_s
          dtau_a0_r_a4_ss[s] = dtau_a0_r_a4_s
          
          SAM_ZG_a0_m_a1_s = AC_SAM_ZG_s +  (fv_feed_a1_s - fv_feed_s) 
          SAM_ZG_a0_m_a2_s = AC_SAM_ZG_s +  (fv_feed_a2_s - fv_feed_s) 
          SAM_ZG_a0_m_a3_s = AC_SAM_ZG_s +  (fv_feed_a3_s - fv_feed_s) 
          SAM_ZG_a0_m_a4_s = AC_SAM_ZG_s +  (fv_feed_a4_s - fv_feed_s) 


          AC_SAM_a0_m_a1_s = np.exp(np.cumsum(SAM_ZG_a0_m_a1_s))
          AC_SAM_a0_m_a2_s = np.exp(np.cumsum(SAM_ZG_a0_m_a2_s))
          AC_SAM_a0_m_a3_s = np.exp(np.cumsum(SAM_ZG_a0_m_a3_s))
          AC_SAM_a0_m_a4_s = np.exp(np.cumsum(SAM_ZG_a0_m_a4_s))
          ##
          dtau_a0_m_a1_s = nearest_coord(1/np.e,AC_SAM_a0_m_a1_s[0:70])-tau_a0_s
          dtau_a0_m_a2_s = nearest_coord(1/np.e,AC_SAM_a0_m_a2_s[0:70])-tau_a0_s
          dtau_a0_m_a3_s = nearest_coord(1/np.e,AC_SAM_a0_m_a3_s[0:70])-tau_a0_s
          dtau_a0_m_a4_s = nearest_coord(1/np.e,AC_SAM_a0_m_a4_s[0:70])-tau_a0_s
          ##
          dtau_a0_m_a1_ss[s] = dtau_a0_m_a1_s
          dtau_a0_m_a2_ss[s] = dtau_a0_m_a2_s
          dtau_a0_m_a3_ss[s] = dtau_a0_m_a3_s
          dtau_a0_m_a4_ss[s] = dtau_a0_m_a4_s
          
          SAM_ZG_a0_brm_a1_s = AC_SAM_ZG_s + (eddy_feed_a1_s - eddy_feed_s)+(damp_feed_a1_s - damp_feed_s)+(fv_feed_a1_s - fv_feed_s) 
          SAM_ZG_a0_brm_a2_s = AC_SAM_ZG_s + (eddy_feed_a2_s - eddy_feed_s)+(damp_feed_a2_s - damp_feed_s)+(fv_feed_a2_s - fv_feed_s) 
          SAM_ZG_a0_brm_a3_s = AC_SAM_ZG_s + (eddy_feed_a3_s - eddy_feed_s)+(damp_feed_a3_s - damp_feed_s)+(fv_feed_a3_s - fv_feed_s)  
          SAM_ZG_a0_brm_a4_s = AC_SAM_ZG_s + (eddy_feed_a4_s - eddy_feed_s)+(damp_feed_a4_s - damp_feed_s)+(fv_feed_a4_s - fv_feed_s) 
          
          AC_SAM_a0_brm_a1_s = np.exp(np.cumsum(SAM_ZG_a0_brm_a1_s))
          AC_SAM_a0_brm_a2_s = np.exp(np.cumsum(SAM_ZG_a0_brm_a2_s))
          AC_SAM_a0_brm_a3_s = np.exp(np.cumsum(SAM_ZG_a0_brm_a3_s))
          AC_SAM_a0_brm_a4_s = np.exp(np.cumsum(SAM_ZG_a0_brm_a4_s))
            ##
          dtau_a0_brm_a1_s = nearest_coord(1/np.e,AC_SAM_a0_brm_a1_s[0:70])-tau_a0_s
          dtau_a0_brm_a2_s = nearest_coord(1/np.e,AC_SAM_a0_brm_a2_s[0:70])-tau_a0_s
          dtau_a0_brm_a3_s = nearest_coord(1/np.e,AC_SAM_a0_brm_a3_s[0:70])-tau_a0_s
          dtau_a0_brm_a4_s = nearest_coord(1/np.e,AC_SAM_a0_brm_a4_s[0:70])-tau_a0_s
            ##
          dtau_a0_brm_a1_ss[s] = dtau_a0_brm_a1_s
          dtau_a0_brm_a2_ss[s] = dtau_a0_brm_a2_s
          dtau_a0_brm_a3_ss[s] = dtau_a0_brm_a3_s
          dtau_a0_brm_a4_ss[s] = dtau_a0_brm_a4_s
          
std_dtau_a1_ss = stdev(dtau_a1_ss)
std_dtau_a2_ss = stdev(dtau_a2_ss)
std_dtau_a3_ss = stdev(dtau_a3_ss)
std_dtau_a4_ss = stdev(dtau_a4_ss)

std_dtau_a0_b_a1_ss = stdev(dtau_a0_b_a1_ss)
std_dtau_a0_b_a2_ss = stdev(dtau_a0_b_a2_ss)
std_dtau_a0_b_a3_ss = stdev(dtau_a0_b_a3_ss)
std_dtau_a0_b_a4_ss = stdev(dtau_a0_b_a4_ss)


std_dtau_a0_r_a1_ss = stdev(dtau_a0_r_a1_ss)
std_dtau_a0_r_a2_ss = stdev(dtau_a0_r_a2_ss)
std_dtau_a0_r_a3_ss = stdev(dtau_a0_r_a3_ss)
std_dtau_a0_r_a4_ss = stdev(dtau_a0_r_a4_ss)

std_dtau_a0_m_a1_ss = stdev(dtau_a0_m_a1_ss)
std_dtau_a0_m_a2_ss = stdev(dtau_a0_m_a2_ss)
std_dtau_a0_m_a3_ss = stdev(dtau_a0_m_a3_ss)
std_dtau_a0_m_a4_ss = stdev(dtau_a0_m_a4_ss)

std_dtau_a0_brm_a1_ss = stdev(dtau_a0_brm_a1_ss)
std_dtau_a0_brm_a2_ss = stdev(dtau_a0_brm_a2_ss)
std_dtau_a0_brm_a3_ss = stdev(dtau_a0_brm_a3_ss)
std_dtau_a0_brm_a4_ss = stdev(dtau_a0_brm_a4_ss)

err_dtau_a1_ss = t_err(std_dtau_a1_ss,samp,0.95)
err_dtau_a2_ss = t_err(std_dtau_a2_ss,samp,0.95)
err_dtau_a3_ss = t_err(std_dtau_a3_ss,samp,0.95)
err_dtau_a4_ss = t_err(std_dtau_a4_ss,samp,0.95)
err_dtau_ss = [err_dtau_a1_ss,err_dtau_a2_ss,err_dtau_a3_ss,err_dtau_a4_ss]

err_dtau_a0_r_a1_ss = t_err(std_dtau_a0_r_a1_ss,samp,0.95)
err_dtau_a0_r_a2_ss = t_err(std_dtau_a0_r_a2_ss,samp,0.95)
err_dtau_a0_r_a3_ss = t_err(std_dtau_a0_r_a3_ss,samp,0.95)
err_dtau_a0_r_a4_ss = t_err(std_dtau_a0_r_a4_ss,samp,0.95)
err_dtau_r_ss = [err_dtau_a0_r_a1_ss,err_dtau_a0_r_a2_ss,err_dtau_a0_r_a3_ss,err_dtau_a0_r_a4_ss]


err_dtau_a0_b_a1_ss = t_err(std_dtau_a0_b_a1_ss,samp,0.95)
err_dtau_a0_b_a2_ss = t_err(std_dtau_a0_b_a2_ss,samp,0.95)
err_dtau_a0_b_a3_ss = t_err(std_dtau_a0_b_a3_ss,samp,0.95)
err_dtau_a0_b_a4_ss = t_err(std_dtau_a0_b_a4_ss,samp,0.95)
err_dtau_b_ss = [err_dtau_a0_b_a1_ss,err_dtau_a0_b_a2_ss,err_dtau_a0_b_a3_ss,err_dtau_a0_b_a4_ss]

err_dtau_a0_m_a1_ss = t_err(std_dtau_a0_m_a1_ss,samp,0.95)
err_dtau_a0_m_a2_ss = t_err(std_dtau_a0_m_a2_ss,samp,0.95)
err_dtau_a0_m_a3_ss = t_err(std_dtau_a0_m_a3_ss,samp,0.95)
err_dtau_a0_m_a4_ss = t_err(std_dtau_a0_m_a4_ss,samp,0.95)
err_dtau_m_ss = [err_dtau_a0_m_a1_ss,err_dtau_a0_m_a2_ss,err_dtau_a0_m_a3_ss,err_dtau_a0_m_a4_ss]

err_dtau_a0_brm_a1_ss = t_err(std_dtau_a0_brm_a1_ss,samp,0.95)
err_dtau_a0_brm_a2_ss = t_err(std_dtau_a0_brm_a2_ss,samp,0.95)
err_dtau_a0_brm_a3_ss = t_err(std_dtau_a0_brm_a3_ss,samp,0.95)
err_dtau_a0_brm_a4_ss = t_err(std_dtau_a0_brm_a4_ss,samp,0.95)
err_dtau_brm_ss = [err_dtau_a0_brm_a1_ss,err_dtau_a0_brm_a2_ss,err_dtau_a0_brm_a3_ss,err_dtau_a0_brm_a4_ss]



          
# ##Calculate with full timeseries##

eddy_feed = CCF_full(SAM_PC1,vorflux_dot_eof_norm)/CCF_full(SAM_PC1,SAM_PC1)
eddy_feed_a1 = CCF_full(SAM_PC1_forc_a1,vorflux_dot_eof_norm_forc_a1)/CCF_full(SAM_PC1_forc_a1,SAM_PC1_forc_a1)
eddy_feed_a2 = CCF_full(SAM_PC1_forc_a2,vorflux_dot_eof_norm_forc_a2)/CCF_full(SAM_PC1_forc_a2,SAM_PC1_forc_a2)
eddy_feed_a3 = CCF_full(SAM_PC1_forc_a3,vorflux_dot_eof_norm_forc_a3)/CCF_full(SAM_PC1_forc_a3,SAM_PC1_forc_a3)
eddy_feed_a4 = CCF_full(SAM_PC1_forc_a4,vorflux_dot_eof_norm_forc_a4)/CCF_full(SAM_PC1_forc_a4,SAM_PC1_forc_a4)

fv_feed = CCF_full(SAM_PC1,fv_dot_eof_norm)/CCF_full(SAM_PC1,SAM_PC1)
fv_feed_a1 = CCF_full(SAM_PC1_forc_a1,fv_dot_eof_norm_forc_a1)/CCF_full(SAM_PC1_forc_a1,SAM_PC1_forc_a1)
fv_feed_a2 = CCF_full(SAM_PC1_forc_a2,fv_dot_eof_norm_forc_a2)/CCF_full(SAM_PC1_forc_a2,SAM_PC1_forc_a2)
fv_feed_a3 = CCF_full(SAM_PC1_forc_a3,fv_dot_eof_norm_forc_a3)/CCF_full(SAM_PC1_forc_a3,SAM_PC1_forc_a3)
fv_feed_a4 = CCF_full(SAM_PC1_forc_a4,fv_dot_eof_norm_forc_a4)/CCF_full(SAM_PC1_forc_a4,SAM_PC1_forc_a4)

damp_feed = -CCF_full(SAM_PC1,damp_dot_eof_norm)/CCF_full(SAM_PC1,SAM_PC1)
damp_feed_a1 = -CCF_full(SAM_PC1_forc_a1,damp_dot_eof_norm_forc_a1)/CCF_full(SAM_PC1_forc_a1,SAM_PC1_forc_a1)
damp_feed_a2 = -CCF_full(SAM_PC1_forc_a2,damp_dot_eof_norm_forc_a2)/CCF_full(SAM_PC1_forc_a2,SAM_PC1_forc_a2)
damp_feed_a3 = -CCF_full(SAM_PC1_forc_a3,damp_dot_eof_norm_forc_a3)/CCF_full(SAM_PC1_forc_a3,SAM_PC1_forc_a3)
damp_feed_a4 = -CCF_full(SAM_PC1_forc_a4,damp_dot_eof_norm_forc_a4)/CCF_full(SAM_PC1_forc_a4,SAM_PC1_forc_a4)

AC_SAM_ZG_pre = np.diff(np.log(ACF_full(SAM_PC1) ))
AC_SAM_ZG = np.zeros(tlen_nd,dtype='d')
AC_SAM_ZG[1:] = AC_SAM_ZG_pre


tau_a0 = nearest_coord(1/np.e,AC_SAM[0:70])
tau_a1 = nearest_coord(1/np.e,AC_SAM_forc_a1[0:70])
tau_a2 = nearest_coord(1/np.e,AC_SAM_forc_a2[0:70])
tau_a3 = nearest_coord(1/np.e,AC_SAM_forc_a3[0:70])
tau_a4 = nearest_coord(1/np.e,AC_SAM_forc_a4[0:70])

dtau_a1 = tau_a1 - tau_a0
dtau_a2 = tau_a2 - tau_a0
dtau_a3 = tau_a3 - tau_a0
dtau_a4 = tau_a4 - tau_a0

AC_SAM_ZG_forc_a1 = np.diff(np.log(ACF_full(SAM_PC1_forc_a1) ))
AC_SAM_ZG_forc_a2 = np.diff(np.log(ACF_full(SAM_PC1_forc_a2) ))
AC_SAM_ZG_forc_a3 = np.diff(np.log(ACF_full(SAM_PC1_forc_a3) ))
AC_SAM_ZG_forc_a4 = np.diff(np.log(ACF_full(SAM_PC1_forc_a4) ))

SAM_ZG_a0_b_a1 = AC_SAM_ZG + (eddy_feed_a1 - eddy_feed)
SAM_ZG_a0_b_a2 = AC_SAM_ZG + (eddy_feed_a2 - eddy_feed) 
SAM_ZG_a0_b_a3 = AC_SAM_ZG + (eddy_feed_a3 - eddy_feed) 
SAM_ZG_a0_b_a4 = AC_SAM_ZG + (eddy_feed_a4 - eddy_feed)


AC_SAM_a0_b_a1 = np.exp(np.cumsum(SAM_ZG_a0_b_a1))
AC_SAM_a0_b_a2 = np.exp(np.cumsum(SAM_ZG_a0_b_a2))
AC_SAM_a0_b_a3 = np.exp(np.cumsum(SAM_ZG_a0_b_a3))
AC_SAM_a0_b_a4 = np.exp(np.cumsum(SAM_ZG_a0_b_a4))

dtau_a0_b_a1 = nearest_coord(1/np.e,AC_SAM_a0_b_a1[0:70])-tau_a0
dtau_a0_b_a2 = nearest_coord(1/np.e,AC_SAM_a0_b_a2[0:70])-tau_a0
dtau_a0_b_a3 = nearest_coord(1/np.e,AC_SAM_a0_b_a3[0:70])-tau_a0
dtau_a0_b_a4 = nearest_coord(1/np.e,AC_SAM_a0_b_a4[0:70])-tau_a0


SAM_ZG_a0_r_a1 = AC_SAM_ZG +  (damp_feed_a1 - damp_feed) 
SAM_ZG_a0_r_a2 = AC_SAM_ZG + (damp_feed_a2 - damp_feed) 
SAM_ZG_a0_r_a3 = AC_SAM_ZG + (damp_feed_a3 - damp_feed) 
SAM_ZG_a0_r_a4 = AC_SAM_ZG + (damp_feed_a4 - damp_feed)


AC_SAM_a0_r_a1 = np.exp(np.cumsum(SAM_ZG_a0_r_a1))
AC_SAM_a0_r_a2 = np.exp(np.cumsum(SAM_ZG_a0_r_a2))
AC_SAM_a0_r_a3 = np.exp(np.cumsum(SAM_ZG_a0_r_a3))
AC_SAM_a0_r_a4 = np.exp(np.cumsum(SAM_ZG_a0_r_a4))

dtau_a0_r_a1 = nearest_coord(1/np.e,AC_SAM_a0_r_a1[0:70])-tau_a0
dtau_a0_r_a2 = nearest_coord(1/np.e,AC_SAM_a0_r_a2[0:70])-tau_a0
dtau_a0_r_a3 = nearest_coord(1/np.e,AC_SAM_a0_r_a3[0:70])-tau_a0
dtau_a0_r_a4 = nearest_coord(1/np.e,AC_SAM_a0_r_a4[0:70])-tau_a0

SAM_ZG_a0_m_a1 = AC_SAM_ZG +  (fv_feed_a1 - fv_feed) 
SAM_ZG_a0_m_a2 = AC_SAM_ZG +  (fv_feed_a2 - fv_feed) 
SAM_ZG_a0_m_a3 = AC_SAM_ZG +  (fv_feed_a3 - fv_feed) 
SAM_ZG_a0_m_a4 = AC_SAM_ZG +  (fv_feed_a4 - fv_feed) 


AC_SAM_a0_m_a1 = np.exp(np.cumsum(SAM_ZG_a0_m_a1))
AC_SAM_a0_m_a2 = np.exp(np.cumsum(SAM_ZG_a0_m_a2))
AC_SAM_a0_m_a3 = np.exp(np.cumsum(SAM_ZG_a0_m_a3))
AC_SAM_a0_m_a4 = np.exp(np.cumsum(SAM_ZG_a0_m_a4))

dtau_a0_m_a1 = nearest_coord(1/np.e,AC_SAM_a0_m_a1[0:70])-tau_a0
dtau_a0_m_a2 = nearest_coord(1/np.e,AC_SAM_a0_m_a2[0:70])-tau_a0
dtau_a0_m_a3 = nearest_coord(1/np.e,AC_SAM_a0_m_a3[0:70])-tau_a0
dtau_a0_m_a4 = nearest_coord(1/np.e,AC_SAM_a0_m_a4[0:70])-tau_a0

SAM_ZG_a0_brm_a1 = AC_SAM_ZG + (eddy_feed_a1 - eddy_feed)+(damp_feed_a1 - damp_feed)+(fv_feed_a1 - fv_feed) 
SAM_ZG_a0_brm_a2 = AC_SAM_ZG + (eddy_feed_a2 - eddy_feed)+(damp_feed_a2 - damp_feed)+(fv_feed_a2 - fv_feed) 
SAM_ZG_a0_brm_a3 = AC_SAM_ZG + (eddy_feed_a3 - eddy_feed)+(damp_feed_a3 - damp_feed)+(fv_feed_a3 - fv_feed)  
SAM_ZG_a0_brm_a4 = AC_SAM_ZG + (eddy_feed_a4 - eddy_feed)+(damp_feed_a4 - damp_feed)+(fv_feed_a4 - fv_feed) 
          
AC_SAM_a0_brm_a1 = np.exp(np.cumsum(SAM_ZG_a0_brm_a1))
AC_SAM_a0_brm_a2 = np.exp(np.cumsum(SAM_ZG_a0_brm_a2))
AC_SAM_a0_brm_a3 = np.exp(np.cumsum(SAM_ZG_a0_brm_a3))
AC_SAM_a0_brm_a4 = np.exp(np.cumsum(SAM_ZG_a0_brm_a4))
          ##
dtau_a0_brm_a1 = nearest_coord(1/np.e,AC_SAM_a0_brm_a1[0:70])-tau_a0
dtau_a0_brm_a2 = nearest_coord(1/np.e,AC_SAM_a0_brm_a2[0:70])-tau_a0
dtau_a0_brm_a3 = nearest_coord(1/np.e,AC_SAM_a0_brm_a3[0:70])-tau_a0
dtau_a0_brm_a4 = nearest_coord(1/np.e,AC_SAM_a0_brm_a4[0:70])-tau_a0

# # # # ### Do grouped bar chart ###

avals = ['a=1','a=3','a=-1','a=-3']
eddies = np.array([dtau_a0_b_a1, dtau_a0_b_a3, dtau_a0_b_a2, dtau_a0_b_a4])
friction =  np.array([dtau_a0_r_a1, dtau_a0_r_a3, dtau_a0_r_a2, dtau_a0_r_a4])                
MMC =  np.array([dtau_a0_m_a1, dtau_a0_m_a3, dtau_a0_m_a2, dtau_a0_m_a4])
BRM = np.array([dtau_a0_brm_a1, dtau_a0_brm_a3, dtau_a0_brm_a2, dtau_a0_brm_a4])
total = np.array([dtau_a1, dtau_a3, dtau_a2, dtau_a4])

#x = np.arange(4)  # the label locations
width = 0.2  # the width of the bars
wid = 3.3
cap = 7
mult = 7.1
eddy_locs = 0.2*np.array([0,mult,2*mult,3*mult])
fric_locs = eddy_locs + width
MMC_locs = fric_locs + width
BRM_locs = MMC_locs + width
total_locs = BRM_locs + width
a_locs = fric_locs + width/2


red_rgba = colors.to_rgba('red')
blue_rgba = colors.to_rgba('blue')
orange_rgba = colors.to_rgba('orange')
purple_rgba = colors.to_rgba('purple')

red_rgba = (red_rgba[0],red_rgba[1],red_rgba[2],0.8)
blue_rgba = (blue_rgba[0],blue_rgba[1],blue_rgba[2],0.8)
orange_rgba = (orange_rgba[0],orange_rgba[1],orange_rgba[2],0.8)
purple_rgba = (purple_rgba[0],purple_rgba[1],purple_rgba[2],0.8)
gray_rgba = (143./255,114./255,114./255,0.4)


# ### Paper Figure 1:  CRE, T residual, autocorrelation, autocorrelation accounting###
# #f’{math.pi:.3f}’Syntax for displaying decimal places

l1 = 0
l2 = 36
inv_e = np.array([1/np.e] * (l2-l1))

#NB if there's a lot of colors, I like taking colors from the following colorset, which is colorblind-friendly
cs = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
              (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
              (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
              (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
              (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
for i in range(len(cs)):  
    r, g, b = cs[i]  
    cs[i] = (r / 255., g / 255., b / 255.)  


wid = 1.5

# col_a1 = cs[0]
# col_a2 = cs[7]
# col_a3 = cs[5]
# col_a4 = cs[12]

col_a1 = (31/255., 119/255., 180/255., 0.6)
col_a2 = (255/255., 152/255., 150/255., 0.7)
col_a3 = (31/255., 119/255., 180/255., 1.0)
col_a4 = (214/255., 39/255., 40/255., 0.7)




heat_rate_work = 86400*heat_rate_work

fig = plt.figure( figsize = (23, 19) ) #GRL proportions?
fig.subplots_adjust(left =0.13, bottom = 0.1, top = 0.95, hspace = 0.1, wspace = 0.2, right = 0.8)
bigsize=35
smallsize=30

red_muted = (153/255,12/255,19/255,0.8)
blue_muted = (13/255,51/255,148/255,0.8)
orange_muted = (232/255,136/255,19/255,0.85)

#CRE_ax = plt.subplot(2,2,1)
CRE_ax = fig.add_axes([0.08,0.55,0.34,0.34])
#bound =  max(abs(heat_rate_work.min()),abs(heat_rate_work.max()))
bound = 0.35
num_levs = 15
levs_CRE = np.zeros(num_levs,dtype='d')
for l in range(num_levs):
    levs_CRE[l] = round(-bound + 2*bound*l/(num_levs-1),2)

im_CRE = CRE_ax.contourf(lat,press,heat_rate_work,vmin=-bound,vmax=bound,cmap=newcmap,levels=levs_CRE)
plt.title("(a) ACRE", fontsize=bigsize,loc = 'left')
plt.xlabel('Latitude',fontsize=bigsize)
plt.ylabel('Pressure [hPa]', fontsize=bigsize)
plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
#plt.yticks([200, 300, 400, 500, 600, 700, 800, 900 ],fontsize = smallsize)
plt.yticks([200, 400, 600, 800, 1000 ],fontsize = smallsize)
CRE_ax.tick_params(axis='x',length=10, width = 3)
CRE_ax.tick_params(axis='y',length=10, width = 3)
plt.gca().invert_yaxis()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
cbar_CRE_ax = fig.add_axes([0.43, 0.55, 0.01, 0.34]) 
cb_CRE = fig.colorbar(im_CRE, cax=cbar_CRE_ax) 
cb_CRE.set_label( "[K day$ ^{-1}$]", fontsize = bigsize  )
cb_CRE.ax.tick_params(labelsize=smallsize) 

# #T_ax = plt.subplot(2,2,2)
T_ax = fig.add_axes([0.56,0.55,0.34,0.34])
temp_zon_av_res_a1 = temp_zon_av_reg_SAM_a1 - temp_zon_av_reg_SAM
#bound = max(abs(temp_zon_av_res_a1.min()),abs(temp_zon_av_res_a1.max()))
bound = 0.52
levs_T = np.zeros(num_levs,dtype='d')
for l in range(num_levs):
    levs_T[l] = round(-bound + 2*bound*l/(num_levs-1),2)

im_T = T_ax.contourf(lat,press,temp_zon_av_res_a1, vmin=-bound,vmax=bound,cmap=newcmap, levels=levs_T)
plt.title("(b) Temperature anomalies, a=1", fontsize=bigsize,loc = 'left')
plt.xlabel('Latitude',fontsize=bigsize)
#plt.ylabel('Pressure (hPa)', fontsize=bigsize)
plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
plt.yticks([200, 400, 600, 800, 1000 ],fontsize = smallsize)
plt.yticks(color='w',fontsize=1)
T_ax.tick_params(axis='x',length=10, width = 3)
T_ax.tick_params(axis='y',length=10, width = 3)
plt.gca().invert_yaxis()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
cbar_T_ax = fig.add_axes([0.909, 0.55, 0.01, 0.34]) 
cb_T = fig.colorbar(im_T, cax=cbar_T_ax) 
cb_T.set_label( "[K]", fontsize = bigsize) 
cb_T.ax.tick_params(labelsize=smallsize)

#AC_ax = plt.subplot(2,2,3)
AC_SAM = ACF_full(SAM_PC1) 
AC_SAM_forc_a1 = ACF_full(SAM_PC1_forc_a1)
AC_SAM_forc_a2 = ACF_full(SAM_PC1_forc_a2) 
AC_SAM_forc_a3 = ACF_full(SAM_PC1_forc_a3) 
AC_SAM_forc_a4 = ACF_full(SAM_PC1_forc_a4) 
AC_ax = fig.add_axes([0.08,0.07,0.34,0.34])
AC_ax.plot(Lags[l1:l2],AC_SAM[l1:l2],color = 'black',label='control',linewidth=wid+1.5)
AC_ax.plot(Lags[l1:l2],AC_SAM_forc_a1[l1:l2], color = col_a1,label='a=1',linewidth=wid+1.5)
AC_ax.plot(Lags[l1:l2],AC_SAM_forc_a2[l1:l2], color = col_a2,label='a=-1',linewidth=wid+1.5)
AC_ax.plot(Lags[l1:l2],AC_SAM_forc_a3[l1:l2], color = col_a3,label='a=3',linewidth=wid+1.5)
AC_ax.plot(Lags[l1:l2],AC_SAM_forc_a4[l1:l2], color = col_a4,label='a=-3',linewidth=wid+1.5)
AC_ax.plot(Lags[l1:l2],inv_e[l1:l2],color = 'black',linestyle='dotted',linewidth=wid+1)
AC_ax.legend(frameon=False,fontsize=smallsize-5)
plt.xlabel('Lag [days]',fontsize=bigsize)
plt.ylabel('Autocorrelation',fontsize=bigsize)
plt.xlim([0,35])
plt.ylim([0,1])
# AC_ax.spines['top'].set_visible(False)
# AC_ax.spines['right'].set_visible(False)
plt.xticks([0, 5, 10, 15, 20, 25, 30, 35],fontsize = smallsize)
#plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],fontsize=smallsize)
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=smallsize)
AC_ax.tick_params(axis='x',length=10, width = 3)
AC_ax.tick_params(axis='y',length=10, width = 3)
plt.title("(c) Autocorrelation of Z", fontsize=bigsize,loc = 'left')

#AC_acc_ax = plt.subplot(2,2,4)
AC_acc_ax = fig.add_axes([0.56,0.07,0.35,0.35])
AC_acc_ax.bar(eddy_locs, eddies, width, color=blue_muted)
AC_acc_ax.bar(fric_locs, friction, width, color=red_muted)
AC_acc_ax.bar(MMC_locs, MMC, width, color=orange_muted)
plt.rcParams['hatch.linewidth'] = 1.5
AC_acc_ax.bar(BRM_locs,BRM,width, color=gray_rgba)
AC_acc_ax.bar(total_locs, total, width, color='black')

AC_acc_ax.legend(["Eddies","Friction","MMC", "E+F+M", "Total"], loc="lower right",fontsize=smallsize-12,frameon = False)
AC_acc_ax.axhline(y = 0.0, color = 'k', linestyle = 'solid', linewidth = 1.0)
plt.yticks([-12,-9,-6,-3,0,3,6,9,12],fontsize=smallsize)
plt.ylim([-14,14])
plt.ylabel('$\Delta \\tau_{AM}$ [days]',fontsize=bigsize)
# plt.errorbar(eddy_locs, eddies, yerr=np.array([7.82, 4.18, 5.65, 7.21]),fmt='o',color='black',linewidth=wid,capsize=cap)
# plt.errorbar(fric_locs, friction, yerr=np.array([3.86, 5.34, 6.28, 2.68]),fmt='o',color='black',linewidth=wid,capsize=cap)
# plt.errorbar(MMC_locs, MMC, yerr=np.array([2.81, 3.12, 3.79, 3.17]),fmt='o',color='black',linewidth=wid,capsize=cap)
# plt.errorbar(BRM_locs, BRM, yerr=np.array([6.68, 6.82, 5.26, 6.87]),fmt='o',color='black',linewidth=wid,capsize=cap)
# plt.errorbar(total_locs, total, yerr=np.array([5.48, 3.77, 4.35, 5.01]),fmt='o',color='black',linewidth=wid,capsize=cap)
plt.errorbar(eddy_locs, eddies, yerr=err_dtau_b_ss,fmt='o',color='black',linewidth=wid,capsize=cap)
plt.errorbar(fric_locs, friction, yerr=err_dtau_r_ss,fmt='o',color='black',linewidth=wid,capsize=cap)
plt.errorbar(MMC_locs, MMC, yerr=err_dtau_m_ss,fmt='o',color='black',linewidth=wid,capsize=cap)
plt.errorbar(BRM_locs, BRM, yerr=err_dtau_brm_ss,fmt='o',color='black',linewidth=wid,capsize=cap)
plt.errorbar(total_locs, total, yerr=err_dtau_ss,fmt='o',color='black',linewidth=wid,capsize=cap)
plt.xticks(a_locs, avals, fontsize=smallsize)
AC_acc_ax.tick_params(axis='x',length=10, width = 3)
AC_acc_ax.tick_params(axis='y',length=10, width = 3)
plt.title("(d) Change in persistence of Z", fontsize=bigsize,loc = 'left')

plt.savefig('ACRE_paper_fig1.png')
plt.savefig('ACRE_paper_fig1.pdf')

### Paper Figure S!! : temperature anomalies for forced simulations ###

# fig = plt.figure( figsize = (23, 19) ) #GRL proportions?
# fig.subplots_adjust(left =0.1, bottom = 0.1, top = 0.95, hspace = 0.1, wspace = 0.35, right = 1.0)
# bigsize=35
# smallsize=30

# red_muted = (153/255,12/255,19/255,0.8)
# blue_muted = (13/255,51/255,148/255,0.8)
# orange_muted = (232/255,136/255,19/255,0.85)

# levs_T =  np.zeros(num_levs,dtype='d')

# #CRE_ax = plt.subplot(2,2,1)
# a1_ax = fig.add_axes([0.12,0.55,0.35,0.35])
# #temp_zon_av_res_a1 = temp_zon_av_reg_SAM_a1 - temp_zon_av_reg_SAM
# #bound = max(abs(temp_zon_av_res_a1.min()),abs(temp_zon_av_res_a1.max()))
# #bound = 0.41
# dT_dy_res_a1 = 1e7*(dT_dy_reg_SAM_a1 - dT_dy_reg_SAM)
# #bound = max(abs(dT_dy_res_a1.min()),abs(dT_dy_res_a1.max()))
# bound = 4.0
# for l in range(num_levs):
#     levs_T[l] = round(-bound + 2*bound*l/(num_levs-1),1)
# #im_a1 = a1_ax.contourf(lat,press,temp_zon_av_res_a1, vmin=-bound,vmax=bound,cmap=newcmap, levels=levs_T)
# im_a1 = a1_ax.contourf(lat,press,dT_dy_res_a1, vmin=-bound,vmax=bound,cmap=newcmap, levels=levs_T)
# #plt.title("(a) Temperature anomalies, a=1", fontsize=bigsize,loc = 'left')
# plt.title("(a) dT/dy anomalies, a=1", fontsize=bigsize,loc = 'left')
# #plt.xlabel('Latitude',fontsize=bigsize)
# plt.ylabel('Pressure [hPa]', fontsize=bigsize)
# plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
# plt.yticks([ ])
# plt.yticks([200, 400, 600, 800, 1000 ],fontsize = smallsize)
# a1_ax.tick_params(axis='x',length=10, width = 3)
# a1_ax.tick_params(axis='y',length=10, width = 3)
# plt.gca().invert_yaxis()
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# cbar_a1_ax = fig.add_axes([0.48, 0.55, 0.01, 0.35]) 
# cb_a1 = fig.colorbar(im_a1, cax=cbar_a1_ax) 
# #cb_a1.set_label( "[K]", fontsize = bigsize) 
# cb_a1.set_label( "[$10^{-4}$ K km $ ^{-1}$]", fontsize = bigsize) 
# cb_a1.ax.tick_params(labelsize=smallsize)


# # #a2_ax = plt.subplot(2,2,2)
# a2_ax = fig.add_axes([0.66,0.55,0.35,0.35])
# #temp_zon_av_res_a2 = temp_zon_av_reg_SAM_a2 - temp_zon_av_reg_SAM
# #bound = max(abs(temp_zon_av_res_a2.min()),abs(temp_zon_av_res_a2.max()))
# #bound = 1.4
# dT_dy_res_a2 = 1e7*(dT_dy_reg_SAM_a2 - dT_dy_reg_SAM)
# bound = max(abs(dT_dy_res_a2.min()),abs(dT_dy_res_a2.max()))
# for l in range(num_levs):
#     levs_T[l] = round(-bound + 2*bound*l/(num_levs-1),3)

# #im_a2 = a2_ax.contourf(lat,press,temp_zon_av_res_a2, vmin=-bound,vmax=bound,cmap=newcmap, levels=levs_T)
# im_a2 = a2_ax.contourf(lat,press,dT_dy_res_a2, vmin=-bound,vmax=bound,cmap=newcmap, levels=levs_T)
# #plt.title("(b) Temperature anomalies, a=-1", fontsize=bigsize,loc = 'left')
# plt.title("(a) dT/dy anomalies, a=-1", fontsize=bigsize,loc = 'left')
# #plt.xlabel('Latitude',fontsize=bigsize)
# #plt.ylabel('Pressure (hPa)', fontsize=bigsize)
# plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
# plt.yticks([ ])
# plt.yticks([200, 400, 600, 800, 1000 ],fontsize = smallsize)
# a2_ax.tick_params(axis='x',length=10, width = 3)
# a2_ax.tick_params(axis='y',length=10, width = 3)
# plt.gca().invert_yaxis()
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# #cbar_a2_ax = fig.add_axes([1.02, 0.07, 0.01, 0.83]) 
# cbar_a2_ax = fig.add_axes([1.02, 0.55, 0.01, 0.35]) 
# cb_a2 = fig.colorbar(im_a2, cax=cbar_a2_ax) 
# #cb_a2.set_label( "[K]", fontsize = bigsize)
# cb_a2.set_label( "[$10^{-4}$ K km $ ^{-1}$]", fontsize = bigsize) 
# cb_a2.ax.tick_params(labelsize=smallsize)

# #a3_ax = plt.subplot(2,2,3)
# a3_ax = fig.add_axes([0.12,0.07,0.35,0.35])
# #temp_zon_av_res_a3 = temp_zon_av_reg_SAM_a3 - temp_zon_av_reg_SAM
# dT_dy_res_a3 = 1e7*(dT_dy_reg_SAM_a3 - dT_dy_reg_SAM)
# #bound = max(abs(temp_zon_av_res_a3.min()),abs(temp_zon_av_res_a3.max()))
# bound = max(abs(dT_dy_res_a3.min()),abs(dT_dy_res_a3.max()))
# for l in range(num_levs):
#     levs_T[l] = round(-bound + 2*bound*l/(num_levs-1),1)

# #im_a3 = a3_ax.contourf(lat,press,temp_zon_av_res_a3, vmin=-bound,vmax=bound,cmap=newcmap, levels=levs_T) 
# im_a3 = a3_ax.contourf(lat,press,dT_dy_res_a3, vmin=-bound,vmax=bound,cmap=newcmap, levels=levs_T) 
# #plt.title("(c) Temperature anomalies, a=3", fontsize=bigsize,loc = 'left')
# plt.title("(a) dT/dy anomalies, a=3", fontsize=bigsize,loc = 'left')
# plt.xlabel('Latitude',fontsize=bigsize)
# plt.ylabel('Pressure [hPa]', fontsize=bigsize)
# plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
# plt.yticks([200, 400, 600, 800, 1000 ],fontsize = smallsize)
# a3_ax.tick_params(axis='x',length=10, width = 3)
# a3_ax.tick_params(axis='y',length=10, width = 3)
# plt.gca().invert_yaxis()
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# cbar_a3_ax = fig.add_axes([0.48, 0.07, 0.01, 0.35]) 
# cb_a3 = fig.colorbar(im_a3, cax=cbar_a3_ax) 
# #cb_a3.set_label( "[K]", fontsize = bigsize)
# cb_a3.set_label( "[$10^{-4}$ K km $ ^{-1}$]", fontsize = bigsize)  
# cb_a3.ax.tick_params(labelsize=smallsize)

# # #AC_acc_ax = plt.subplot(2,2,4)
# a4_ax = fig.add_axes([0.66,0.07,0.35,0.35])
# #temp_zon_av_res_a4 = temp_zon_av_reg_SAM_a4 - temp_zon_av_reg_SAM
# #bound = max(abs(temp_zon_av_res_a4.min()),abs(temp_zon_av_res_a4.max()))
# #bound=1.4
# dT_dy_res_a4 = 1e7*(dT_dy_reg_SAM_a4 - dT_dy_reg_SAM)
# bound = max(abs(dT_dy_res_a4.min()),abs(dT_dy_res_a4.max()))
# for l in range(num_levs):
#     levs_T[l] = round(-bound + 2*bound*l/(num_levs-1),1)

# #im_a4 = a4_ax.contourf(lat,press,temp_zon_av_res_a4, vmin=-bound,vmax=bound,cmap=newcmap, levels=levs_T)
# im_a4 = a4_ax.contourf(lat,press,dT_dy_res_a4, vmin=-bound,vmax=bound,cmap=newcmap, levels=levs_T) 
# #plt.title("(d) Temperature anomalies, a=-3", fontsize=bigsize,loc = 'left')
# plt.title("(a) dT/dy anomalies, a=-3", fontsize=bigsize,loc = 'left')
# plt.xlabel('Latitude',fontsize=bigsize)
# #plt.ylabel('Pressure (hPa)', fontsize=bigsize)
# plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
# #plt.yticks([ ])
# plt.yticks([200, 400, 600, 800, 1000 ],fontsize = smallsize)
# a4_ax.tick_params(axis='x',length=10, width = 3)
# a4_ax.tick_params(axis='y',length=10, width = 3)
# plt.gca().invert_yaxis()
# # ax.spines['top'].set_visible(False)
# cbar_a4_ax = fig.add_axes([1.02, 0.07, 0.01, 0.35]) 
# cb_a4 = fig.colorbar(im_a4, cax=cbar_a4_ax) 
# #cb_a4.set_label( "[K]", fontsize = bigsize)
# cb_a4.set_label( "[$10^{-4}$ K km $ ^{-1}$]", fontsize = bigsize)  
# cb_a4.ax.tick_params(labelsize=smallsize)


### Paper Figure 3:  power spectrum & low vs. high pass autocorrelation###

# pspec_nc = Dataset("ACRE_pspecs.nc",'r',format="NETCDF4")
# pspec_freqs = pspec_nc.variables["freqs"][:]
# eddy_pspec = pspec_nc.variables["pow"][:]
# eddy_pspec_a1 = pspec_nc.variables["pow_a1"][:]
# eddy_pspec_a2 = pspec_nc.variables["pow_a-1"][:]
# eddy_pspec_a3 = pspec_nc.variables["pow_a3"][:]
# eddy_pspec_a4 = pspec_nc.variables["pow_a-3"][:]
# pspec_nc.close()


# SAM_PC1_high_filt = specfilt_Butterworth(SAM_PC1,1/50,3)
# SAM_PC1_high_filt_a1 = specfilt_Butterworth(SAM_PC1_forc_a1,1/50,3)
# SAM_PC1_high_filt_a2 = specfilt_Butterworth(SAM_PC1_forc_a2,1/50,3)
# SAM_PC1_high_filt_a3 = specfilt_Butterworth(SAM_PC1_forc_a3,1/50,3)
# SAM_PC1_high_filt_a4 = specfilt_Butterworth(SAM_PC1_forc_a4,1/50,3)
# SAM_PC1_low_filt = specfilt_Butterworth(SAM_PC1,1/50,-3)
# SAM_PC1_low_filt_a1 = specfilt_Butterworth(SAM_PC1_forc_a1,1/50,-3)
# SAM_PC1_low_filt_a2 = specfilt_Butterworth(SAM_PC1_forc_a2,1/50,-3)
# SAM_PC1_low_filt_a3 = specfilt_Butterworth(SAM_PC1_forc_a3,1/50,-3)
# SAM_PC1_low_filt_a4 = specfilt_Butterworth(SAM_PC1_forc_a4,1/50,-3)


# AC_SAM_high = ACF_full(SAM_PC1_high_filt)
# AC_SAM_high_a1 = ACF_full(SAM_PC1_high_filt_a1)
# AC_SAM_high_a2 = ACF_full(SAM_PC1_high_filt_a2)
# AC_SAM_high_a3 = ACF_full(SAM_PC1_high_filt_a3)
# AC_SAM_high_a4 = ACF_full(SAM_PC1_high_filt_a4)
# AC_SAM_low = ACF_full(SAM_PC1_low_filt)
# AC_SAM_low_a1 = ACF_full(SAM_PC1_low_filt_a1)
# AC_SAM_low_a2 = ACF_full(SAM_PC1_low_filt_a2)
# AC_SAM_low_a3 = ACF_full(SAM_PC1_low_filt_a3)
# AC_SAM_low_a4 = ACF_full(SAM_PC1_low_filt_a4)


# fig = plt.figure( figsize = (23, 9) ) #GRL proportions?
# fig.subplots_adjust(left =0.1, bottom = 0.25, top = 0.9, hspace = 0.6, wspace = 0.65, right = 0.9)
# bigsize = 38
# smallsize= 33
# wid=4.5

# l1 = 0
# l2 = 50


# pspec_ax = plt.subplot(1,3,1)
# pspec_ax.plot(pspec_freqs[f1:f2], eddy_pspec[f1:f2],color='black',label='control',linewidth=wid)
# pspec_ax.plot(pspec_freqs[f1:f2], eddy_pspec_a1[f1:f2],color=col_a1,label='a=1',linewidth=wid)
# pspec_ax.plot(pspec_freqs[f1:f2], eddy_pspec_a2[f1:f2],color=col_a2,label='a=-1',linewidth=wid)
# pspec_ax.plot(pspec_freqs[f1:f2], eddy_pspec_a3[f1:f2],color=col_a3,label='a=3',linewidth=wid)
# pspec_ax.plot(pspec_freqs[f1:f2], eddy_pspec_a4[f1:f2],color=col_a4,label='a=-3',linewidth=wid)
# pspec_ax.set_xscale('log')
# plt.xticks([10**(-4),10**(-3), 10**-(2), 10**(-1)],fontsize=smallsize)
# #plt.yticks([0.2,0.4,0.6,0.8,1.0,1.2,1.4],fontsize=smallsize)
# plt.yticks([0,0.3,0.6,0.9,1.2,1.5],fontsize=smallsize)
# plt.xlim([5e-5,0.2])
# plt.ylim([0,1.5])
# #plt.axvline(x = 1./50, color = 'k', linestyle = 'dotted', linewidth = 1.0)
# pspec_ax.legend(frameon=False, fontsize=smallsize-13, loc = 'upper right',ncol=1)
# plt.xlabel('Frequency [day$ ^{-1}$]', fontsize=bigsize)
# plt.ylabel('Power [day$ ^{-2}$]', fontsize=bigsize)
# pspec_ax.tick_params(axis='x',length=10, width = 3)
# pspec_ax.tick_params(axis='y',length=10, width = 3)
# plt.title("(a) Power spectra of m", fontsize=bigsize,loc = 'left')
# pspec_ax.spines['top'].set_visible(False)
# pspec_ax.spines['right'].set_visible(False)

# acc_ax = plt.subplot(1,3,2)
# acc_ax.plot(Lags[l1:l2],AC_SAM_high[l1:l2],color='black',linewidth=wid)
# acc_ax.plot(Lags[l1:l2],AC_SAM_high_a1[l1:l2],color=col_a1,linewidth=wid)
# acc_ax.plot(Lags[l1:l2],AC_SAM_high_a2[l1:l2],color=col_a2,linewidth=wid)
# acc_ax.plot(Lags[l1:l2],AC_SAM_high_a3[l1:l2],color=col_a3,linewidth=wid)
# acc_ax.plot(Lags[l1:l2],AC_SAM_high_a4[l1:l2],color=col_a4,linewidth=wid)
# plt.xlabel('Lag [days]', fontsize=bigsize)
# plt.ylabel('Autocorrelation', fontsize=bigsize)
# plt.yticks([-0.2,0.0,0.2,0.4,0.6,0.8,1.0],fontsize=smallsize)
# plt.xlim([l1,l2-1])
# plt.xticks(Lags[l1:l2:10],fontsize=smallsize)
# acc_ax.tick_params(axis='x',length=10, width = wid)
# acc_ax.tick_params(axis='y',length=10, width = wid)
# plt.title("(b) $C_{ZZ}$ (high-pass)", fontsize=bigsize,loc = 'left')
# acc_ax.spines['top'].set_visible(False)
# acc_ax.spines['right'].set_visible(False)

# acc_low_ax = plt.subplot(1,3,3)
# acc_low_ax.plot(Lags[l1:l2],AC_SAM_low[l1:l2],color='black',linewidth=wid)
# acc_low_ax.plot(Lags[l1:l2],AC_SAM_low_a1[l1:l2],color=col_a1,linewidth=wid)
# acc_low_ax.plot(Lags[l1:l2],AC_SAM_low_a2[l1:l2],color=col_a2,linewidth=wid)
# acc_low_ax.plot(Lags[l1:l2],AC_SAM_low_a3[l1:l2],color=col_a3,linewidth=wid)
# acc_low_ax.plot(Lags[l1:l2],AC_SAM_low_a4[l1:l2],color=col_a4,linewidth=wid)
# plt.xlabel('Lag [days]', fontsize=bigsize)
# #plt.ylabel('Autocorrelation', fontsize=bigsize)
# #plt.yticks([0.2,0.4,0.6,0.8,1.0],fontsize=smallsize)
# plt.yticks([-0.2,0.0,0.2,0.4,0.6,0.8,1.0],fontsize=smallsize)
# plt.xlim([l1,l2-1])
# plt.ylim([-0.2,1.0])
# plt.xticks(Lags[l1:l2:10],fontsize=smallsize)
# acc_low_ax.tick_params(axis='x',length=10, width = 3)
# acc_low_ax.tick_params(axis='y',length=10, width = 3)
# plt.title("(c) $C_{ZZ}$ (low-pass)", fontsize=bigsize,loc = 'left')
# acc_low_ax.spines['top'].set_visible(False)
# acc_low_ax.spines['right'].set_visible(False)

# plt.savefig('ACRE_paper_fig3.png')
# plt.savefig('ACRE_paper_fig3.pdf')

#### Supplementary autocorrelation plot ###
# fig = plt.figure( figsize = (23, 19) ) #GRL proportions?
# fig.subplots_adjust(left =0.1, bottom = 0.1, top = 0.95, hspace = 0.05, wspace = 0.25, right = 1.0)
# bigsize = 42
# smallsize=35

# wid=3
# a1_ax = fig.add_axes([0.1,0.55,0.35,0.35])
# a1_ax.plot(Lags[l1:l2],AC_SAM[l1:l2],color = 'black',label='control',linewidth=wid)
# a1_ax.plot(Lags[l1:l2],AC_SAM_a0_b_a1[l1:l2],color = blue_rgba,label='+Eddies',linestyle='dashed',linewidth=wid)
# a1_ax.plot(Lags[l1:l2],AC_SAM_a0_r_a1[l1:l2],color = red_rgba,label='+Friction',linestyle='dashed',linewidth=wid)
# a1_ax.plot(Lags[l1:l2],AC_SAM_a0_m_a1[l1:l2],color = orange_rgba,label='+MMC',linestyle='dashed',linewidth=wid)
# a1_ax.plot(Lags[l1:l2],AC_SAM_a0_brm_a1[l1:l2],color = purple_rgba,label='+Total',linestyle='dashed',linewidth=wid)
# a1_ax.plot(Lags[l1:l2],AC_SAM_forc_a1[l1:l2],color = purple_rgba,label='ACRE',linewidth=wid)

# #plt.xlabel('Lag (days)',fontsize=bigsize)
# plt.ylabel('Autocorrelation',fontsize=bigsize)
# a1_ax.spines['top'].set_visible(False)
# a1_ax.spines['right'].set_visible(False)
# plt.xticks([0, 5, 10, 15, 20, 25, 30, 35],fontsize = smallsize)
# #plt.xticks([ ])
# plt.yticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],fontsize=smallsize)
# plt.xlim([0,35])
# a1_ax.legend(frameon=False,fontsize=smallsize-5)
# a1_ax.tick_params(axis='y',length=10, width = 3)
# a1_ax.tick_params(axis='x',length=10, width = 3)
# plt.title('(i) a=1',loc='left',fontsize=smallsize)


# a2_ax = fig.add_axes([0.55,0.55,0.35,0.35])
# a2_ax.plot(Lags[l1:l2],AC_SAM[l1:l2],color = 'black',label='control',linewidth=wid)
# a2_ax.plot(Lags[l1:l2],AC_SAM_forc_a2[l1:l2],color = purple_rgba,label='ACRE',linewidth=wid)
# a2_ax.plot(Lags[l1:l2],AC_SAM_a0_b_a2[l1:l2],color = blue_rgba,label='+Eddies',linestyle='dashed',linewidth=wid)
# a2_ax.plot(Lags[l1:l2],AC_SAM_a0_r_a2[l1:l2],color = red_rgba,label='+Friction',linestyle='dashed',linewidth=wid)
# a2_ax.plot(Lags[l1:l2],AC_SAM_a0_m_a2[l1:l2],color = orange_rgba,label='+MMC',linestyle='dashed',linewidth=wid)
# a2_ax.plot(Lags[l1:l2],AC_SAM_a0_brm_a2[l1:l2],color = purple_rgba,label='+Total',linestyle='dashed',linewidth=wid)
# #plt.xlabel('Lag (days)',fontsize=bigsize)
# #plt.ylabel('Autocorrelation',fontsize=bigsize)
# a2_ax.spines['top'].set_visible(False)
# a2_ax.spines['right'].set_visible(False)
# plt.xticks([0, 5, 10, 15, 20, 25, 30, 35],fontsize = smallsize)
# #plt.xticks([ ])
# plt.yticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],fontsize=smallsize)
# #plt.yticks([ ])
# plt.xlim([0,35])
# #a2_ax.legend(frameon=False,fontsize=bigsize)
# a2_ax.tick_params(axis='y',length=10, width = 3)
# a2_ax.tick_params(axis='x',length=10, width = 3)
# plt.title('(ii) a=-1',loc='left',fontsize=smallsize)

# a3_ax = fig.add_axes([0.1,0.1,0.35,0.35])
# a3_ax.plot(Lags[l1:l2],AC_SAM[l1:l2],color = 'black',label='control',linewidth=wid)
# a3_ax.plot(Lags[l1:l2],AC_SAM_forc_a3[l1:l2],color = purple_rgba,label='ACRE',linewidth=wid)
# a3_ax.plot(Lags[l1:l2],AC_SAM_a0_b_a3[l1:l2],color = blue_rgba,label='+Eddies',linestyle='dashed',linewidth=wid)
# a3_ax.plot(Lags[l1:l2],AC_SAM_a0_r_a3[l1:l2],color = red_rgba,label='+Friction',linestyle='dashed',linewidth=wid)
# a3_ax.plot(Lags[l1:l2],AC_SAM_a0_m_a3[l1:l2],color = orange_rgba,label='+MMC',linestyle='dashed',linewidth=wid)
# a3_ax.plot(Lags[l1:l2],AC_SAM_a0_brm_a3[l1:l2],color = purple_rgba,label='+Total',linestyle='dashed',linewidth=wid)
# plt.xlabel('Lag [days]',fontsize=bigsize)
# plt.ylabel('Autocorrelation',fontsize=bigsize)
# a3_ax.spines['top'].set_visible(False)
# a3_ax.spines['right'].set_visible(False)
# plt.xticks([0, 5, 10, 15, 20, 25, 30, 35],fontsize = smallsize)
# plt.yticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],fontsize=smallsize)
# plt.xlim([0,35])
# a3_ax.tick_params(axis='x',length=10, width = 3)
# a3_ax.tick_params(axis='y',length=10, width = 3)
# plt.title('(iii) a=3',loc='left',fontsize=smallsize)

# a4_ax = fig.add_axes([0.55,0.1,0.35,0.35])
# a4_ax.plot(Lags[l1:l2],AC_SAM[l1:l2],color = 'black',label='control',linewidth=wid)
# a4_ax.plot(Lags[l1:l2],AC_SAM_forc_a4[l1:l2],color = purple_rgba,label='ACRE',linewidth=wid)
# a4_ax.plot(Lags[l1:l2],AC_SAM_a0_b_a4[l1:l2],color = blue_rgba,label='+Eddies',linestyle='dashed',linewidth=wid)
# a4_ax.plot(Lags[l1:l2],AC_SAM_a0_r_a4[l1:l2],color = red_rgba,label='+Friction',linestyle='dashed',linewidth=wid)
# a4_ax.plot(Lags[l1:l2],AC_SAM_a0_m_a4[l1:l2],color = orange_rgba,label='+MMC',linestyle='dashed',linewidth=wid)
# a4_ax.plot(Lags[l1:l2],AC_SAM_a0_brm_a4[l1:l2],color = purple_rgba,label='+Total',linestyle='dashed',linewidth=wid)
# plt.xlabel('Lag [days]',fontsize=bigsize)
# #plt.ylabel('Autocorrelation',fontsize=bigsize)
# a4_ax.spines['top'].set_visible(False)
# a4_ax.spines['right'].set_visible(False)
# plt.xticks([0, 5, 10, 15, 20, 25, 30, 35],fontsize = smallsize)
# plt.yticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],fontsize=smallsize)
# #plt.yticks([ ])
# plt.xlim([0,35])
# #a4_ax.legend(frameon=False,fontsize=bigsize)
# a4_ax.tick_params(axis='x',length=10, width = 3)
# a4_ax.tick_params(axis='y',length=10, width = 3)
# plt.title('(iv) a=-3',loc='left',fontsize=smallsize)

# plt.savefig('ACRE_suppfig1.png')
# plt.savefig('ACRE_suppfig1.pdf')


#### Paper Figure 2, AM structure

# u_zon_av_reg_SAM = np.zeros((len(press),len(lat)),dtype='d')
# u_zon_av_reg_SAM_a1 = np.zeros((len(press),len(lat)),dtype='d')
# u_zon_av_reg_SAM_a3 = np.zeros((len(press),len(lat)),dtype='d')

# dT_dy_reg_SAM = np.zeros((len(press),len(lat)),dtype='d')
# dT_dy_reg_SAM_a1 = np.zeros((len(press),len(lat)),dtype='d')

# N_sq_reg_SAM = np.zeros((len(press),len(lat)),dtype='d')
# N_sq_reg_SAM_a1 = np.zeros((len(press),len(lat)),dtype='d')


     
# for l in range(len(lat)):
#             for p in range(len(press)):
              
#                     u_zon_av_reg_SAM[p,l] = CCF_full(SAM_PC1,u_zon_av_nd[:,p,l])[0]/CCF_full(SAM_PC1,SAM_PC1)[0]
#                     u_zon_av_reg_SAM_a1[p,l] = CCF_full(SAM_PC1_forc_a1,u_zon_av_forc_a1_nd[:,p,l])[0]/CCF_full(SAM_PC1_forc_a1, SAM_PC1_forc_a1)[0]
                           
#                     dT_dy_reg_SAM[p,l] = CCF_full(SAM_PC1,dT_dy[:,p,l])[0]/CCF_full(SAM_PC1,SAM_PC1)[0]
#                     dT_dy_reg_SAM_a1[p,l] = CCF_full(SAM_PC1_forc_a1,dT_dy_a1[:,p,l])[0]/CCF_full(SAM_PC1_forc_a1, SAM_PC1_forc_a1)[0]
                
#                     N_sq_reg_SAM[p,l] = CCF_full(SAM_PC1,N_sq[:,p,l])[0]/CCF_full(SAM_PC1,SAM_PC1)[0]
#                     N_sq_reg_SAM_a1[p,l] = CCF_full(SAM_PC1_forc_a1,N_sq_a1[:,p,l])[0]/CCF_full(SAM_PC1_forc_a1, SAM_PC1_forc_a1)[0]
            
                   
# fig = plt.figure( figsize = (23, 7) ) #GRL proportions?
# fig.subplots_adjust(left =0.07, bottom = 0.25, top = 0.9, hspace = 0.40, wspace = 0.57, right = 0.85)
# bigsize = 31
# smallsize = 28

# reg_diff = (u_zon_av_reg_SAM_a1 - u_zon_av_reg_SAM)
# dT_dy_diff = 1e6*(dT_dy_reg_SAM_a1 - dT_dy_reg_SAM)
# N_sq_diff = 1e5*(N_sq_reg_SAM_a1 - N_sq_reg_SAM)

# bound_u =  1.02*max(abs(reg_diff.min()),abs(reg_diff.max()))
# bound_grad =  3/3.1*max(abs(dT_dy_diff.min()),abs(dT_dy_diff.max()))
# #bound_grad = 3.55
# bound_stat = 2/2.1*max(abs(N_sq_diff.min()),abs(N_sq_diff.max()))
# #bound_stat = 3.0

# num_levs = 15
# num_levs_stat = 11
# levs_u = np.zeros(num_levs,dtype='d')
# levs_grad = np.zeros(num_levs,dtype='d')
# levs_stat = np.zeros(num_levs_stat,dtype='d')
# for l in range(num_levs):
#     levs_u[l] = round(-bound_u + 2*bound_u*l/(num_levs-1),2)
#     levs_grad[l] = round(-bound_grad + 2*bound_grad*l/(num_levs-1),2)
# for l in range(num_levs_stat):
#     levs_stat[l] = round(-bound_stat + 2*bound_stat*l/(num_levs_stat-1),1)




# baro_ax = plt.subplot(1,3,1)
# im_baro = baro_ax.contourf(lat,press, 3/3.1*dT_dy_diff, vmin=-bound_grad,vmax=bound_grad,cmap=newcmap,levels=levs_grad)
# im_baro_ctrl = baro_ax.contour(lat,press, dT_dy_reg_SAM, colors='black',levels=8)
# plt.xlabel('Latitude',fontsize=bigsize)
# plt.ylabel('Pressure [hPa]', fontsize=bigsize)
# plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
# plt.yticks([200, 400, 600, 800 ],fontsize = smallsize)
# # plt.axvline(x = -50, color = 'k', linestyle = 'dotted', linewidth = 1.4)
# # plt.axvline(x = -31, color = 'k', linestyle = 'dotted', linewidth = 1.4)
# plt.gca().invert_yaxis()
# plt.title("(a) d$\\bar{T}$/dy", fontsize=bigsize,loc = 'left')
# cbar_baro_ax = fig.add_axes([0.265, 0.25, 0.01, 0.65])
# cb_baro = fig.colorbar(im_baro, cax=cbar_baro_ax) 
# cb_baro.set_label( " [$10^{-3} $ K km $ ^{-1}$]", fontsize = bigsize  )
# cb_baro.ax.tick_params(labelsize=smallsize)
# baro_ax.tick_params(axis='x',length=10, width = 3)
# baro_ax.tick_params(axis='y',length=10, width = 3)


# stat_ax = plt.subplot(1,3,2)
# im_stat = stat_ax.contourf(lat,press, 2/2.1*N_sq_diff, vmin=-bound_stat,vmax=bound_stat,cmap=newcmap,levels=levs_stat)
# im_stat_ctrl = stat_ax.contour(lat,press,N_sq_reg_SAM , colors='black',levels=8)
# plt.xlabel('Latitude',fontsize=bigsize)
# #plt.ylabel('Pressure [hPa]', fontsize=bigsize)
# plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
# plt.yticks([200, 400, 600, 800 ],fontsize = 1)
# plt.yticks(color='w')
# plt.gca().invert_yaxis()
# plt.title("(b) $\\bar{N}^{2}$", fontsize=bigsize,loc = 'left')
# cbar_stat_ax = fig.add_axes([0.561, 0.25, 0.01, 0.65])
# cb_stat = fig.colorbar(im_stat, cax=cbar_stat_ax) 
# cb_stat.set_label( "[$10^{-5} s^{-2}$]", fontsize = bigsize  )
# cb_stat.ax.tick_params(labelsize=smallsize)
# stat_ax.tick_params(axis='x',length=10, width = 3)
# stat_ax.tick_params(axis='y',length=10, width = 3)

# u_ax = plt.subplot(1,3,3)
# im_u = u_ax.contourf(lat,press, reg_diff, cmap = newcmap2, vmin=-bound_u, vmax=bound_u, levels=levs_u)
# im_u_ctrl = u_ax.contour(lat,press,u_zon_av_reg_SAM , colors='black',levels=8)
# plt.xlabel('Latitude',fontsize=bigsize)
# #plt.ylabel('Pressure [hPa]', fontsize=bigsize)

# plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
# plt.yticks([200, 400, 600, 800 ],fontsize = 1)
# plt.yticks(color='w')
# plt.gca().invert_yaxis()
# plt.title("(c) $\\bar{u}$", fontsize=bigsize,loc = 'left')
# cbar_u_ax = fig.add_axes([0.857, 0.25, 0.01, 0.65])
# cb_u = fig.colorbar(im_u, cax=cbar_u_ax) 
# cb_u.set_label( "[m $ s^{-1}$]", fontsize = bigsize  )
# cb_u.ax.tick_params(labelsize=smallsize)
# u_ax.tick_params(axis='x',length=10, width = 3)
# u_ax.tick_params(axis='y',length=10, width = 3)

# plt.savefig('ACRE_paper_fig2.png')
# plt.savefig('ACRE_paper_fig2.pdf')

###Look at contour intervals ###
# fig,ax = plt.subplots()
# im_ctrl = ax.contourf(lat,press, 1e7*dT_dy_reg_SAM,levels=8)
# ax.tick_params(axis='x',length=10, width = 3)
# ax.tick_params(axis='y',length=10, width = 3)
# plt.title('dT_dy')
# cbar = fig.add_axes([1.01, 0.1, 0.01, 0.85])
# cb = fig.colorbar(im_ctrl, cax=cbar) 
# #cb.set_label( "[m $ s^{-1}$]", fontsize = bigsize  )
# #cb.tick_params(labelsize=smallsize)

# fig,ax = plt.subplots()
# im_ctrl = ax.contourf(lat,press, 1e5*N_sq_reg_SAM,levels=8)
# ax.tick_params(axis='x',length=10, width = 3)
# ax.tick_params(axis='y',length=10, width = 3)
# plt.title('N^2')
# cbar = fig.add_axes([1.01, 0.1, 0.01, 0.85])
# cb = fig.colorbar(im_ctrl, cax=cbar) 
# #cb.set_label( "[m $ s^{-1}$]", fontsize = bigsize  )
# #cb.tick_params(labelsize=smallsize)

# fig,ax = plt.subplots()
# im_ctrl = ax.contourf(lat,press, u_zon_av_reg_SAM,levels=8)
# ax.tick_params(axis='x',length=10, width = 3)
# ax.tick_params(axis='y',length=10, width = 3)
# plt.title('u')
# cbar = fig.add_axes([1.01, 0.1, 0.01, 0.85])
# cb = fig.colorbar(im_ctrl, cax=cbar) 
# #cb.set_label( "[m $ s^{-1}$]", fontsize = bigsize  )
# #cb.tick_params(labelsize=smallsize)



#### Paper Figure 4 ####

# lag=8
# ccf = np.zeros((len(press),len(lat)),dtype='d')
# ccf_a1 = np.zeros((len(press),len(lat)),dtype='d')
# ccf_a2 = np.zeros((len(press),len(lat)),dtype='d')
# ccf_a3 = np.zeros((len(press),len(lat)),dtype='d')
# ccf_a4 = np.zeros((len(press),len(lat)),dtype='d')

# for l in range(len(lat)):
#     for p in range(len(press)):
#         ccf[p,l] = CCF_full(SAM_PC1,vorflux_zon_av_nd[:,p,l])[lag]/CCF_full(SAM_PC1,SAM_PC1)[lag]
#         ccf_a1[p,l] = CCF_full(SAM_PC1_forc_a1,vorflux_zon_av_forc_a1_nd[:,p,l])[lag]/CCF_full(SAM_PC1_forc_a1, SAM_PC1_forc_a1)[lag]
#         ccf_a2[p,l] = CCF_full(SAM_PC1_forc_a2,vorflux_zon_av_forc_a2_nd[:,p,l])[lag]/CCF_full(SAM_PC1_forc_a2, SAM_PC1_forc_a2)[lag]
#         ccf_a3[p,l] = CCF_full(SAM_PC1_forc_a3,vorflux_zon_av_forc_a3_nd[:,p,l])[lag]/CCF_full(SAM_PC1_forc_a3, SAM_PC1_forc_a3)[lag]
#         ccf_a4[p,l] = CCF_full(SAM_PC1_forc_a4,vorflux_zon_av_forc_a4_nd[:,p,l])[lag]/CCF_full(SAM_PC1_forc_a4, SAM_PC1_forc_a4)[lag]

        
# ccf = 86400*ccf/S
# ccf_a1 = 86400*ccf_a1/S
# ccf_a2 = 86400*ccf_a2/S
# ccf_a3 = 86400*ccf_a3/S
# ccf_a4 = 86400*ccf_a4/S

# fig = plt.figure( figsize = (23, 19) ) #GRL proportions?
# fig.subplots_adjust(left =0.1, bottom = 0.1, top = 0.9, hspace = 0.25, wspace = 0.2, right = 0.75)
# bigsize = 45
# smallsize = 40


# ccf_diff_a1 = ccf_a1 - ccf
# ccf_diff_a2 = ccf_a2 - ccf
# ccf_diff_a3 = ccf_a3 - ccf
# ccf_diff_a4 = ccf_a4 - ccf
# base_var = ccf_diff_a3
# bound_all = 1.2*max(np.abs(base_var.min()), np.abs(base_var.max()))
# num_levs = 15
# levs_ccf = np.zeros(num_levs,dtype='d')
# for l in range(num_levs):
#     levs_ccf[l] = round(-bound_all + 2*bound_all*l/(num_levs-1),3)


# a1_ax = plt.subplot(2,2,1)
# im_a1 = a1_ax.contourf(lat,press, ccf_diff_a1, vmin=-bound_all,vmax=bound_all,cmap=newcmap2,levels=levs_ccf)
# im_ctrl = a1_ax.contour(lat,press, ccf, colors='black',levels=6)
# #plt.xlabel('Latitude',fontsize=bigsize)
# plt.ylabel('Pressure [hPa]', fontsize=bigsize)
# plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
# plt.xticks(color='w',fontsize=1)
# plt.yticks([200, 400, 600, 800],fontsize = smallsize)
# plt.gca().invert_yaxis()
# plt.title("(i) a=1", fontsize=bigsize,loc = 'left')
# a1_ax.tick_params(axis='x',length=10, width = 3)
# a1_ax.tick_params(axis='y',length=10, width = 3)

# a2_ax = plt.subplot(2,2,2)
# im_a2 = a2_ax.contourf(lat,press, ccf_diff_a2, vmin=-bound_all,vmax=bound_all,cmap=newcmap2,levels=levs_ccf)
# im_ctrl = a2_ax.contour(lat,press, ccf, colors='black',levels=6)
# #plt.xlabel('Latitude',fontsize=bigsize)
# #plt.ylabel('Pressure [hPa]', fontsize=bigsize)
# plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
# plt.xticks(color='w',fontsize=1)
# plt.yticks([200, 400, 600, 800],fontsize = smallsize)
# plt.yticks(color='w',fontsize=1)
# plt.gca().invert_yaxis()
# plt.title("(ii) a=-1", fontsize=bigsize,loc = 'left')
# a2_ax.tick_params(axis='x',length=10, width = 3)
# a2_ax.tick_params(axis='y',length=10, width = 3)

# a3_ax = plt.subplot(2,2,3)
# im_a3 = a3_ax.contourf(lat,press, ccf_diff_a3, vmin=-bound_all,vmax=bound_all,cmap=newcmap2,levels=levs_ccf)
# im_ctrl = a3_ax.contour(lat,press, ccf, colors='black',levels=6)
# plt.xlabel('Latitude',fontsize=bigsize)
# plt.ylabel('Pressure [hPa]', fontsize=bigsize)
# plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
# plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
# plt.yticks([200, 400, 600, 800],fontsize = smallsize)
# plt.gca().invert_yaxis()
# plt.title("(iii) a=3", fontsize=bigsize,loc = 'left')
# a3_ax.tick_params(axis='x',length=10, width = 3)
# a3_ax.tick_params(axis='y',length=10, width = 3)

# a4_ax = plt.subplot(2,2,4)
# im_a4 = a4_ax.contourf(lat,press, ccf_diff_a4, vmin=-bound_all,vmax=bound_all,cmap=newcmap2,levels=levs_ccf)
# im_ctrl = a4_ax.contour(lat,press, ccf, colors='black',levels=6)
# plt.xlabel('Latitude',fontsize=bigsize)
# #plt.ylabel('Pressure [hPa]', fontsize=bigsize)
# plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
# plt.xticks([-70, -60, -50, -40, -30 ],fontsize = smallsize)
# plt.yticks([200, 400, 600, 800],fontsize = smallsize)
# plt.yticks(color='w',fontsize=1)
# plt.gca().invert_yaxis()
# plt.title("(iv) a=-3", fontsize=bigsize,loc = 'left')
# cbar_a4_ax = fig.add_axes([0.8, 0.1, 0.02, 0.802])
# cb_a4 = fig.colorbar(im_a4, cax=cbar_a4_ax) 
# cb_a4.set_label( " [day $ ^{-1}$]", fontsize = bigsize  )
# cb_a4.ax.tick_params(labelsize=smallsize)
# a4_ax.tick_params(axis='x',length=10, width = 3)
# a4_ax.tick_params(axis='y',length=10, width = 3)
# cb_a4.set_ticks([-0.02,-0.01,0,0.01,0.02])
# plt.savefig('ACRE_paper_fig4.png')
# plt.savefig('ACRE_paper_fig4.pdf')

### Look at contour intervals ###
# fig,ax = plt.subplots()
# im_ctrl = ax.contourf(lat,press, ccf,levels=6)
# cb_ax =  fig.add_axes([0.95, 0.1, 0.02, 0.85])
# cb = fig.colorbar(im_ctrl, cax=cb_ax) 
# cb.set_label( " [days $ ^{-1}$]", fontsize = bigsize  )
# cb.ax.tick_params(labelsize=smallsize)
# ax.tick_params(axis='x',length=10, width = 3)
# ax.tick_params(axis='y',length=10, width = 3)


###Supplementary vertical profile plot ###

# u_prof_reg_Z = np.zeros(len(press),dtype='d')
# u_prof_reg_Z_a1 = np.zeros(len(press),dtype='d')
# u_prof_reg_Z_a2 = np.zeros(len(press),dtype='d')
# u_prof_reg_Z_a3 = np.zeros(len(press),dtype='d')
# u_prof_reg_Z_a4 = np.zeros(len(press),dtype='d')
# for p in range(len(press)):
#     u_prof_reg_Z[p] = CCF_full(SAM_PC1,u_zon_av_nd[:,p,14])[0]/CCF_full(SAM_PC1,SAM_PC1)[0]
#     u_prof_reg_Z_a1[p] = CCF_full(SAM_PC1_forc_a1,u_zon_av_forc_a1_nd[:,p,14])[0]/CCF_full(SAM_PC1_forc_a1,SAM_PC1_forc_a1)[0]
#     u_prof_reg_Z_a2[p] = CCF_full(SAM_PC1_forc_a2,u_zon_av_forc_a2_nd[:,p,14])[0]/CCF_full(SAM_PC1_forc_a2,SAM_PC1_forc_a2)[0]
#     u_prof_reg_Z_a3[p] = CCF_full(SAM_PC1_forc_a3,u_zon_av_forc_a3_nd[:,p,14])[0]/CCF_full(SAM_PC1_forc_a3,SAM_PC1_forc_a3)[0]
#     u_prof_reg_Z_a4[p] = CCF_full(SAM_PC1_forc_a4,u_zon_av_forc_a4_nd[:,p,14])[0]/CCF_full(SAM_PC1_forc_a4,SAM_PC1_forc_a4)[0]


# fig = plt.figure( figsize = (23, 19) ) #GRL proportions?
# fig.subplots_adjust(left =0.1, bottom = 0.1, top = 0.95, hspace = 0.60, wspace = 0.5, right = 0.5)
# bigsize = 45
# smallsize = 40
# wid =3.5
# u_ax = plt.subplot(1,1,1)
# plt.title("a) $\Delta \\bar{u}$", loc = "left", fontsize = bigsize)
# u_ax.plot(u_prof_reg_Z_a1-u_prof_reg_Z, press, color = col_a1 ,label='a=1',linewidth=wid)
# u_ax.plot(u_prof_reg_Z_a2-u_prof_reg_Z, press, color = col_a2 ,label='a=-1',linewidth=wid)
# u_ax.plot(u_prof_reg_Z_a3-u_prof_reg_Z, press, color = col_a3 ,label='a=3',linewidth=wid)
# u_ax.plot(u_prof_reg_Z_a4-u_prof_reg_Z, press, color = col_a4 ,label='a=-3',linewidth=wid)
# plt.xlabel("[m $s^{-1}$]", fontsize = bigsize)
# plt.ylabel("Pressure [hPa]",fontsize=bigsize)
# plt.legend(frameon=False, fontsize = smallsize-5, loc = "lower right")
# u_ax.spines["top"].set_visible(False)
# u_ax.spines["right"].set_visible(False)
# #plt.ylim([-6., 0.])
# #plt.xlim([-0.3, 0.3])
# plt.yticks([200., 400., 600., 800.,1000], fontsize = smallsize)
# plt.xticks([-0.4, -0.2, 0, 0.2,0.4], fontsize = smallsize)
# #plt.xticks([-40., -20., 0., 20., 40.], fontsize = smallsize)
# plt.axvline(x = 0, color = 'k')
# plt.gca().invert_yaxis()
# u_ax.tick_params(axis='x',length=12, width = 4)
# u_ax.tick_params(axis='y',length=12, width = 4)

# plt.savefig("ACRE_suppfig7.png")
# plt.savefig("ACRE_suppfig7.pdf")


#### Eliassen response vs. MMC supplementary figure ###



fv_zon_av_dot_ind = np.zeros((37,36),dtype='d')
fv_zon_av_dot_ind_forc_a1 = np.zeros((37,36),dtype='d')
fv_zon_av_dot_ind_forc_a2 = np.zeros((37,36),dtype='d')
fv_zon_av_dot_ind_forc_a3 = np.zeros((37,36),dtype='d')
fv_zon_av_dot_ind_forc_a4 = np.zeros((37,36),dtype='d')


Omega = 7.27e-5
fs = 2*Omega*np.sin(lat*np.pi/180)

# for p in range(36):
#     for l in range(37):
#         fv_zon_av_dot_ind[l,p] = np.dot(fs[l]*v_zon_av_nd[:,p,l],SAM_PC1)/np.dot(SAM_PC1,SAM_PC1)
        # fv_zon_av_dot_ind_forc_a1[l,p] = np.dot(fs[l]*v_zon_av_forc_a1_nd[:,p,l],SAM_PC1_forc_a1)/np.dot(SAM_PC1_forc_a1,SAM_PC1_forc_a1)
        #fv_zon_av_dot_ind_forc_a3[l,p] = np.dot(fs[l]*v_zon_av_forc_a3_nd[:,p,l],SAM_PC1_forc_a3)/np.dot(SAM_PC1_forc_a3,SAM_PC1_forc_a3)
       
        # fv_zon_av_dot_ind[l,p] = lag_dot(fs[l]*v_zon_av_nd[:,p,l],SAM_PC1,10)/np.dot(SAM_PC1,SAM_PC1)
        # fv_zon_av_dot_ind_forc_a1[l,p] = lag_dot(fs[l]*v_zon_av_forc_a1_nd[:,p,l],SAM_PC1_forc_a1,10)/np.dot(SAM_PC1_forc_a1,SAM_PC1_forc_a1)
        # fv_zon_av_dot_ind_forc_a3[l,p] = lag_dot(fs[l]*v_zon_av_forc_a3_nd[:,p,l],SAM_PC1_forc_a3,10)/np.dot(SAM_PC1_forc_a3,SAM_PC1_forc_a3)
        # vorflux_dot_ind[l,p] = lag_dot(vorflux_zon_av_nd[:,p,l],SAM_PC1,10)/np.dot(SAM_PC1,SAM_PC1)
        # vorflux_dot_ind_forc_a1[l,p] = lag_dot(vorflux_zon_av_forc_a1_nd[:,p,l],SAM_PC1_forc_a1,10)/np.dot(SAM_PC1_forc_a1,SAM_PC1_forc_a1)
        # vorflux_dot_ind_forc_a3[l,p] = lag_dot(vorflux_zon_av_forc_a3_nd[:,p,l],SAM_PC1_forc_a3,10)/np.dot(SAM_PC1_forc_a3,SAM_PC1_forc_a3)
        
        # fv_zon_av_dot_ind[l,p] = CCF_full(fs[l]*v_zon_av_nd[:,p,l],SAM_PC1)[0]/np.dot(SAM_PC1,SAM_PC1)
        # fv_zon_av_dot_ind_forc_a1[l,p] = CCF_full(fs[l]*v_zon_av_forc_a1_nd[:,p,l],SAM_PC1_forc_a1)[0]/np.dot(SAM_PC1_forc_a1,SAM_PC1_forc_a1)
        # fv_zon_av_dot_ind_forc_a3[l,p] = CCF_full(fs[l]*v_zon_av_forc_a3_nd[:,p,l],SAM_PC1_forc_a3)[0]/np.dot(SAM_PC1_forc_a3,SAM_PC1_forc_a3)

# fv_res_a1 = fv_zon_av_dot_ind_forc_a1 - fv_zon_av_dot_ind
# fv_res_a3 = fv_zon_av_dot_ind_forc_a3 - fv_zon_av_dot_ind
#WC = 1/10.


# ncfile = Dataset('Eliassen_response.nc',mode='r',format='NETCDF4')
# Eli_a3 = ncfile.variables['torque'][:]
# press_eli = ncfile.variables['press'][:]
# lat_eli = ncfile.variables['lat'][:]
# ncfile.close()

# fig = plt.figure( figsize = (23, 19) ) #GRL proportions?
# fig.subplots_adjust(left =0.1, bottom = 0.1, top = 0.95, hspace = 0.20, wspace = 0.35, right = 0.9)
# bigsize = 48
# smallsize = 42

# Var = Eli_a3
# base_var = 86400*fv_res_a3
# bound = max(np.abs(base_var.min()), np.abs(base_var.max()))
# #bound = 0.15
# levs_fv = np.zeros(33,dtype='d')
# for l in range(33):
#       levs_fv[l] = round(-bound + 2*bound*l/(33-1),3)

# Eli_ax = plt.subplot(2,1,1)

# im_Eli = Eli_ax.contourf(lat_eli[7:28],press,Var[-1:2:-1,7:28], cmap = newcmap4, levels=levs_fv )
# plt.title("(a) Eliassen response", fontsize=bigsize,loc = 'left')
# plt.gca().invert_yaxis()
# #plt.xlabel('Latitude',fontsize=bigsize)
# plt.ylabel('Pressure [hPa]',fontsize=bigsize)
# plt.xticks([])
# plt.yticks([200,400,600,800],fontsize = smallsize)
# cbar_Eli_ax = fig.add_axes([0.93, 0.1, 0.015, 0.85])
# cb_Eli = fig.colorbar(im_Eli, cax=cbar_Eli_ax)
# cb_Eli.set_label( " [m $ s^{-1} day^{-1}$]", fontsize = bigsize  )
# cb_Eli.ax.tick_params(labelsize=smallsize)
# cb_Eli.set_ticks([-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3])
# Eli_ax.tick_params(axis='x',length=12, width = 4)
# Eli_ax.tick_params(axis='y',length=12, width = 4)

# fv_ax = plt.subplot(2,1,2)
# Var = 86400*fv_res_a3

# im_fv = fv_ax.contourf(lat,press,np.transpose(Var), cmap = newcmap4, levels=levs_fv)
# plt.title("(b) MMC residual", fontsize=bigsize,loc = 'left')
# plt.gca().invert_yaxis()
# plt.xlabel('Latitude',fontsize=bigsize)
# plt.ylabel('Pressure [hPa]',fontsize=bigsize)
# plt.xticks(fontsize = smallsize)
# plt.yticks([200,400,600,800],fontsize = smallsize)
# fv_ax.tick_params(axis='x',length=12, width = 4)
# fv_ax.tick_params(axis='y',length=12, width = 4)
# plt.savefig('ACRE_suppfig8.png')
# plt.savefig('ACRE_suppfig8.pdf')





