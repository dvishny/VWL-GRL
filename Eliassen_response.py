#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 10:20:15 2021

@author: davidvishny

"""

import numpy as np
import pandas
import scipy as sp 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import Dataset
from statsmodels.graphics import tsaplots
from scipy.fftpack import fftfreq
import calendar as cal
from scipy import signal

AM_NC = 'SAM_CW_a-1_retest_clima.nc'
daily_data = Dataset(AM_NC)
fine_press = daily_data.variables['press'][:]
pdiff = np.diff(fine_press)
fine_lat = daily_data.variables['lat'][:]
u_zon_av = daily_data.variables['u_zon_av'][:]
temp_zon_av = daily_data.variables['temp_zon_av'][:]

forc_NC = 'CW_SAM_heating.nc'
forc_data = Dataset(forc_NC)
lat = forc_data.variables['lat'][:]
latb = forc_data.variables['latb'][:]
heat_rate = forc_data.variables['local_heating'][0,:,:,0] #heating field here
heat_rate = heat_rate
press = forc_data.variables['pfull'][:]
press = press[::-1]
#press_coarse = press[::-2]
press_coarse = press

# forc_NC = 'Low_cloud_clim_heating.nc'
# forc_data = Dataset(forc_NC)
# lat = forc_data.variables['lat'][:]
# latb = forc_data.variables['latb'][:]
# heat_rate = forc_data.variables['local_heating'][0,:,:,0]
# press = forc_data.variables['pfull'][:]

Press = np.zeros(np.shape(press),dtype='d')
Heat_rate = np.zeros(np.shape(heat_rate),dtype='d')
for p in range(len(press)):
    Press[p] =  press[-1-p]
    Heat_rate[p,:] = heat_rate[-1-p,:]
press = Press
heat_rate = Heat_rate
#heat_rate_coarse = heat_rate[::2,:]
heat_rate_coarse = heat_rate


####Functions##########

def near_ind(x,arr):
    diff = np.Infinity
    ind = 0
    for i in range(len(arr)):
        if abs(arr[i] - x) < diff:
            ind = i
            diff = abs(arr[i] - x)
    return ind

def unwind(arr): #Convert 2D arr to 1D arr
    return arr.reshape(len(arr[:,0])*len(arr[0,:]))

def rewind(arr):
    return arr.reshape(len(press_coarse),len(lat))

def sgn_mask(x):
    if x >= 0:
        return 1.
    else:
        return np.nan
#####################


l1 = near_ind(-51,fine_lat)
l2 = near_ind(-30,fine_lat)
l1n = near_ind(30,fine_lat)
l2n = near_ind(51,fine_lat)
u_zon_av_vertsum = np.sum(u_zon_av,axis=1)
SAM_ind = u_zon_av_vertsum[:,l1] - u_zon_av_vertsum[:,l2] #positive when poleward
SAM_ind = SAM_ind - np.mean(SAM_ind)
# NAM_ind = u_zon_av_vertsum[:,l2n] - u_zon_av_vertsum[:,l1n] #positive when poleward
# NAM_ind = NAM_ind - np.mean(NAM_ind)

Mask_pos = np.array(list(map(sgn_mask,SAM_ind)))[:,np.newaxis,np.newaxis] #AM>0
Mask_neg = np.array(list(map(sgn_mask,-SAM_ind)))[:,np.newaxis,np.newaxis] #AM<0
u_zon_av_pos = u_zon_av * Mask_pos
u_zon_av_neg = u_zon_av * Mask_neg


T_clim_fine = np.mean(temp_zon_av,axis=0)
u_clim_fine = np.mean(u_zon_av,axis=0)
u_clim_pos_fine = np.nanmean(u_zon_av_pos,axis=0)
u_clim_neg_fine = np.nanmean(u_zon_av_neg,axis=0)
T_clim = np.zeros((len(press_coarse),len(lat)),dtype='d')
u_clim = np.zeros((len(press_coarse),len(lat)),dtype='d')
u_clim_pos = np.zeros((len(press_coarse),len(lat)),dtype='d') #SAM>0
u_clim_neg = np.zeros((len(press_coarse),len(lat)),dtype='d') #SAM<0
lat_rad = lat * np.pi/180.
Lat_rad = np.zeros((len(press_coarse),len(lat)),dtype='d')
Lat = np.zeros((len(press_coarse),len(lat)),dtype='d')
Press = np.zeros((len(press_coarse),len(lat)),dtype='d')
sigma = np.array(press_coarse)/1000.


kappa = 2./7 #C_p/C_v = 1.4 or R/C_p = 2/7?

g = 9.81
T_star = 255. #ref temp. for  buoyancy
a = 6.37e6
ka = 1/(40.*86400)
ks = 1/(4.*86400)
sigma_b = 0.7
H_star = 7.47e3 #scale height
omega = 7.27e-5 #rotation rate
rho = 1.225 #kg/m^3

r = (sigma[:,np.newaxis]-sigma_b)/(1.-sigma_b)

u_anom_pos = np.zeros(np.shape(heat_rate_coarse),dtype='d')
u_anom_neg = np.zeros(np.shape(heat_rate_coarse),dtype='d')
u_anom_pos_sol = np.zeros(np.shape(heat_rate_coarse),dtype='d')
u_anom_pos_dash = np.zeros(np.shape(heat_rate_coarse),dtype='d')
u_anom_neg_sol = np.zeros(np.shape(heat_rate_coarse),dtype='d')
u_anom_neg_dash = np.zeros(np.shape(heat_rate_coarse),dtype='d')

for p in range(len(press_coarse)):
  for l in range(len(lat)):
    L = near_ind(lat[l],fine_lat)
    P = near_ind(press_coarse[p],fine_press)
    # L2 = near_ind(lat[l],lat2)
    # P2 = near_ind(press[p],press2)
    
    # heat_rate[p,l] = heat_rate2[P2,L2]
    T_clim[p,l] = T_clim_fine[P,L]
    u_clim[p,l] = u_clim_fine[P,L]
    u_clim_pos[p,l] = u_clim_pos_fine[P,L]
    u_clim_neg[p,l] = u_clim_neg_fine[P,L]
    Lat_rad[p,l] = lat_rad[l]
    Lat[p,l] = lat[l]
    #Press[p,l] = press[p]
    Press[p,l] = press_coarse[p]

    u_anom_pos[p,l] =  u_clim_pos[p,l] - u_clim[p,l]
    u_anom_neg[p,l] = u_clim_neg[p,l] - u_clim[p,l]


    u_anom_pos_sol[p,l] = u_anom_pos[p,l] * sgn_mask(u_anom_pos[p,l]) #contours
    u_anom_pos_dash[p,l] = u_anom_pos[p,l] * sgn_mask(-u_anom_pos[p,l]) 
    u_anom_neg_sol[p,l] = u_anom_neg[p,l] * sgn_mask(u_anom_neg[p,l])
    u_anom_neg_dash[p,l] = u_anom_neg[p,l] * sgn_mask(-u_anom_neg[p,l])

 
#heat_rate = unwind(heat_rate) #comment/uncomment
heat_rate_coarse = unwind(heat_rate_coarse) #comment/uncomment

B = g/T_star*(T_clim*(press_coarse[-1]/press_coarse[:,np.newaxis])**(kappa) - T_star)
B = unwind(B) #Buoyancy climatology
#B = unwind(g*(T_clim - T_star)/T_star) #Buoyancy climatology


M = unwind((u_clim_pos + a*np.cos(lat_rad[np.newaxis,:])*omega )*a*np.cos(lat_rad[np.newaxis,:])) #ang.  climatology#
alpha_b = unwind(ka + (ks-ka)*(r + np.abs(r))/2.*np.cos(lat_rad[np.newaxis,:])**4) #buoyancy restoring
b_e = g/T_star*heat_rate_coarse/alpha_b #why negative sign? Comment this too.
eps = unwind(2*np.sin(Lat_rad)/(a*np.cos(Lat_rad))**3 * 100.*press_coarse[:,np.newaxis]/H_star) #epsilon
Lat_rad = unwind(Lat_rad)


heat_rate_RP = np.zeros(np.shape(heat_rate_coarse),dtype='d') #reproduce buoyancy forcing from R&P
q_RP = np.zeros(np.shape(heat_rate_coarse),dtype='d')
RP_base_b_e = -4E-13 
DL = 16/ 90**2
DR = 300/ 90**2
H =  25 / (press[-1]- press[0])**2
cent = -50.
# for l in range(len(lat)): #comment/uncomment
#     for p in range(len(press_coarse)):
#       if lat[l] < cent:
#         heat_rate_RP[p,l] = 0*RP_base_b_e/5.*np.exp(-DL*(Lat[p,l]-cent)**2- 0.07*H*(Press[p,l] - Press[-7,l])**2)+ RP_base_b_e*np.exp(-DL*(Lat[p,l]-cent)**2 - H*(Press[p,l] - Press[-1,l])**2) 
#       #heat_rate_RP[p,l] = RP_base_b_e*(1 - (Lat[p,l]-cent)/(Lat[p,0]-cent))*np.exp( - H*(Press[p,l] - Press[-1,l])**4)
#       else:
#         heat_rate_RP[p,l] = 0*RP_base_b_e/5.*np.exp(-DR*(Lat[p,l]-cent)**2 - 0.07*H*(Press[p,l] - Press[-7,l])**2)+ RP_base_b_e*np.exp(-DR*(Lat[p,l]-cent)**2 - H*(Press[p,l] - Press[-1,l])**2)
#bf = unwind(heat_rate_RP) #comment/uncomment

# q1 = 10. #reproduce q_eff from R&P
# q2 = -8.
# DS2 = 70./ 90**2
# DH2 = 5./ (press[-1]- press[0])**2
# DS1 = 160./ 90**2
# DH1 = 80. / (press[-1]- press[0])**2
# cent_l = -60.
# cent_p = 400.

# q1 = 0. #reproduce q_eff from R&P
# q2 = -15.
# DS2 = 250./ 90**2
# DH2 = 60./ (press[-1]- press[0])**2
# DS1 = 160./ 90**2
# DH1 = 80. / (press[-1]- press[0])**2
# cent_l = -55.
# cent_p = 400.
# cent_l2 = -30.

# for l in range(len(lat)):
#     for p in range(len(press)):
#        q_RP[p,l] = q2*np.exp(-DS2*(Lat[p,l]-cent_l)**2 - DH2*(Press[p,l] - cent_p)**2) \
#            + q1*np.exp(-2.*DS1*(Lat[p,l]-cent_l)**2 - DH1*(Press[p,l] - Press[-1,l])**2)
              

########Eliassen Response Calculation#######

#Obtain bottom boundary condition from inversion of q_eff and from simulation?

del_l = 2.5 * np.pi/180.
#del_p = 25.* 100.
del_p = (press_coarse[1] - press_coarse[0])* 100.
 
d_p = np.zeros((len(lat)*len(press_coarse),len(lat)*len(press_coarse)),dtype='d')
d2_p = np.zeros((len(lat)*len(press_coarse),len(lat)*len(press_coarse)),dtype='d')
d_l = np.zeros((len(lat)*len(press_coarse),len(lat)*len(press_coarse)),dtype='d')
d2_l = np.zeros((len(lat)*len(press_coarse),len(lat)*len(press_coarse)),dtype='d')
d3_l = np.zeros((len(lat)*len(press_coarse),len(lat)*len(press_coarse)),dtype='d')
d_p_b = np.zeros((len(lat)*len(press_coarse),len(lat)*len(press_coarse)),dtype='d') #backward difference
D_p = np.zeros((len(lat)*len(press_coarse),len(lat)*len(press_coarse)),dtype='d') #for variables other than chi_E
D2_p = np.zeros((len(lat)*len(press_coarse),len(lat)*len(press_coarse)),dtype='d') #Linearize at edge
D_l = np.zeros((len(lat)*len(press_coarse),len(lat)*len(press_coarse)),dtype='d') #
D2_l = np.zeros((len(lat)*len(press_coarse),len(lat)*len(press_coarse)),dtype='d') #Linearize at edge
I = np.zeros((len(lat)*len(press_coarse),len(lat)*len(press_coarse)),dtype='d')

for j in range(len(lat)*len(press_coarse)): #Derivative operators on unwound arrays
   d_p[j,j] = -1.
   d_l[j,j] = -1.
   d2_p[j,j] = -2.
   d2_l[j,j] = -2.
   D_p[j,j] = -1.
   D_l[j,j] = -1.
   D2_p[j,j] = -2.
   D2_l[j,j] = -2.
   d_p_b[j,j] = 1.
   
   if j > len(lat)-1: 
    d2_p[j,j - len(lat)] = 1
    D2_p[j,j - len(lat)] = 1
    d_p_b[j,j - len(lat)] = -1
   
   
   if j < len(lat)*(len(press_coarse)-1): 
    d_p[j,j + len(lat)] = 1
    d2_p[j,j + len(lat)] = 1
    D_p[j,j + len(lat)] = 1
    D2_p[j,j + len(lat)] = 1
   else:
       D_p[j,:] = D_p[j-len(lat),:]
       d_p[j,j] = 0 #Neumann at surf
       d2_p[j,j] = -1 #Neumann at surf
       #d2_p[j,:] = 0 #Try this
   if j <= len(lat)-1 or j >= len(lat)*(len(press_coarse)-1):
       D2_p[j,:] = 0
       
   if j % len(lat) != len(lat) -1: #latitude
       d_l[j,j+1] = 1.
       d2_l[j,j+1] = 1.
       d3_l[j,j+1] = -2.
       D_l[j,j+1] = 1.
       D2_l[j,j+1] = 1.  
   else:
       D_l[j,:] =  D_l[j-1,:]
       
       
   if j % len(lat) < len(lat) - 2:
       d3_l[j,j+2] = 1.
       
   if  j % len(lat) != 0 :  
      d2_l[j,j-1] = 1.
      d3_l[j,j-1] = 2.
      D2_l[j,j-1] = 1.
   if not  j % len(lat) != len(lat) -1 or not j % len(lat) != 0 :
        D2_l[j,:] = 0
    
   if j % len(lat) > 1:
       d3_l[j,j-2] = -1.
       
   I[j,j] = 1.
    
d_p *= 1./del_p # coarse FD matrices must be adjusted for size. Just comment.

d_l *= 1./del_l
d_p_b *= 1./del_p
d2_p *= 1./del_p**2

d2_l *= 1./del_l**2
d3_l *= 0.5/del_l**3
D_p *= 1./del_p

D_l *= 1./del_l
D2_p *= 1./del_p**2
D2_l *= 1./del_l**2


#S = 1.*(unwind(kappa*(g*(T_clim - T_star)/T_star + g)/(press[:,np.newaxis]*100)) - np.matmul(d_p,B))
S = 1.*(unwind(kappa*(g*(T_clim - T_star)/T_star + g)/(press_coarse[:,np.newaxis]*100)) - np.matmul(d_p,B))

dB_dl = np.matmul(D_l,B)
dM_dl = np.matmul(D_l,M)
dM_dp = np.matmul(D_p,M)
dB_dp = np.matmul(D_p,B)
dS_dl = np.matmul(D_l,S)
D_l_p = np.matmul(D_l,D_p)
d_l_p = np.matmul(d_l,d_p)

#Need to consider boundary conditions for S,B,M etc.

A1 = 1*(np.matmul(D2_l,B)[:,np.newaxis]*d_p + dB_dl[:,np.newaxis]*d_l_p)
A2 = 1*S[:,np.newaxis] * d2_l
A3 = 1*-S[:,np.newaxis]*np.tan(Lat_rad[:,np.newaxis])*d_l
A4 = 1*-(S[:,np.newaxis]/np.cos(Lat_rad[:,np.newaxis])**2)*I
A5 = dS_dl[:,np.newaxis]*(d_l - np.tan(Lat_rad[:,np.newaxis])*I)

A6 = 1*dM_dp[:,np.newaxis]*(dM_dp[:,np.newaxis]*d_l - dM_dl[:,np.newaxis]*d_p) \
    + 1*M[:,np.newaxis]*(np.matmul(D2_p,M)[:,np.newaxis]*d_l + 1*dM_dp[:,np.newaxis]*d_l_p  \
                       - 1*np.matmul(D_l_p,M)[:,np.newaxis]*d_p - 1*dM_dl[:,np.newaxis]*d2_p)
        
A7 =  1*(-np.tan(Lat_rad[:,np.newaxis])*(1*dM_dp[:,np.newaxis]**2 * I + 1*M[:,np.newaxis]*np.matmul(D2_p,M)[:,np.newaxis]*I \
                                    + 1*M[:,np.newaxis]*dM_dp[:,np.newaxis]*d_p))


A = 1/(a**2) * (A1+ A2 + A3 + A4 + A5) + (eps[:,np.newaxis]/a)*(A6+A7)




####A lot of BS for the bbc#####
y = 1./(a*np.cos(Lat_rad))
x = np.cos(Lat_rad)/np.sin(Lat_rad)**2
W = -1./a*(d_l - np.tan(Lat_rad[:,np.newaxis])*I) # chi -> w

dw_dl = -( np.matmul(D_l,y)[:,np.newaxis]*(np.cos(Lat_rad[:,np.newaxis])*d_l - np.sin(Lat_rad[:,np.newaxis])*I) \
          + y[:,np.newaxis]*(np.cos(Lat_rad[:,np.newaxis])*d2_l - 2.*np.sin(Lat_rad[:,np.newaxis])*d_l \
                             -np.cos(Lat_rad[:,np.newaxis])*I) )
d2w_dl2 = -( np.matmul(D2_l,y)[:,np.newaxis]*(np.cos(Lat_rad[:,np.newaxis])*d_l - np.sin(Lat_rad[:,np.newaxis])*I) \
            +2*np.matmul(D_l,y)[:,np.newaxis]*(np.cos(Lat_rad[:,np.newaxis])*d2_l - 2.*np.sin(Lat_rad[:,np.newaxis])*d_l \
                             -np.cos(Lat_rad[:,np.newaxis])*I) \
                + y[:,np.newaxis]*(np.cos(Lat_rad[:,np.newaxis])*d3_l- 3.*np.sin(Lat_rad[:,np.newaxis])*d2_l \
                                   -3.*np.cos(Lat_rad[:,np.newaxis])*d_l + np.sin(Lat_rad[:,np.newaxis])*I) )

dw_dp = 0.25*omega**(-2)/(a*rho)*y[:,np.newaxis]*(np.matmul(D_l,x)[:,np.newaxis]*dw_dl + x[:,np.newaxis]*d2w_dl2)     
W_inv = np.linalg.inv(W)
K = np.matmul(W_inv,dw_dp) #dchi_dp


G1 = 1*(np.matmul(D2_l,B)[:,np.newaxis]*K +1*dB_dl[:,np.newaxis]*np.matmul(K,d_l))
G2 = 0
G3 = 0
G4 = 0
G5 = 0
G6 = -dM_dp[:,np.newaxis]*dM_dl[:,np.newaxis]*K + 1*M[:,np.newaxis]*dM_dp[:,np.newaxis]*np.matmul(d_l,K) \
    - M[:,np.newaxis]*np.matmul(D_l_p,M)[:,np.newaxis]*K + -1./del_p*M[:,np.newaxis]*dM_dl[:,np.newaxis]*K
G7 = 1*(-np.tan(Lat_rad)[:,np.newaxis]*M[:,np.newaxis]*dM_dp[:,np.newaxis]*K)
G = 1/(a**2) * (G1+ G2 + G3 + G4 + G5) + (eps/a)*(G6+G7)
G[0:-len(lat),:] = 0
 
############

H = 1/a*(dM_dl[:,np.newaxis]*d_p_b - dM_dp[:,np.newaxis]*(d_l - np.tan(Lat_rad[:,np.newaxis])*I))   


bf = 1/a*alpha_b*np.matmul(d_l,b_e) #Comment this line to switch forc.
Chi_E = np.linalg.solve(A+1.*G,bf)
q_eff = -np.matmul(H,Chi_E)
q_eff = rewind(q_eff)

# del_p = 100 * (press[1] - press[0])
# De_p = np.zeros((len(press),len(press)),dtype='d')
# for j in range(len(press)):
#     if j > 0 and j < len(press) -1 :
#         De_p[j,j-1] = -1.
#         De_p[j,j+1] = 1.
# De_p *= 1./(2*del_p)
# dp_q_eff = np.zeros(np.shape(q_eff),dtype='d')
# for l  in range(len(lat)):
#   dp_q_eff[:,l] = np.matmul(De_p,q_eff[:,l])


# S = rewind(S)
# f = 1/a* np.cos(lat_rad)*q_eff
# f_vertsum = np.sum(f,axis=0)
# f_proj_SAM = (f_vertsum[15] - f_vertsum[23])*86400.




Var = rewind(heat_rate_coarse)
#Var = heat_rate_RP
bound = max(abs(Var.min()),abs(Var.max()))
Prof, ax = plt.subplots()
col = ax.contourf(lat,press_coarse,Var,cmap= 'RdBu_r',vmin=-bound,vmax=bound)
cbar = Prof.colorbar(col)
plt.xlabel('Latitude (degrees)')
plt.ylabel('Pressure (hPa)')
plt.gca().invert_yaxis()
cbar.set_label('Buoyancy forcing ($s^{-3}$)')






Var = q_eff[2:,0:30]*86400/(6.37e6*np.cos(lat[np.newaxis,0:30]*np.pi/180))
#Var = q_eff[2:,0:30]
bound = max(abs(Var.min()),abs(Var.max()))
# #bound = abs(Var.min())
Prof, ax = plt.subplots()
col = ax.contourf(lat[0:30],press_coarse[2:],Var,cmap= 'RdBu_r',vmin=-bound,vmax=bound)
cbar = Prof.colorbar(col)
cont_pos = ax.contour(lat[0:30],press_coarse[2:],u_anom_pos[2:,0:30],colors=['black'])
#cont_neg = ax.contour(lat[0:30],press,u_anom_pos_dash[:,0:30],colors=['black'],linestyles=['dashed'])
plt.xlabel('Latitude',fontsize=13)
plt.ylabel('Pressure (hPa)',fontsize=13)
plt.xlim(-70,-20)
plt.gca().invert_yaxis()
#cbar.set_label('Effective torque ($ m^2 s^{-2} $)',fontsize=12)
#plt.savefig('Eliassen_response_SAM_heating.pdf')
#cbar.set_label('Effective Torque ($m^2 s^{-2} $)',fontsize=12)
cbar.set_label('Effective torque ($m^2 s^{-2}}$)',fontsize=12)

#plt.savefig('Eliassen_buoyancy_response_SAM_heating_ms_per_day.pdf')


######Tests######

###Test difference operators, small sensitivity to definition at edges###
# du_dp = np.matmul(d_p,unwind(u_clim))
# du_dp = rewind(du_dp)
# du_dl = np.matmul(d_l,unwind(u_clim))
# du_dl = rewind(du_dl)
# diff_l_u = np.diff(u_clim,axis=1)/np.diff(rewind(Lat_rad),axis=1)
# diff_p_u = np.diff(u_clim,axis=0)/np.diff(100*press[:,np.newaxis],axis=0)
# diff_test1 = (du_dl[:,0:-1] - diff_l_u)/diff_l_u
# diff_test2 = (du_dp[0:-1,:] - diff_p_u)/diff_p_u

# d_l_d_l = np.matmul(d_l,d_l)
# diff_test3 = d_l_d_l - d2_l

# d_p_d_p = np.matmul(d_p,d_p)
# diff_test4 = d_p_d_p - d2_p

# test_S = kappa*(g*(T_clim[0:-1,:] - T_star)/T_star + g)/(press[0:-1,np.newaxis]*100) - \
#    np.diff(rewind(B),axis=0)/2500 #Pa
# test1 = (rewind(S)[0:-1,:] - test_S)/test_S 
# test2 = (g*(T_clim - T_star)/T_star - rewind(B))/rewind(B)


###Test surface boundary condition ###
# x = np.cos(Lat_rad)/np.sin(Lat_rad)**2
# w = np.matmul(W,Chi_E)
# dw_dp_surf = np.matmul(d_p_b,w)
# dw_dl_surf = np.matmul(D_l,w)
# d2w_dl2_surf = np.matmul(D2_l,w)
# dchi_dp_surf= np.matmul(D_p,Chi_E)
# K_chi = np.matmul(K,Chi_E)
# surf_test_RHS= 4*omega**2*a**2*dw_dp_surf 
# surf_test_LHS = 1./(rho*np.cos(Lat_rad))*(np.matmul(D_l,x)*dw_dl_surf + x*d2w_dl2_surf)

# fig,ax = plt.subplots()
# ax.plot(lat[0:30],surf_test_RHS[-len(lat):-len(lat)+30],label='RHS')
# ax.plot(lat[0:30],surf_test_LHS[-len(lat):-len(lat)+30],label='LHS')
# ax.plot(lat[0:30],dchi_dp_surf[-len(lat):-len(lat)+30],label='RHS')
# ax.plot(lat[0:30],K_chi[-len(lat):-len(lat)+30],label='LHS')
#ax.legend()

#multiply by 3 because a=3 gives better signal
torque = 3*q_eff*86400/(6.37e6*np.cos(lat[np.newaxis,:]*np.pi/180))

ncfile = Dataset('Eliassen_response.nc',mode='w',format='NETCDF4')
ncfile.createDimension('lat', len(lat))
ncfile.createDimension('press', len(press))


lat_var = ncfile.createVariable('lat', np.float32, ('lat',))
press_var = ncfile.createVariable('press', np.float32, ('press',))
torq_var = ncfile.createVariable('torque', np.float32, ('press','lat'))
u_var = ncfile.createVariable('u_anom', np.float32, ('press','lat'))



lat_var[:] = lat
press_var[:] = press[::-1]
torq_var[:] = torque
u_var[:] = u_anom_pos

ncfile.close()


