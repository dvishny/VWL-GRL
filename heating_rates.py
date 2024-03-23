#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:15:54 2020

@author: davidvishny
"""


### This Python script is used to generate the  diabatic heating field that mimics ACRE in our simulations. ###

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import netCDF4 as nc4
import scipy.io





heating_SAM = np.zeros((14,14),dtype='d')
heating_SBAM = np.zeros((14,14),dtype='d')

heating_SAM[4,0:3] = [-0.18,-0.18,-0.12] 
heating_SAM[5,0:3] = [-0.18,-0.22,-0.12] 
heating_SAM[6,0:3] = [-0.22,-0.26,-0.20]
heating_SAM[7,0:3] = [-0.18,-0.22,-0.12]


heating_SAM[6,4:7] = [0.16,0.20,0.12]
heating_SAM[7,4:7] = [0.20,0.28,0.20]
heating_SAM[8,4:7] = [0.20,0.24,0.20]



#heating_SAM[4,13] = 0.20 #nonessential?
#heating_SAM[5,4:7] = [0.08,0.12,0.04]
heating_SAM[0,3] = 0.26
heating_SAM[0,6:10] = [-0.12,-0.12,-0.12,-0.12]


heating_SBAM[4,1:6] = [0.08,0.10,0.10,0.10,0.10]
heating_SBAM[5,0:6] = [0.12,0.12,0.14,0.16,0.14,0.12]
heating_SBAM[6,0:6] = [0.12,0.14,0.20,0.28,0.28,0.16]
heating_SBAM[7,0:7] = [0.12,0.16,0.24,0.32,0.32,0.18,0.08]
heating_SBAM[8,0:7] = [0.10,0.10,0.16,0.20,0.24,0.16,0.08]

heating_SBAM[3,9:12] = [-0.12,-0.20,-0.08]
heating_SBAM[4,9:11] = [-0.16,-0.16]
heating_SBAM[6,10:12] = [0.08,0.08]
heating_SBAM[7,9:11] = [0.08,0.14]
heating_SBAM[8,9:11] = [0.08,0.12]
heating_SBAM[9,10] = 0.12
heating_SBAM[10,10] = 0.08
heating_SBAM[0,3:10] = [-0.08,-0.12,-0.08,0,-0.12,-0.14,-0.16]
heating_SBAM[1,7:9] = [-0.12,-0.08]




heating_SAM_conv = np.zeros((40,64),dtype='d')
heating_SBAM_conv = np.zeros((40,64),dtype='d')


###################functions###################################################

def nearest_coord(val,arr):
    diff = abs(arr - val)
    diff_min = np.inf
    for i in range(len(arr)):
        if diff[i] < diff_min:
            diff_min = diff[i]
            coord = i
    return coord

def Cov(X,Y):
    X_m = np.mean(X)
    Y_m = np.mean(Y)
    Sum  = np.dot(X-X_m,Y-Y_m)
    return Sum/len(X)

def EOF1(X):
    col = len(X[0])
    M = np.zeros((col,col), dtype = 'd')
    for i in range(col):
        for j in range(col):
            Mi = X[:,i]
            Mj = X[:,j]
            M[i,j] = Cov(Mi,Mj)
    #print np.linalg.eigvals(M)/np.sum(np.linalg.eigvals(M))      
    u = -1.0 + 2* np.random.rand(col,1)[:,0]
    u = u/np.linalg.norm(u)
    u_temp = u
    u = np.matmul(M,u)
    while abs(np.dot(u,u_temp)/(np.linalg.norm(u)*np.linalg.norm(u_temp)) - 1.) > 1E-5:
        u_temp = u
        u = np.matmul(M,u)
    return u/np.linalg.norm(u)

 ##############################################################################   
    

### Map coarse heating field onto model dimensions###
lats = np.zeros(14,dtype = 'd')
heights = np.zeros(14,dtype = 'd')
merid_T = np.zeros(14,dtype = 'd') #used to get sigma from height - 0.6K/deg
sig = np.zeros((14,14),dtype = 'd')

exp_data = nc4.Dataset('era_i_temp.nc','r',format='NETCDF4')
full_lats = exp_data['lat'][:]
full_lons = exp_data['lon'][:]
lat_edges = exp_data['latb'][:]
lon_edges = exp_data['lonb'][:]
exp_data.close()

AM_NC = 'ctrl_12.nc'
daily_data = nc4.Dataset(AM_NC)
u_zon_av = daily_data.variables['u_zon_av'][:]
temp_zon_av = daily_data.variables['temp_zon_av'][:]
lat_HS = daily_data.variables['lat'][:]
press_HS = daily_data.variables['press'][:]



for j in range(7):
    
  AM_NC = 'ctrl_' + str(2*j+3) + str(2*j+4) + '.nc'
  daily_data_2 = nc4.Dataset(AM_NC)
  u_zon_av_2 = daily_data_2.variables['u_zon_av'][:]
  temp_zon_av_2 = daily_data_2.variables['temp_zon_av'][:]
  u_zon_av = np.append(u_zon_av,u_zon_av_2,axis=0)
  temp_zon_av = np.append(temp_zon_av,temp_zon_av_2,axis=0)

u_zon_av_vertsum = np.sum(u_zon_av,axis=1)
u_wtd = u_zon_av_vertsum * np.sqrt((np.cos(lat_HS*np.pi/180)))[np.newaxis,:]
SAM_EOF = EOF1(u_wtd)
SAM_PC1 = np.zeros(len(u_wtd),dtype='d')
if SAM_EOF[14] < 0:
    SAM_EOF= -SAM_EOF
for t in range(len(u_wtd)):
  SAM_PC1[t] = np.dot(u_zon_av_vertsum[t,:],SAM_EOF)

SAM_PC1 = SAM_PC1 - np.mean(SAM_PC1)
SAM_PC1 = SAM_PC1/np.std(SAM_PC1)
    
# SAM_ind_exp =   u_zon_av_vertsum[:,14] - u_zon_av_vertsum[:,29] #positive when poleward
# SAM_ind_exp = SAM_ind_exp - np.mean(SAM_ind_exp)
# SAM_PC1 = SAM_ind_exp/np.std(SAM_ind_exp)

temp_zon_av_dot_ind = np.zeros((37,32),dtype='d')
u_zon_av_dot_ind = np.zeros((37,32),dtype='d')

temp_zon_av = temp_zon_av - np.mean(temp_zon_av,axis=0)[np.newaxis,:,:]
u_zon_av = u_zon_av - np.mean(u_zon_av,axis=0)[np.newaxis,:,:]
for k in range(32): 
      for j in range(37):
          temp_zon_av_dot_ind[j,k] = \
          np.dot(temp_zon_av[:,k,j],SAM_PC1)/np.linalg.norm(SAM_PC1)**2.
          u_zon_av_dot_ind[j,k] = \
          np.dot(u_zon_av[:,k,j],SAM_PC1)/np.linalg.norm(SAM_PC1)**2.
          



for i in range(14):
    lats[i] = -80. + i*80./13.
    heights[i] = 1000.*i*15./13.
    merid_T[i] = 303 + 0.6*lats[i]
#    

   



#lat_thick = 3
#for k in range(14):
#    for j in range(14):
#        sig[k,j] = np.exp(-9.81*heights[k]/(287.05*merid_T[j]))
#sig_thick = abs(0.5*np.diff(sig,axis=0)/0.025)
#for k in range(14):
#    for j in range(14):
#        sig_ind = nearest_coord(sig[k,j],model_sig)
#        lat_ind = nearest_coord(lats[j]+9.,full_lats) #heating shifted equatorward
#                                                        #to accommodate jet bias
#        if k < 13:
#            sthick = int(sig_thick[k,j]) + 1
#        else:
#            sthick = 0
#        lb_sig =  max(sig_ind-sthick,0)
#        ub_sig = min(sig_ind+sthick,29)
#        lb_lat = max(lat_ind-lat_thick,0)
#        ub_lat = min(lat_ind+lat_thick,63)
#        heating_SAM_conv[lb_sig:ub_sig,lb_lat:ub_lat] = heating_SAM[k,j]
#        heating_SBAM_conv[lb_sig:ub_sig,lb_lat:ub_lat] = heating_SBAM[k,j]
#
#const_heating = np.zeros((40,64,128),dtype='d')
#crit_ind_1 = nearest_coord(-51.1,full_lats)
#crit_ind_2 = nearest_coord(-30.1,full_lats)
#cool_val = -0.4/86400.
#hot_val = 1.2/86400.
#
#for k in range(40):
#    for j in range(64):
#        for i in range(128):
#            Cool = cool_val
#            Warm = hot_val
#            if j < crit_ind_1:
#               const_heating[k,j,i] = Cool
#            if j > crit_ind_1 and j < crit_ind_2:
#                const_heating[k,j,i] = Warm
#            if j > crit_ind_2 and j < 32:
#                const_heating[k,j,i] = Cool
#            
#print crit_ind_1 + 32 - crit_ind_2
#print crit_ind_2 - crit_ind_1

#Read Casey's data
# mat = scipy.io.loadmat('LW_ACRE_regression_SAM.mat') 
# era_lats = mat['lat'] #shape=(1,34)
# era_lat_edges = mat['lat_edges'] #(1,35)
# p_edges = mat['p_edges'] #(1,40)
# p_levs = mat['p'] #(1,39)
# regs = mat['dLW_ACRE_dSAM']/86400. #(34,39) #convert to K/s
# regs_shift = np.zeros((34,39),dtype='d')
# for l in range(34):
#         if l >= 4:
#             regs_shift[l,:] = regs[l-4,:]
#         else:
#             regs_shift[l,:] = np.zeros(39,dtype='d')


# era_edges_full = np.zeros(73,dtype='d')
# for l in range(73):
#     era_edges_full[l] = -90. + 2.5*l
# era_lats_full = np.zeros(72,dtype='d')
# for l in range(72):
#     era_lats_full[l] = -88.75 + 2.5*l
# regs_full = np.zeros((39,72,128),dtype='d')
# for l in range(128):
#     regs_full[:,3:37,l] = np.transpose(regs_shift)
    

SAM_heat_data= nc4.Dataset('CW_SAM_heating.nc','r',format='NETCDF4')
heat_field = SAM_heat_data.variables['local_heating'][:]
full_lat = SAM_heat_data.variables['lat'][:]
full_press = SAM_heat_data.variables['pfull'][:]


#SAM_heat_data= nc4.Dataset('CW_SAM_heating.nc','w',format='NETCDF4')
#SAM_heat_data.createDimension('time', 1)
#SAM_heat_data.createDimension('lat', 72)
#SAM_heat_data.createDimension('latb', 73)
#SAM_heat_data.createDimension('lon', 128)
#SAM_heat_data.createDimension('lonb', 129)
#SAM_heat_data.createDimension('pfull', 39)
#SAM_heating_rates = SAM_heat_data.createVariable('local_heating', np.float32, \
#                                                 ('time','pfull','lat','lon',))
#SAM_heating_rates.units = 'K/s'
#full_lats_SAM = SAM_heat_data.createVariable('lat', np.float32, ('lat'))
#lat_edges_SAM = SAM_heat_data.createVariable('latb', np.float32, ('latb'))
#full_lats_SAM.units = 'degrees'
#lat_edges_SAM.units = 'degrees'
#full_lons_SAM = SAM_heat_data.createVariable('lon', np.float32, ('lon'))
#lon_edges_SAM = SAM_heat_data.createVariable('lonb', np.float32, ('lonb'))
#lon_edges_SAM.units ='degrees'
#full_lons_SAM.units = 'degrees'
#full_p_SAM = SAM_heat_data.createVariable('pfull', np.float32, ('pfull'))
#full_p_SAM.units = 'hPa'
#Time = SAM_heat_data.createVariable('time', np.float32, ('time'))
#Time.units = 'days since 0000-00-00 00:00:00'
#full_lats_SAM[:] = era_lats_full
#full_lons_SAM[:] = full_lons
#lat_edges_SAM[:] = era_edges_full
#lon_edges_SAM[:] = lon_edges
#full_p_SAM[:] = p_levs
#SAM_heating_rates[0,:,:,:] = regs_full
#SAM_heat_data.close()


# vert_prof = 0.3 * np.exp(-(0.45*(p_levs-865.)/25.)**2 )
# fig,ax = plt.subplots()
# ax.plot(vert_prof[0],p_levs[0])
# plt.gca().invert_yaxis()
# SAM_heating_rates[:,:,:,:] = 0
# SAM_heating_rates[0,:,9:16,:] = vert_prof[0,:,np.newaxis,np.newaxis]/86400.
# SAM_heating_rates[0,:,:,:] *= np.sin(full_lons*2.*np.pi/180)[np.newaxis,np.newaxis,:]
# SAM_heat_data.close()







#heating_SAM_full = np.zeros((40,64,128),dtype='d')
#heating_SBAM_full = np.zeros((40,64,128),dtype='d')

#SAM_heating_tot = np.sum(heating_SAM_full[0,0:64,:])
#pos_tot = 0
#for j in range(64):
#    for k in range(40):
#        if heating_SAM_full[0,j,k] > 0:
#            pos_tot += heating_SAM_full[0,j,k]
#amp_fac = 1 - SAM_heating_tot/pos_tot    
    
    

#Prof, ax = plt.subplots()
#col = ax.pcolor(era_lats[0], p_levs[0], regs_full[:,:,1], cmap = my_cmap)
#cbar = Prof.colorbar(col)
#cbar.set_label('Heating rate (K/day)')
#plt.xlabel('Latitude (deg)')
#plt.ylabel('Pressure (hPa)')
#plt.gca().invert_yaxis()
#plt.savefig('ACRE_SAM_model_constant.pdf')








#SAM_heat_data.close()
#SAM_heat_group = SAM_heat_data.createGroup('Heating_field')
#SAM_heat_data.createDimension('lat', 127)
#SAM_heat_data.createDimension('latb', 128)
#SAM_heat_data.createDimension('lon', 255)
#SAM_heat_data.createDimension('lonb', 256)
#SAM_heat_data.createDimension('pfull', 40)
#SAM_heating_rates = SAM_heat_data.createVariable('local_heating', np.float32, \
#                                                 ('pfull','lat','lon',))
#SAM_heating_rates.units = 'K/s'
#full_lats_SAM = SAM_heat_data.createVariable('lat', np.float32, ('lat'))
#lat_edges_SAM = SAM_heat_data.createVariable('latb', np.float32, ('latb'))
#full_lats_SAM.units = 'degrees'
#lat_edges_SAM.units = 'degrees'
#full_lons_SAM = SAM_heat_data.createVariable('lon', np.float32, ('lon'))
#lon_edges_SAM = SAM_heat_data.createVariable('lonb', np.float32, ('lonb'))
#lon_edges_SAM.units ='degrees'
#full_lons_SAM.units = 'degrees'
#
#full_p_SAM = SAM_heat_data.createVariable('pfull', np.float32, ('pfull'))
#full_p_SAM.units = 'hPa'

#full_lats_SAM[:] = full_lats
#full_lons_SAM[:] = full_lons
#lat_edges_SAM[:] = lat_edges
#lon_edges_SAM[:] = lon_edges
#full_p_SAM[:] = model_sig*1000.
#SAM_heating_rates[:] = heating_SAM_full
#print SAM_heat_data.variables['latb'][:]
#print SAM_heat_data:
#SAM_heat_data.close()

#SBAM_heat_data= nc4.Dataset('SBAM_heating.nc','w',format='NETCDF4')
##SBAM_heat_group = SBAM_heat_data.createGroup('Heating_field')
#SBAM_heat_data.createDimension('lat', 127)
#SBAM_heat_data.createDimension('latb', 128)
#SBAM_heat_data.createDimension('lon', 255)
#SBAM_heat_data.createDimension('lonb', 256)
#SBAM_heat_data.createDimension('pfull', 40)
#SBAM_heating_rates = SBAM_heat_data.createVariable('local_heating', np.float32, \
#                                                 ('pfull','lat','lon',))
#SBAM_heating_rates.units = 'K/s'
#full_lats_SBAM = SBAM_heat_data.createVariable('lat', np.float32, ('lat'))
#full_lats_SBAM.units = 'degrees'
#lat_edges_SBAM = SBAM_heat_data.createVariable('latb', np.float32, ('latb'))
#lat_edges_SBAM.units = 'degrees'
#full_lons_SBAM = SBAM_heat_data.createVariable('lon', np.float32, ('lon'))
#full_lons_SBAM.units = 'degrees'
#lon_edges_SBAM = SBAM_heat_data.createVariable('lonb', np.float32, ('lonb'))
#lon_edges_SBAM.units ='degrees'
#full_p_SBAM = SBAM_heat_data.createVariable('pfull', np.float32, ('pfull'))
#full_p_SBAM.units = 'hPa'
#
#full_lats_SBAM[:] = full_lats
#full_lons_SBAM[:] = full_lons
#lat_edges_SBAM[:] = lat_edges
#lon_edges_SBAM[:] = lon_edges
#full_p_SBAM[:] = model_sig*1000.
#SBAM_heating_rates[:] = heating_SBAM_full

#test = SBAM_heat_data.variables['latb']
#print test
#print SBAM_heat_data.variables['latb'][:]
#print SBAM_heat_data
#SBAM_heat_data.close()
#
#
#

# SAM_heat_data = nc4.Dataset('SAM_heating.nc','r',format='NETCDF4')
# heat_field = SAM_heat_data.variables['local_heating'][0,:,:]
# old_lat = SAM_heat_data.variables['lat'][:]
# old_press = SAM_heat_data.variables['pfull'][:]

Max = np.max(temp_zon_av_dot_ind)
Min = np.min(temp_zon_av_dot_ind)
step = (Max-Min)/8.


# Prof, ax = plt.subplots()
# col = ax.pcolor(era_lats_full[0:36], p_levs[0], 86400.*regs_full[:,0:36], cmap = 'RdBu_r', vmin=-0.22, vmax=0.22)
# ax.contour(lat_model,press_model, np.transpose(temp_zon_av_dot_ind), colors='black', \
#            levels = np.arange(Min,Max+step,step) )
# cbar = Prof.colorbar(col)
# cbar.set_label('Heating rate (K/day)', fontsize=13)
# plt.xlabel('Latitude',fontsize=13)
# plt.ylabel('Pressure (hPa)',fontsize=13)
# plt.ylim(200,1000)
# plt.xlim(-70,-20)
# plt.gca().invert_yaxis()
#plt.savefig('SAM_anom_heating_model.pdf')

fig = plt.figure( figsize = (15, 14) )
fig.subplots_adjust(right = 0.96, top = 0.94, left = 0.15, bottom = 0.08, hspace = 0.3, wspace = 0.3) #Adjust relative spacing?

ax = plt.subplot(2, 1, 2)
Var = 86400.*heat_field[0,6:,7:29,0]
#bound = max(np.abs(Var.min()), np.abs(Var.max()))
bound = 0.3
#Prof, ax = plt.subplots()
col = ax.contourf(full_lat[7:29], full_press[6:], Var, cmap = 'RdBu_r', vmin=-bound, vmax=bound)
ax.contour(lat_HS,press_HS, np.transpose(temp_zon_av_dot_ind), colors='black', \
            levels = np.arange(Min,Max+step,step) )
cbar_1 = plt.colorbar(col)
cbar_1.set_label('Heating rate (K $day^{-1}$)', fontsize=17)
plt.xlabel('Latitude (degrees)',fontsize=17)
plt.ylabel('Pressure (hPa)',fontsize=17)
plt.ylim(200,1000)
plt.xlim(-70,-20)
col.set_clim(vmin=-0.3, vmax=0.3)
plt.gca().invert_yaxis()
plt.title(label='(b)',loc='left',fontsize=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = plt.subplot(2, 1, 1)
Var = np.transpose(u_zon_av_dot_ind)
#bound = max(np.abs(Var.min()), np.abs(Var.max()))
bound = 6
#Prof, ax = plt.subplots()

#NICK:
#ci_levels = np.arange(-6., 6.1, 0.1) 
#col = ax.contourf(lat_HS, press_HS, Var,ci_levels, cmap = 'RdBu_r', vmin=-bound, vmax=bound)
#or:
#ci_levs = [-6., -5., -4., -3., -2., -1., 1., ...]
#ax.contourf(lon, lat, temp, ci_levs)

col = ax.contourf(lat_HS, press_HS, Var, levs = np.arange(-6., 6.1, 0.1), cmap = 'RdBu_r', vmin=-bound, vmax=bound)
cbar_2 = plt.colorbar(col)
cbar_2.set_label('Zonal wind (m $s^{-1}$)', fontsize=17)
#plt.xlabel('Latitude (degrees)',fontsize=15)
plt.ylabel('Pressure (hPa)',fontsize=17)
plt.ylim(200,1000)
plt.xlim(-70,-20)
plt.gca().invert_yaxis()
plt.title(label='(a)', loc='left',fontsize=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('ACRE_Paper_fig_1.png')


