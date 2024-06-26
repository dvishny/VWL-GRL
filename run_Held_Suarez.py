####################################
#
# Script to run MiMA on Cheyenne
# Last updated 24/05/20 by Nick Lutsko
# Last updated 23/07/20 by David Vishny; modified from gray bucket scheme
####################################
import os
import glob
import numpy as np
import shutil
import string
import sys
from isca import IscaCodeBase, DiagTable, Experiment, Namelist, GFDL_BASE

casename = sys.argv[1]
print('building case:'+casename)
builddir = '/glade/u/home/'+os.environ["USER"]+'/models/ISCA/Isca/Cases/'+casename
print('in:'+builddir)
#executable_name=casename+'.x'
executable_name='isca.x'
NCORES= int(sys.argv[2])
print('Using '+str(NCORES)+' cores')
RESOLUTION = 'T85', 40
todo = sys.argv[3]

base_dir = '/glade/u/home/'+os.environ["USER"]+'/models/ISCA/Isca/exp/'+casename+'/'

#Step 1: Soft link the original code. Point the python script to the location of the code (the bash environment variable GFDL_BASE is that directory)
if 'compile' in todo:
	shutil.rmtree(builddir, ignore_errors=True)
	os.makedirs(builddir)
	if not os.path.exists(builddir+'/Isca'):
		os.symlink(os.environ['GFDL_BASE'],builddir+'/Isca')
	#if not os.path.exists(builddir):
	#	os.makedirs(builddir)
	#if not os.path.exists(builddir+'/Isca'):
	#	os.symlink(os.environ['GFDL_BASE'],builddir+'/Isca')


objdir=builddir.replace('/','_')+'_Isca'
if 'clean' in todo:
	if os.path.exists(os.environ['GFDL_WORK']+'/codebase/'+objdir):
		print('clean up '+os.environ['GFDL_WORK']+'/codebase/'+objdir)
		shutil.rmtree(os.environ['GFDL_WORK']+'/codebase/'+objdir)
	if os.path.exists(os.environ['GFDL_WORK']+'/experiments/'+casename):
		print('clean up '+os.environ['GFDL_WORK']+'/experiments/'+casename)
		shutil.rmtree(os.environ['GFDL_WORK']+'/experiments/'+casename)
	if os.path.exists(builddir):
		print('clean up '+builddir)
		shutil.rmtree(builddir)
		
#cb = IscaCodeBase.from_directory(builddir+'/Isca',storedir=builddir)  # for saving obj files in Isca_caseroot/
cb = IscaCodeBase.from_directory(builddir+'/Isca')

if 'reinit' in todo:
	if os.path.exists(os.environ['GFDL_DATA']+casename):
		shutil.rmtree(os.environ['GFDL_DATA']+casename)

# srcmod:
if 'compile' in todo:
	if not os.path.exists(builddir+'/srcmod'):
		os.mkdir(builddir+'/srcmod')
	if os.path.exists(builddir+'/srcmod'):
		os.removedirs(builddir+'/srcmod')
	if os.path.exists('isca_srcmod'):
		shutil.copytree('./isca_srcmod', builddir+'/srcmod')
	extra_pathnames_tmp = os.listdir(builddir+'/srcmod/')
	extra_pathnames = [builddir+'/srcmod/' + s for s in extra_pathnames_tmp]
	print('Including the following modified CODEs:')
	print(extra_pathnames)
	for s in extra_pathnames_tmp[:]:
		ss = os.path.splitext(s)[0]
		for sss in glob.glob(os.environ["GFDL_WORK"]+"/codebase/"+objdir+"/build/isca/"+ss+"*"):
			os.remove(sss)

#Step 2. Provide the necessary inputs for the model to run:
##Should be land.nc?##
#inputfiles = [os.path.join(base_dir,'input/land1.nc'),os.path.join(GFDL_BASE,'input/rrtm_input_files/ozone_1990.nc')]
# copy/link qflux file to builddir
#qflux_file_name = 'merlis_schneider_30_16'
#if not os.path.exists(builddir+'/'+qflux_file_name+'.nc'):
#	shutil.copy(os.environ['GFDL_BASE']+'/ictp-isca-workshop-2018/experiments/earth_projects/project_5_qfluxes/'+qflux_file_name+'.nc',builddir+'/'+qflux_file_name+'.nc')
#qflux_file_name = 'off'


#Step 3. Define the diagnostics we want to be output from the model
diag = DiagTable()
diag.add_file('atmos_monthly', 30, 'days', time_units='days')
diag.add_file('atmos_daily', 1, 'days', time_units='days')
#Tell model which diagnostics to write to those files

diag.add_field('dynamics', 'ps', time_avg=True)
diag.add_field('dynamics', 'bk')
diag.add_field('dynamics', 'pk')
diag.add_field('dynamics', 'ucomp', time_avg=True)
diag.add_field('dynamics', 'vcomp', time_avg=True)
diag.add_field('dynamics', 'temp', time_avg=True)
diag.add_field('dynamics', 'vor', time_avg=True)
diag.add_field('dynamics', 'div', time_avg=True)
diag.add_field('hs_forcing', 'local_heating', time_avg=True)

#Step 4. Define the namelist options, which will get passed to the fortran to configure the model.
runlen=20
todoappend=''
if 'debug' in todo:
	runlen=1
	todoappend='debug'
namelist = Namelist({
	'main_nml':{
	 'days'   : runlen,
	 'hours'  : 0,
	 'minutes': 0,
	 'seconds': 0,
	 'dt_atmos':720,
	 'current_date' : [1,1,1,0,0,0],
	 'calendar' : 'thirty_day'
	},
#	'idealized_moist_phys_nml': {
#		'do_damping': True,
#		'turb':True,
#		'mixed_layer_bc':True,
#		'do_virtual' :False,
#		'do_simple': True,
#		'roughness_mom':2.e-4, #Ocean roughness lengths
#		'roughness_heat':2.e-4,
#		'roughness_moist':2.e-4,
#		'two_stream_gray':False, #Don't use grey radiation
#		'do_rrtm_radiation':True, #Do use RRTM radiation
#		'convection_scheme':'SIMPLE_BETTS_MILLER', #Use the simple betts-miller convection scheme
#		'land_option':'input', #Use land mask from input file
#		'land_file_name': 'INPUT/land.nc', #Tell model where to find input file
#		'bucket':True, #Run with the bucket model
#		'init_bucket_depth_land':1., #Set initial bucket depth over land
#		'max_bucket_depth_land':.15, #Set max bucket depth over land
#	},


#	'vert_turb_driver_nml': {
#		'do_mellor_yamada': False,     # default: True
#		'do_diffusivity': True,        # default: False
#		'do_simple': True,             # default: False
#		'constant_gust': 0.0,          # default: 1.0
#		'use_tau': False
#	},

#	'diffusivity_nml': {
#		'do_entrain':False,
#		'do_simple': True,
#	},

#	'surface_flux_nml': {
#		'use_virtual_temp': False,
#		'do_simple': True,
#		'old_dtaudv': True
#	},

	'atmosphere_nml': {
		'idealized_moist_model': False
	},

        #Use a large mixed-layer depth, and the Albedo of the CTRL case in Jucker & Gerber, 2017
#        'mixed_layer_nml': {
#		'tconst' : 285.,
#		'prescribe_initial_dist':True,
#		'evaporation':True,
#		'depth':20., #Mixed layer depth
#		'land_option':'input',    #Tell mixed layer to get land mask from input file
#		'land_h_capacity_prefactor': 0.1, #What factor to multiply mixed-layer depth by over land. 
#		'albedo_value': 0.25, #Ocean albedo value
#		'land_albedo_prefactor' : 1.3, #What factor to multiply ocean albedo by over land
#		'do_qflux' : False #Do not use prescribed qflux formula
#       },


#	'betts_miller_nml': {
#	   'rhbm': .7   ,
#	   'do_simp': False,
#	   'do_shallower': True,
#	},

#	'lscale_cond_nml': {
#		'do_simple':True,
#		'do_evap':True
#	},

#	'sat_vapor_pres_nml': {
#		'do_simple':True
#	},

#	'damping_driver_nml': {
#		'do_rayleigh': True,
#		'trayfric': -0.5,              # neg. value: time in *days*
#		'sponge_pbottom':  150., #Setting the lower pressure boundary for the model sponge layer in Pa.
#		'do_conserve_energy': True,
#	},

#	'rrtm_radiation_nml': {
#		'do_read_ozone':True,
#		'ozone_file':'ozone_1990',
#		'solr_cnst': 1360., #s set solar constant to 1360, rather than default of 1368.22
#		'dt_rad': 4320, #Use 4320 as RRTM radiation timestep
#	},

	# FMS Framework configuration
	'diag_manager_nml': {
		'mix_snapshot_average_fields': False  # time avg fields are labelled with time in middle of window
	},

	'fms_nml': {
		'domains_stack_size': 600000                        # default: 0
	},

	'fms_io_nml': {
		'threading_write': 'single',                         # default: multi
		'fileset_write': 'single',                           # default: multi
	},

	'spectral_dynamics_nml': {
        'damping_order'           : 4,                      # default: 2
        'water_correction_limit'  : 200.e2,                 # default: 0
        'reference_sea_level_press': 1.0e5,                  # default: 101325
        'valid_range_t'           : [100., 800.],           # default: (100, 500)
        'initial_sphum'           : 0.0,                  # default: 0
        'vert_coord_option'       : 'even_sigma',         # default: 'even_sigma'
        'scale_heights': 6.0,
        'exponent': 7.5,
        'surf_res': 0.5
    },
'hs_forcing_nml': {
        't_zero': 315.,    # temperature at reference pressure at equator (default 315K)
        't_strat': 200.,   # stratosphere temperature (default 200K)
        'delh': 60.,       # equator-pole temp gradient (default 60K)
        'delv': 10.,       # lapse rate (default 10K)
        'eps': 0.,         # stratospheric latitudinal variation (default 0K)
        'sigma_b': 0.7,    # boundary layer friction height (default p/ps = sigma = 0.7)

        # negative sign is a flag indicating that the units are days
        'ka':   -40.,      # Constant Newtonian cooling timescale (default 40 days)
        'ks':    -4.,      # Boundary layer dependent cooling timescale (default 4 days)
        'kf':   -1.,       # BL momentum frictional timescale (default 1 days)

        'do_conserve_energy':   True,  # convert dissipated momentum into heat (default True)
    }
})

#Step 5. Compile the fortran code
if 'compile' in todo:
	cb.compile(extra_pathnames=extra_pathnames, executable_name=executable_name)
	#if os.path.exists(builddir+'/run_isca.py'):
	#	os.remove(builddir+'/run_isca.py')
##DV changed run_isca.py to run_Held_Suarez.py
	shutil.copy(os.path.realpath(__file__),builddir+'/run_Held_Suarez.py')
	if os.path.exists(builddir+'/run.sub'):
		os.remove(builddir+'/run.sub')
	runsub_template=os.environ['GFDL_BASE']+'/src/extra/python/isca/templates/run.sub'
	with open(runsub_template, 'r') as file :
		filedata = file.read()
	filedata = filedata.replace('_CASENAME_', casename)
	filedata = filedata.replace('_NNODES_', str(1)) #Check this
	filedata = filedata.replace('_NCPUS_', str(NCORES))
	filedata = filedata.replace('_TODO_', 'run'+todoappend)
	with open(builddir+'/run.sub', 'w') as file :
		file.write(filedata)

#Step 6. Run the fortran code
if 'run' in todo:
	exp = Experiment(casename, codebase=cb)
	exp.clear_rundir()

	exp.diag_table = diag
	exp.namelist = namelist.copy()

#	if qflux_file_name!='off':
#		inputfiles.append(os.path.join(builddir,qflux_file_name+'.nc'))
#		exp.namelist['mixed_layer_nml']['load_qflux'] = True
#		exp.namelist['mixed_layer_nml']['time_varying_qflux'] = False
#		exp.namelist['mixed_layer_nml']['qflux_file_name'] = qflux_file_name
#	else:
#		exp.namelist['mixed_layer_nml']['load_qflux'] = False

#	exp.inputfiles = inputfiles
	exp.set_resolution(*RESOLUTION)
    ##Change nlutsko to dvishny?##
	exp.run(1, use_restart=True, num_cores=NCORES, restart_file='/glade/scratch/dvishny/isca_out/run_Held_Suarez/restarts/res0001.tar.gz')
	#for i in range(2,20):
	#	exp.run(i, num_cores=NCORES)
