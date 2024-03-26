!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!                                                                   !!
!!                   GNU General Public License                      !!
!!                                                                   !!
!! This file is part of the Flexible Modeling System (FMS).          !!
!!                                                                   !!
!! FMS is free software; you can redistribute it and/or modify it    !!
!! under the terms of the GNU General Public License as published by !!
!! the Free Software Foundation, either version 3 of the License, or !!
!! (at your option) any later version.                               !!
!!                                                                   !!
!! FMS is distributed in the hope that it will be useful,            !!
!! but WITHOUT ANY WARRANTY; without even the implied warranty of    !!
!! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the      !!
!! GNU General Public License for more details.                      !!
!!                                                                   !!
!! You should have received a copy of the GNU General Public License !!
!! along with FMS. if not, see: http://www.gnu.org/licenses/gpl.txt  !!
!!                                                                   !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module hs_forcing_mod

!-----------------------------------------------------------------------

#ifdef INTERNAL_FILE_NML
use mpp_mod, only: input_nml_file, mpp_sum
#else
use fms_mod, only: open_namelist_file
#endif

use     constants_mod, only: KAPPA, CP_AIR, GRAV, PI, SECONDS_PER_DAY, &
                            orbital_period, stefan, solar_const

use           fms_mod, only: error_mesg, FATAL, file_exist,       &
                             check_nml_error,                     &
                             mpp_pe, mpp_root_pe, close_file,     &
                             write_version_number, stdlog,        &
                             uppercase, read_data, write_data, set_domain

use  time_manager_mod, only: time_type, get_time

use  diag_manager_mod, only: register_diag_field, send_data

use  field_manager_mod, only: MODEL_ATMOS, parse
use tracer_manager_mod, only: query_method, get_number_tracers
use   interpolator_mod, only: interpolate_type, interpolator_init, &
                              interpolator, interpolator_end, &
                              CONSTANT, INTERP_WEIGHTED_P

use      astronomy_mod, only: diurnal_exoplanet, astronomy_init, obliq, ecc
use     transforms_mod, only: grid_domain, get_grid_domain


implicit none
private

!-----------------------------------------------------------------------
!---------- interfaces ------------

   public :: hs_forcing, hs_forcing_init, hs_forcing_end

   type(interpolate_type),save         ::  heating_source_interp
   type(interpolate_type),save         ::  u_interp
   type(interpolate_type),save         ::  v_interp
   type(interpolate_type),save         ::  temp_interp

!-----------------------------------------------------------------------
!-------------------- namelist -----------------------------------------

   logical :: no_forcing = .false.

   real :: t_zero=315., t_strat=200., delh=60., delv=10., eps=0., sigma_b=0.7
   real :: P00 = 1.e5, p_trop  = 1.e4, alpha = 2./7

   real :: ka = -40., ks =  -4., kf = -1. ! negative sign is a flag indicating that the units are days

   logical :: do_conserve_energy = .true.

   real :: trflux = 1.e-5   !  surface flux for optional tracer
   real :: trsink = -4.     !  damping time for tracer

   character(len=256) :: local_heating_option='from_file' ! Valid options are 'from_file' and 'Isidoro'. Local heating not done otherwise.
   character(len=256) :: SAM_heating_file='local_heating'   ! Name of file relative to $work/INPUT  Used only when local_heating_option='from_file'
    character(len=256) :: SBAM_heating_file='SBAM_heating'   ! Name of file relative to $work/INPUT  Used only when local_heating_option='from_file'
   real :: local_heating_srfamp=0.0              ! Degrees per day.   Used only when local_heating_option='Isidoro'
   real :: local_heating_xwidth=10.              ! degrees longitude  Used only when local_heating_option='Isidoro'
   real :: local_heating_ywidth=10.              ! degrees latitude   Used only when local_heating_option='Isidoro'
   real :: local_heating_xcenter=180.            ! degrees longitude  Used only when local_heating_option='Isidoro'
   real :: local_heating_ycenter=45.             ! degrees latitude   Used only when local_heating_option='Isidoro'
   real :: local_heating_vert_decay=1.e4         ! pascals            Used only when local_heating_option='Isidoro'

   logical :: relax_to_specified_wind = .false.
   character(len=256) :: u_wind_file='u', v_wind_file='v' ! Name of files relative to $work/INPUT  Used only when relax_to_specified_wind=.true.

   character(len=256) :: equilibrium_t_option = 'Held_Suarez'  ! Valid options are 'Held_Suarez', 'from_file', 'exoplanet'
   character(len=256) :: equilibrium_t_file='temp'  ! Name of file relative to $work/INPUT  Used only when equilibrium_t_option='from_file'
   character(len=256) :: stratosphere_t_option = 'extend_tp'

   real :: peri_time=0.25, smaxis=1.5e6, albedo=0.3
   real :: lapse=6.5, h_a=2, tau_s=5
   real :: heat_capacity=4.2e6      ! equivalent to a 1m mixed layer water ocean
   real :: ml_depth=1               ! depth for heat capacity calculation
   real :: spinup_time=10800.     ! number of days to spin up heat capacity for - req. multiple of orbital_period
   real :: a=1.                   ! amplification factor for SAM
   real :: b=0.                   ! amplification factor for SBAM



!-----------------------------------------------------------------------

   namelist /hs_forcing_nml/  no_forcing, t_zero, t_strat, delh, delv, eps,  &
                              sigma_b, ka, ks, kf, do_conserve_energy,       &
                              trflux, trsink, local_heating_srfamp,          &
                              local_heating_xwidth,  local_heating_ywidth,   &
                              local_heating_xcenter, local_heating_ycenter,  &
                              local_heating_vert_decay, local_heating_option,&
                              SAM_heating_file, SBAM_heating_file,&
                              relax_to_specified_wind,   &
                              u_wind_file, v_wind_file, equilibrium_t_option,&
                              equilibrium_t_file, p_trop, alpha, peri_time, smaxis, albedo, &
                              lapse, h_a, tau_s, orbital_period,         &
                              heat_capacity, ml_depth, spinup_time, stratosphere_t_option, P00,a,b

!-----------------------------------------------------------------------

   character(len=128) :: version='$Id: hs_forcing.F90,v 19.0 2012/01/06 20:10:01 fms Exp $'
   character(len=128) :: tagname='$Name: siena_201211 $'

   real :: tka, tks, vkf
   real :: trdamp, twopi

   real, allocatable, dimension(:,:) :: tg_prev

   integer :: id_teq, id_h_trop, id_tdt, id_udt, id_vdt, id_tdt_diss, id_diss_heat, id_local_heating, id_newtonian_damping
   integer :: id_SAM_ind, id_SBAM_ind, id_eke, id_uhs, id_vhs
   real    :: missing_value = -1.e10
   real    :: xwidth, ywidth, xcenter, ycenter ! namelist values converted from degrees to radians
   real    :: srfamp ! local_heating_srfamp converted from deg/day to deg/sec
   character(len=14) :: mod_name = 'hs_forcing'

   logical :: module_is_initialized = .false.

!-----------------------------------------------------------------------

contains

!#######################################################################

 subroutine hs_forcing ( is, ie, js, je, dt, Time, lon, lat, p_half, p_full, &
                         u, v, t, r, um, vm, tm, rm, udt, vdt, tdt, rdt, zfull,&
                          mask, kbot )

!-----------------------------------------------------------------------
   integer, intent(in)                        :: is, ie, js, je
      real, intent(in)                        :: dt
 type(time_type), intent(in)                  :: Time
      real, intent(in),    dimension(:,:)     :: lon, lat
      real, intent(in),    dimension(:,:,:)   :: p_half, p_full
      real, intent(in),    dimension(:,:,:)   :: u, v, t, um, vm, tm, zfull
      real, intent(in),    dimension(:,:,:,:) :: r, rm
      real, intent(inout), dimension(:,:,:)   :: udt, vdt, tdt
      real, intent(inout), dimension(:,:,:,:) :: rdt
      !real, intent(in),    dimension(:,:,:) :: ufull, vfull
      real, intent(in),    dimension(:,:,:), optional :: mask
   integer, intent(in),    dimension(:,:)  , optional :: kbot
!-----------------------------------------------------------------------
   real, dimension(size(t,1),size(t,2))           :: ps, diss_heat, h_trop
   real, dimension(size(t,1),size(t,2),size(t,3)) :: ttnd, utnd, vtnd, teq, pmass
   real, dimension(size(r,1),size(r,2),size(r,3)) :: rst, rtnd
   integer :: i, j, k, kb, n, num_tracers
   logical :: used
   real    :: flux, sink, value
   character(len=128) :: scheme, params

!-----------------------------------------------------------------------
     if (no_forcing) return

     if (.not.module_is_initialized) call error_mesg ('hs_forcing','hs_forcing_init has not been called', FATAL)

!-----------------------------------------------------------------------
!     surface pressure

     if (present(kbot)) then
         do j=1,size(p_half,2)
         do i=1,size(p_half,1)
            kb = kbot(i,j)
            ps(i,j) = p_half(i,j,kb+1)
         enddo
         enddo
     else
            ps(:,:) = p_half(:,:,size(p_half,3))
     endif

!-----------------------------------------------------------------------
!     rayleigh damping of wind components near the surface

      call rayleigh_damping ( Time, ps, p_full, p_half, u, v, utnd, vtnd, mask=mask )

      if (do_conserve_energy) then
         ttnd = -((um+.5*utnd*dt)*utnd + (vm+.5*vtnd*dt)*vtnd)/CP_AIR
         tdt = tdt + ttnd
!         if (id_tdt_diss > 0) used = send_data ( id_tdt_diss, ttnd, Time, is, js)
         if (id_tdt_diss > 0) used = send_data ( id_tdt_diss, ttnd, Time) !st 2013 FMS seems to have paralelisation issues when called with ..., Time, is, js)
       ! vertical integral of ke dissipation
         if ( id_diss_heat > 0 ) then
          do k = 1, size(t,3)
            pmass(:,:,k) = p_half(:,:,k+1)-p_half(:,:,k)
          enddo
          diss_heat = CP_AIR/GRAV * sum( ttnd*pmass, 3)
!          used = send_data ( id_diss_heat, diss_heat, Time, is, js)
          used = send_data ( id_diss_heat, diss_heat, Time)
         endif
      endif

      udt = udt + utnd
      vdt = vdt + vtnd

!     if (id_udt > 0) used = send_data ( id_udt, utnd, Time, is, js)
      if (id_udt > 0) used = send_data ( id_udt, utnd, Time)
!      if (id_vdt > 0) used = send_data ( id_vdt, vtnd, Time, is, js)
      if (id_vdt > 0) used = send_data ( id_vdt, vtnd, Time)

!-----------------------------------------------------------------------
!     thermal forcing for held & suarez (1994) benchmark calculation
      if (trim(equilibrium_t_option) == 'top_down') then
         call top_down_newtonian_damping(Time, lat, ps, p_full, p_half, t, ttnd, teq, dt, h_trop, zfull, mask )
      else
         call newtonian_damping ( Time, lat, lon, ps, p_full, p_half, t, ttnd, teq, mask )
      endif
      tdt = tdt + ttnd
!      if (id_newtonian_damping > 0) used = send_data(id_newtonian_damping, ttnd, Time, is, js)
      if (id_newtonian_damping > 0) used = send_data(id_newtonian_damping, ttnd, Time)

      if(trim(local_heating_option) /= '') then
        call local_heating ( Time, is, js, lon, lat, ps, p_full, p_half, u, v, a, b, id_SAM_ind, id_SBAM_ind, id_eke, id_uhs, id_vhs, ttnd )
        !print *, 'dims tdt_outsub=', size(tdt,1), size(tdt,2), size(tdt,3) !DV 10/10/20
        !print *, 'dims ttnd_outsub=', size(ttnd,1), size(ttnd,2), size(ttnd,3)
        tdt = tdt + ttnd
!        if (id_local_heating > 0) used = send_data ( id_local_heating, ttnd, Time, is, js)
        if (id_local_heating > 0) used = send_data ( id_local_heating, ttnd, Time)
        
        !print *, 'id_local_heating=', id_local_heating
        !print *, 'local heating used=', used
        !print *, 'send data local heating=', send_data ( id_local_heating, ttnd, Time)

      endif

!      if (id_tdt > 0) used = send_data ( id_tdt, tdt, Time, is, js)
      if (id_tdt > 0) used = send_data ( id_tdt, tdt, Time)
!      if (id_teq > 0) used = send_data ( id_teq, teq, Time, is, js)
      if (id_teq > 0) used = send_data ( id_teq, teq, Time)
      if (id_h_trop > 0) used = send_data ( id_h_trop, h_trop, Time) !, is, js)
      !if (id_SAM_ind > 0) used = send_data ( id_SAM_ind, SAM_ind_norm, Time)
      !if (id_SBAM_ind > 0) used = send_data ( id_SBAM_ind, SBAM_ind_norm, Time)
!-----------------------------------------------------------------------
!     -------- tracers -------

      call get_number_tracers(MODEL_ATMOS, num_tracers=num_tracers)
      if(num_tracers == size(rdt,4)) then
        do n = 1, size(rdt,4)
           flux = trflux
           sink = trsink
           if (query_method('tracer_sms', MODEL_ATMOS, n, scheme, params)) then
               if (uppercase(trim(scheme)) == 'NONE') cycle
               if (uppercase(trim(scheme)) == 'OFF') then
                 flux = 0.; sink = 0.
               else
                 if (parse(params,'flux',value) == 1) flux = value
                 if (parse(params,'sink',value) == 1) sink = value
               endif
           endif
           rst = rm(:,:,:,n) + dt*rdt(:,:,:,n)
           call tracer_source_sink ( flux, sink, p_half, rst, rtnd, kbot )
           rdt(:,:,:,n) = rdt(:,:,:,n) + rtnd
        enddo
      else
        call error_mesg('hs_forcing','size(rdt,4) not equal to num_tracers', FATAL)
      endif

!-----------------------------------------------------------------------

 end subroutine hs_forcing

!#######################################################################

 subroutine hs_forcing_init ( axes, Time, lonb, latb, lat )

!-----------------------------------------------------------------------
!
!           routine for initializing the model with an
!              initial condition at rest (u & v = 0)
!
!-----------------------------------------------------------------------

           integer, intent(in) :: axes(4)
   type(time_type), intent(in) :: Time
   real, intent(in), dimension(:,:) :: lat
   real, intent(in), optional, dimension(:,:) :: lonb, latb


!-----------------------------------------------------------------------
   integer  unit, io, ierr

   real, dimension(size(lat,1),size(lat,2)) :: s, t_radbal, t_trop, h_trop, t_surf, hour_angle, tg
   integer :: spin_count, seconds, days, dt_integer
   real :: dec, orb_dist, step_days
   integer :: is, ie, js, je


   call get_grid_domain(is, ie, js, je)
   call set_domain(grid_domain)
   call get_time(Time, seconds, days)
   dt_integer = 86400*days + seconds

   allocate(tg_prev       (size(lonb,1)-1, size(latb,2)-1))

!     ----- read namelist -----

#ifdef INTERNAL_FILE_NML
     read (input_nml_file, nml=hs_forcing_nml, iostat=io)
     ierr = check_nml_error(io, 'hs_forcing_nml')
#else
      if (file_exist('input.nml')) then
         unit = open_namelist_file ( )
         ierr=1; do while (ierr /= 0)
            read  (unit, nml=hs_forcing_nml, iostat=io, end=10)
            ierr = check_nml_error (io, 'hs_forcing_nml')
         enddo
  10     call close_file (unit)
      endif
#endif

!     ----- write version info and namelist to log file -----

      call write_version_number (version,tagname)
      if (mpp_pe() == mpp_root_pe()) write (stdlog(),nml=hs_forcing_nml)

      if (no_forcing) return

      twopi = 2*PI

   ! ---- spin-up simple heat capacity used in top-down code ----

  if (trim(equilibrium_t_option) == 'top_down') then
  if(file_exist(trim('INPUT/hs_forcing.res.nc'))) then
     !call nullify_domain()
     call read_data(trim('INPUT/hs_forcing.res.nc'), 'tg_prev', tg_prev, grid_domain)
	 print *, 'READING PREVIOUS HEAT CAPACITY DATA'
  else
	print *, 'SPINNING UP HEAT CAPACITY'
	! spin up the surface temps with heat capacity
	print *, 'Depth:', ml_depth
  	tg = 250        ! starting temperature for surface
	spin_count = 0
	step_days = 1
	do
		tg_prev = tg
		dt_integer = dt_integer + 86400*step_days		! step by a day at a time
		spin_count = spin_count + 1
		call update_orbit(dt_integer, dec, orb_dist)
		call calc_hour_angle(lat, dec, hour_angle)
		s(:,:) = solar_const/pi*(hour_angle*sin(lat)*sin(dec) + cos(lat)*cos(dec)*sin(hour_angle))
		t_radbal = ((1-albedo)*s(:,:)/stefan)**0.25
		t_trop(:,:) = t_radbal(:,:)/(2**0.25)

		h_trop = 1.0/(16*lapse)*(1.3863*t_trop + sqrt((1.3863*t_trop)**2 + 32*lapse*tau_s*h_a*t_trop))
		t_surf = t_trop + h_trop*lapse

		tg(:,:) =  stefan*86400*step_days/(ml_depth*heat_capacity)*(t_surf**4 - tg_prev**4) + tg_prev

		if (spin_count >= spinup_time) then
			print *, 'SPINUP COMPLETE AFTER ', spin_count, 'ITERATIONS'
			exit
		endif

	 enddo

  endif
  endif

!     ----- convert local heating variables from degrees to radians -----

      xwidth  = local_heating_xwidth*PI/180.
      ywidth  = local_heating_ywidth*PI/180.
      xcenter = local_heating_xcenter*PI/180.
      ycenter = local_heating_ycenter*PI/180.

!     ----- Make sure xcenter falls in the range zero to 2*PI -----

      xcenter = xcenter - twopi*floor(xcenter/twopi)

!     ----- convert local_heating_srfamp from deg/day to deg/sec ----

      srfamp = local_heating_srfamp/SECONDS_PER_DAY

!     ----- compute coefficients -----

! If positive, damping time units are (1/s),  value is the inverse of damping time.
! If negative, damping time units are (days), value is the damping time. It is converted to (1/s)

      if (ka < 0.) then
        tka = -1./(86400*ka)
      else
        tka = ka
      endif
      if (ks < 0.) then
        tks = -1./(86400*ks)
      else
        tks = ks
      endif
      if (kf < 0.) then
        vkf = -1./(86400*kf)
      else
        vkf = kf
      endif

!     ----- for tracers -----

      if (trsink < 0.) trsink = -86400.*trsink
      trdamp = 0.; if (trsink > 0.) trdamp = 1./trsink

!     ----- register diagnostic fields -----

      id_teq = register_diag_field ( mod_name, 'teq', axes(1:3), Time, &
                      'equilibrium temperature (deg K)', 'deg_K'   , &
                      missing_value=missing_value, range=(/0.,700./) )

      if (trim(equilibrium_t_option) == 'top_down') then
      id_h_trop = register_diag_field ( mod_name, 'h_trop', axes(1:2), Time, &
                      'tropopause height (km)', 'km'   , &
                      missing_value=missing_value, range=(/0.,200./) )
      endif

      id_newtonian_damping = register_diag_field ( mod_name, 'tdt_ndamp', axes(1:3), Time, &
                      'Heating due to newtonian damping (deg/sec)', 'deg/sec' ,    &
                       missing_value=missing_value     )

      id_tdt = register_diag_field ( mod_name, 'tdt', axes(1:3), Time, &
                      'Total heating: newtonian damping + local heating (deg/sec)', 'deg/sec' ,    &
                       missing_value=missing_value     )

      if(trim(local_heating_option) /= '') then
        id_local_heating=register_diag_field ( mod_name, 'local_heating', axes(1:3), Time, &
                        'Local heating (deg/sec)', 'deg/sec' ,    &
                         missing_value=missing_value     )
      endif

      id_udt = register_diag_field ( mod_name, 'udt_rdamp', axes(1:3), Time, &
                      'zonal wind tendency due to rayleigh damping (m/s2)', 'm/s2',       &
                       missing_value=missing_value     )

      id_vdt = register_diag_field ( mod_name, 'vdt_rdamp', axes(1:3), Time, &
                      'meridional wind tendency due to rayleigh damping (m/s2)', 'm/s2',  &
                       missing_value=missing_value     )
      id_SAM_ind = register_diag_field( mod_name, 'SAM_ind_norm', axes(1:2), Time, &
                       'simplified SAM index', 'unitless', &
                         missing_value=missing_value   )
      id_SBAM_ind = register_diag_field( mod_name, 'SBAM_ind_norm', axes(1:2), Time, &
                       'simplified SBAM index', 'unitless', &
                       missing_value=missing_value   )
      id_eke      = register_diag_field( mod_name, 'EKE', axes(2:2), Time, &
                       'EKE at 300 hPa', 'm2/s2', &
                       missing_value=missing_value   )
      id_vhs      = register_diag_field( mod_name, 'vhs', axes(2:3), Time, &
                       'meridional wind', 'm/s', &
                       missing_value=missing_value   )
      id_uhs      = register_diag_field( mod_name, 'uhs', axes(2:3), Time, &
                       'zonal  wind', 'm/s', &
                       missing_value=missing_value   )
      
 

      if (do_conserve_energy) then
         id_tdt_diss = register_diag_field ( mod_name, 'tdt_diss_rdamp', axes(1:3), &
                   Time, 'Dissipative heating from Rayleigh damping (deg/sec)', 'deg/sec',&
                   missing_value=missing_value     )

         id_diss_heat = register_diag_field ( mod_name, 'diss_heat_rdamp', axes(1:2), &
                   Time, 'Vertically integrated dissipative heating from Rayleigh damping (W/m2)', 'W/m2')
      endif

     if(trim(local_heating_option) == 'from_file') then
        call interpolator_init(heating_source_interp, trim(SAM_heating_file)//'.nc', lonb, latb, data_out_of_bounds=(/CONSTANT/))
     endif
     if(trim(equilibrium_t_option) == 'from_file') then
       call interpolator_init (temp_interp, trim(equilibrium_t_file)//'.nc', lonb, latb, data_out_of_bounds=(/CONSTANT/))
     endif
     if(relax_to_specified_wind) then
       call interpolator_init (u_interp,    trim(u_wind_file)//'.nc', lonb, latb, data_out_of_bounds=(/CONSTANT/))
       call interpolator_init (v_interp,    trim(v_wind_file)//'.nc', lonb, latb, data_out_of_bounds=(/CONSTANT/))
     endif

     call astronomy_init()

     module_is_initialized  = .true.

 end subroutine hs_forcing_init

!#######################################################################

 subroutine hs_forcing_end

!-----------------------------------------------------------------------
!
!       routine for terminating held-suarez benchmark module
!             (this routine currently does nothing)
!
!-----------------------------------------------------------------------

 if(trim(local_heating_option) == 'from_file') then
    call interpolator_end(heating_source_interp)
 endif

 if(trim(equilibrium_t_option) == 'from_file') then
   call interpolator_end(temp_interp)
 endif

 if(relax_to_specified_wind) then
   call interpolator_end(u_interp)
   call interpolator_end(v_interp)
 endif

 call set_domain(grid_domain)
 if (trim(equilibrium_t_option) == 'top_down') then
   call write_data(trim('RESTART/hs_forcing.res'), 'tg_prev', tg_prev, grid_domain)
   deallocate (tg_prev)
 endif

 module_is_initialized = .false.

 end subroutine hs_forcing_end

!#######################################################################

 subroutine newtonian_damping ( Time, lat, lon, ps, p_full, p_half, t, tdt, teq, mask )

!-----------------------------------------------------------------------
!
!   routine to compute thermal forcing for held & suarez (1994)
!   benchmark calculation.
!
!-----------------------------------------------------------------------

type(time_type), intent(in)         :: Time
real, intent(in),  dimension(:,:)   :: lat, ps, lon
real, intent(in),  dimension(:,:,:) :: p_full, t, p_half
real, intent(out), dimension(:,:,:) :: tdt, teq
real, intent(in),  dimension(:,:,:), optional :: mask

!-----------------------------------------------------------------------

          real, dimension(size(t,1),size(t,2)) :: &
     sin_lat, cos_lat, sin_lat_2, cos_lat_2, t_star, cos_lat_4, &
     tstr, sigma, the, tfactr, rps, p_norm, sin_sublon_2, coszen, fracday

       real, dimension(size(t,1),size(t,2),size(t,3)) :: tdamp
       real, dimension(size(t,2),size(t,3)) :: tz
       real :: rrsun

       integer :: k, i, j
       real    :: tcoeff, pref

!-----------------------------------------------------------------------
!------------latitudinal constants--------------------------------------

      sin_lat  (:,:) = sin(lat(:,:))
      cos_lat (:,:) = cos(lat(:,:))
      sin_lat_2(:,:) = sin_lat(:,:)*sin_lat(:,:)
      cos_lat_2(:,:) = 1.0-sin_lat_2(:,:)
      cos_lat_4(:,:) = cos_lat_2(:,:)*cos_lat_2(:,:)

      t_star(:,:) = t_zero - delh*sin_lat_2(:,:) - eps*sin_lat(:,:)
      tstr  (:,:) = t_strat - eps*sin_lat(:,:)

!-----------------------------------------------------------------------
      if(trim(equilibrium_t_option) == 'from_file') then
         call get_zonal_mean_temp(Time, p_half, tz)
      endif
      tcoeff = (tks-tka)/(1.0-sigma_b)
      pref = P00
      rps  = 1./ps

      do k = 1, size(t,3)

!  ----- compute equilibrium temperature (teq) -----

      if(equilibrium_t_option == 'from_file') then
         do i=1, size(t,1)
         do j=1, size(t,2)
           teq(i,j,k)=tz(j,k)
         enddo
         enddo
      else if(trim(equilibrium_t_option) == 'Held_Suarez') then
         p_norm(:,:) = p_full(:,:,k)/pref
         the   (:,:) = t_star(:,:) - delv*cos_lat_2(:,:)*log(p_norm(:,:))
         teq(:,:,k) = the(:,:)*(p_norm(:,:))**KAPPA
         teq(:,:,k) = max( teq(:,:,k), tstr(:,:) )
      else if(uppercase(trim(equilibrium_t_option)) == 'EXOPLANET') then
         call diurnal_exoplanet(lat, lon, Time, coszen, fracday, rrsun)
         t_star(:,:) = t_zero - delh*(1 - coszen(:,:)) - eps*sin_lat(:,:)
         p_norm(:,:) = p_full(:,:,k)/pref
         the   (:,:) = t_star(:,:) - delv*coszen(:,:)*log(p_norm(:,:))
         teq(:,:,k) = the(:,:)*(p_norm(:,:))**KAPPA
         teq(:,:,k) = max( teq(:,:,k), tstr(:,:) )
      else if(uppercase(trim(equilibrium_t_option)) == 'EXOPLANET2') then
         call diurnal_exoplanet(lat, lon, Time, coszen, fracday, rrsun)
         t_star(:,:) = t_strat
         p_norm(:,:) = p_full(:,:,k)/p_trop
         teq(:,:,k) = t_star(:,:)*cos_lat(:,:)*(p_norm(:,:))**alpha
         teq(:,:,k) = max( teq(:,:,k), t_strat )
      else
         call error_mesg ('hs_forcing_nml', &
         '"'//trim(equilibrium_t_option)//'"  is not a valid value for equilibrium_t_option',FATAL)
      endif

!  ----- compute damping -----
      sigma(:,:) = p_full(:,:,k)*rps(:,:)
      where (sigma(:,:) <= 1.0 .and. sigma(:,:) > sigma_b)
        tfactr(:,:) = tcoeff*(sigma(:,:)-sigma_b)
        tdamp(:,:,k) = tka + cos_lat_4(:,:)*tfactr(:,:)
      elsewhere
        tdamp(:,:,k) = tka
      endwhere

      enddo

      do k=1,size(t,3)
         tdt(:,:,k) = -tdamp(:,:,k)*(t(:,:,k)-teq(:,:,k))
      enddo

      !print *, 'tdt dims newt=', size(tdt,1), size(tdt,2), size(tdt,3) 

      if (present(mask)) then
         tdt = tdt * mask
         teq = teq * mask
      endif

!-----------------------------------------------------------------------

 end subroutine newtonian_damping

!#######################################################################

 subroutine rayleigh_damping ( Time, ps, p_full, p_half, u, v, udt, vdt, mask )

!-----------------------------------------------------------------------
!
!           rayleigh damping of wind components near surface
!
!-----------------------------------------------------------------------

type(time_type), intent(in)         :: Time
real, intent(in),  dimension(:,:)   :: ps
real, intent(in),  dimension(:,:,:) :: p_full, p_half, u, v
real, intent(out), dimension(:,:,:) :: udt, vdt
real, intent(in),  dimension(:,:,:), optional :: mask

!-----------------------------------------------------------------------

real, dimension(size(u,1),size(u,2)) :: sigma, vfactr, rps

integer :: i,j,k
real    :: vcoeff
real, dimension(size(u,2),size(u,3)) :: uz, vz
real :: umean, vmean

!-----------------------------------------------------------------------
!----------------compute damping----------------------------------------

      if(relax_to_specified_wind) then
        call get_zonal_mean_flow(Time, p_half, uz, vz)
      endif

      vcoeff = -vkf/(1.0-sigma_b)
      rps = 1./ps

      do k = 1, size(u,3)
      if (relax_to_specified_wind) then
         do j=1, size(u,2)
            umean=sum(u(:,j,k))/size(u,1)
            vmean=sum(v(:,j,k))/size(v,1)
            udt(:,j,k) = (uz(j,k)-umean)*vkf
            vdt(:,j,k) = (vz(j,k)-vmean)*vkf
         enddo
      else

         sigma(:,:) = p_full(:,:,k)*rps(:,:)

         where (sigma(:,:) <= 1.0 .and. sigma(:,:) > sigma_b)
            vfactr(:,:) = vcoeff*(sigma(:,:)-sigma_b)
            udt(:,:,k)  = vfactr(:,:)*u(:,:,k)
            vdt(:,:,k)  = vfactr(:,:)*v(:,:,k)
         elsewhere
            udt(:,:,k) = 0.0
            vdt(:,:,k) = 0.0
         endwhere

      endif
      enddo

      if (present(mask)) then
          udt = udt * mask
          vdt = vdt * mask
      endif

!-----------------------------------------------------------------------

 end subroutine rayleigh_damping

!#######################################################################

 subroutine tracer_source_sink ( flux, damp, p_half, r, rdt, kbot )

!-----------------------------------------------------------------------
      real, intent(in)  :: flux, damp, p_half(:,:,:), r(:,:,:)
      real, intent(out) :: rdt(:,:,:)
   integer, intent(in), optional :: kbot(:,:)
!-----------------------------------------------------------------------
      real, dimension(size(r,1),size(r,2),size(r,3)) :: source, sink
      real, dimension(size(r,1),size(r,2))           :: pmass

      integer :: i, j, kb
      real    :: rdamp
!-----------------------------------------------------------------------

      rdamp = damp
      if (rdamp < 0.) rdamp = -86400.*rdamp   ! convert days to seconds
      if (rdamp > 0.) rdamp = 1./rdamp

!------------ simple surface source and global sink --------------------

      source(:,:,:)=0.0

   if (present(kbot)) then
      do j=1,size(r,2)
      do i=1,size(r,1)
         kb = kbot(i,j)
         pmass (i,j)    = p_half(i,j,kb+1) - p_half(i,j,kb)
         source(i,j,kb) = flux/pmass(i,j)
      enddo
      enddo
   else
         kb = size(r,3)
         pmass (:,:)    = p_half(:,:,kb+1) - p_half(:,:,kb)
         source(:,:,kb) = flux/pmass(:,:)
   endif

     sink(:,:,:) = rdamp*r(:,:,:)
     rdt(:,:,:) = source(:,:,:)-sink(:,:,:)

!-----------------------------------------------------------------------

 end subroutine tracer_source_sink

!######################################################################

subroutine local_heating ( Time, is, js, lon, lat, ps, p_full, p_half, u, v, a, b, id_SAM, id_SBAM, id_eke, id_uhs, id_vhs, tdt )

type(time_type), intent(in)         :: Time
integer, intent(in)                 :: is,js
real, intent(in)                    :: a,b !a,b = SAM,SBAM scale facs
real, intent(in),  dimension(:,:)   :: lon, lat, ps
real, intent(in),  dimension(:,:,:) :: p_full, u, v
real, intent(in),  dimension(:,:,:) :: p_half
real, intent(out), dimension(:,:,:) :: tdt
integer, intent(in)                 :: id_SAM, id_SBAM, id_eke, id_uhs, id_vhs

logical :: used
integer :: i, j, k, o, istart,iend,jstart,jend
real :: lon_temp, x_temp, p_factor, ua, va
real :: SAM_ind, SBAM_ind
real :: SAM_ind_mean, SAM_ind_std, SBAM_ind_mean, SBAM_ind_std
real :: SAM_ind_norm,SBAM_ind_norm
real, dimension(128) :: SAM_EOF_full
real, dimension(37) :: SBAM_EOF
real, dimension(128) :: SBAM_EOF_full
real, dimension(size(lon,1),size(lon,2)) :: lon_factor
real, dimension(size(lat,1),size(lat,2)) :: lat_factor
real, dimension(size(p_half,1),size(p_half,2),size(p_half,3)) :: p_half2
real, dimension(size(u,2)) :: eke_SBAM
real, dimension(size(u,1),size(u,2)) :: eke_SBAM_lon
real,dimension(size(u,2)) :: uz
real,dimension(size(u,1),size(u,2)) :: SAM_diag, SBAM_diag 
real,dimension(size(u,2),size(u,3)) :: udiag, vdiag

SAM_EOF_full(:) = 0.
SAM_EOF_full(28) = 1.
SAM_EOF_full(43) = -1.

SAM_ind_mean=0.327
SAM_ind_std=7.495
SBAM_ind_mean=1326.09
SBAM_ind_std=247.33

do j=1,size(u,2)
  uz(j) = sum(u(:,j,21))/size(u,1)
enddo

do j=1,size(u,2)
  do o=1,size(u,3)
  udiag(j,o) = sum(u(:,j,o))/size(u,1)
  vdiag(j,o) = sum(v(:,j,o))/size(v,1)
enddo
enddo

call get_grid_domain(istart,iend,jstart,jend)
SAM_ind = DOT_PRODUCT(uz, SAM_EOF_full(jstart:jend))
call mpp_sum(SAM_ind)
SAM_ind_norm = (SAM_ind - SAM_ind_mean)/SAM_ind_std
!SAM_ind = sum(ufull(:,43,2))/size(u,1) - sum(ufull(:,28,2))/size(u,1)

SBAM_EOF = (/-1.07e-3, -4.27e-4,5.04e-4,1.86e-3,3.94e-3,6.93e-3,1.09e-2, &
1.61e-2,2.30e-2,3.12e-2,4.08e-2,5.22e-2,6.55e-2,8.07e-2,9.76e-2,1.16e-1, &
1.37e-1, 1.59e-1, 1.82e-1,2.06e-1,2.31e-1,2.53e-1,2.72e-1,2.87e-1,2.96e-1, &
2.98e-1, 2.91e-1, 2.77e-1,2.56e-1,2.30e-1, 2.00e-1, 1.68e-1, 1.36e-1,1.06e-1, &
8.00e-2,5.82e-2,4.10e-2 /)
 

SBAM_EOF_full(:) = 0.
SBAM_EOF_full(14:50) = SBAM_EOF
!print *, 'SBAM_EOF_full mag squared=', DOT_PRODUCT(SBAM_EOF_full,SBAM_EOF_full) 


do i=1,size(p_half,3)
  p_half2(:,:,i)=p_half(:,:,size(p_half,3)-i+1)
enddo

!do j=1,size(u,2)
! do i=1,size(u,1)
!  ua=sum(u(:,j,13))/size(u,1)
!  va=sum(v(:,j,13))/size(v,1)
!  eke_SBAM_lon(i,j) = 0.5*((u(i,j,13)-ua)**2 + (v(i,j,13)-va)**2)
! enddo
! eke_SBAM(j) = sum(eke_SBAM_lon(:,j))/size(u,1)
!enddo

!SBAM_ind = DOT_PRODUCT(eke_SBAM, SBAM_EOF_full(jstart:jend))
!call mpp_sum(SBAM_ind)
!SBAM_ind_norm = (SBAM_ind - SBAM_ind_mean)/SBAM_ind_std

!print *, 'SAM_ind', SAM_ind_norm
!print *, 'SBAM ind =', SBAM_ind_norm
!print *, 'size(u,1)=', size(u,1)

tdt(:,:,:)=0.


if(trim(local_heating_option) == 'from_file') then
   call interpolator(heating_source_interp, p_half, tdt, &
        trim(SAM_heating_file))
  ! call interpolator(heating_source_interp, p_half, tdt_SBAM, & 
       ! trim(SBAM_heating_file))
!print *, 'SAM_amplitude=', a
  do i=1,size(tdt,1)
   do j=1,size(tdt,2)
    do k=1,size(tdt,3)
      tdt(i,j,k) = a*(tdt(i,j,k)*SAM_ind_norm)! &
      ! + b*(tdt_SBAM(i,j,k)*SBAM_ind_norm)
    enddo
   enddo
  enddo !DV 10/7/20
  ! tdt(:,:,:) = a_mat(:,:,:)*(tdt_SAM(:,:,:)*SAM_norm_mat(:,:,:)) & 
  ! + b_mat(:,:,:)*(tdt_SBAM(:,:,:)*SBAM_norm_mat(:,:,:)) 
   !Heating scheme is added here
   !print *, 'constant test 1=', MAXVAL(tdt(:,:,:))
  ! print *, 'SAM_ind_norm=', SAM_ind_norm
  ! print *, 'dims tdt_SAM =', size(p_half,1),size(p_half,2),size(p_half,3) - 1
  ! print *, 'dims tdt =', size(tdt,1),size(tdt,2),size(tdt,3)
 !  print *, 'constant test 2 =',a*MAXVAL(tdt_SAM(:,:,:))
 !  print *, 'nonconst test =',a*(MAXVAL(tdt_SAM(:,:,:))*SAM_ind_norm)
else if(trim(local_heating_option) == 'Isidoro') then
   do j=1,size(lon,2)
   do i=1,size(lon,1)
     lon_temp = lon(i,j)
     ! Make sure lon_temp falls in the range zero to 2*PI
     x_temp = floor(lon_temp/twopi)
     lon_temp = lon_temp - twopi*x_temp
     lon_factor(i,j) = exp(-.5*((lon_temp-xcenter)/xwidth)**2)
     lat_factor(i,j) = exp(-.5*((lat(i,j)-ycenter)/ywidth)**2)
     do k=1,size(p_full,3)
       p_factor = exp((p_full(i,j,k)-ps(i,j))/local_heating_vert_decay)
       tdt(i,j,k) = srfamp*lon_factor(i,j)*lat_factor(i,j)*p_factor
     enddo
   enddo
   enddo
else
  call error_mesg ('hs_forcing_nml','"'//trim(local_heating_option)//'"  is not a valid value for local_heating_option',FATAL)
endif

do i=1,size(u,1)
 do j=1,size(u,2)
    SAM_diag(i,j) = SAM_ind_norm
    !SBAM_diag(i,j) = SBAM_ind_norm
enddo
enddo

if (id_SAM > 0) used = send_data ( id_SAM, SAM_diag, Time)
!if (id_SBAM > 0) used = send_data ( id_SBAM, SBAM_diag, Time)
if (id_eke > 0) used = send_data ( id_eke, eke_SBAM, Time)
if (id_uhs > 0) used = send_data ( id_uhs, udiag, Time)
if (id_vhs > 0) used = send_data ( id_vhs, vdiag, Time)
!print *, 'id_SBAM=', id_SBAM
!print *, 'SBAM used=', used
!print *, 'send data SBAM=', send_data ( id_SBAM, SBAM_ind_norm, Time)

end subroutine local_heating

!#######################################################################


!#######################################################################

subroutine get_zonal_mean_flow ( Time, p_half, uz, vz)

type(time_type), intent(in)         :: Time
real, intent(in),  dimension(:,:,:) :: p_half
real, intent(inout), dimension(:,:) :: uz,vz

integer :: j, k
real, dimension(size(p_half,1),size(p_half,2),size(p_half,3)-1) :: uf,vf
call interpolator( u_interp, p_half, uf, trim(u_wind_file))
call interpolator( v_interp, p_half, vf, trim(v_wind_file))

do j=1,size(p_half,2)
do k=1,size(p_half,3)-1
  uz(j,k)=sum(uf(:,j,k))/size(uf,1)
  vz(j,k)=sum(vf(:,j,k))/size(vf,1)
enddo
enddo
end subroutine get_zonal_mean_flow
!#######################################################################

subroutine get_zonal_mean_temp ( Time, p_half, tm)

type(time_type), intent(in)         :: Time
real, intent(in),  dimension(:,:,:) :: p_half
real, intent(inout), dimension(:,:) :: tm

integer :: j, k
real, dimension(size(p_half,1),size(p_half,2),size(p_half,3)-1) :: tf
call interpolator( temp_interp, p_half, tf, trim(equilibrium_t_file))

do j=1,size(p_half,2)
do k=1,size(p_half,3)-1
  tm(j,k)=sum(tf(:,j,k))/size(tf,1)
enddo
enddo
end subroutine get_zonal_mean_temp

!#######################################################################
!#######################################################################

! Functions for top-down newtonian forcing
! Future work will integrate these with astronomy_mod and
! other existing code.

!#######################################################################
!#######################################################################

subroutine update_orbit(current_time, dec, orb_dist)

integer, intent(in)					:: current_time
real, intent(out)					:: dec, orb_dist

real :: theta, mean_anomaly, ecc_anomaly, true_anomaly


	mean_anomaly = 2*pi/(orbital_period*86400)*(current_time-peri_time*orbital_period*86400)
    call calc_ecc_anomaly(mean_anomaly, ecc, ecc_anomaly)
    true_anomaly = 2*atan(((1 + ecc)/(1 - ecc))**0.5 * tan(ecc_anomaly/2))
    orb_dist = smaxis * (1 - ecc**2)/(1 + ecc*cos(true_anomaly))
    theta = 2*pi*current_time/(orbital_period*86400)
    dec = asin(sin(obliq*pi/180)*sin(theta))

end subroutine update_orbit

!########################################################################

subroutine calc_hour_angle(lat, dec, hour_angle)

real, intent(in) 					:: dec
real, intent(in), dimension(:,:) 	:: lat
real, intent(out), dimension(:,:)	:: hour_angle

real, dimension(size(lat,1), size(lat,2)) :: inv_hour_angle

inv_hour_angle = -tan(lat(:,:))*tan(dec)
where (inv_hour_angle > 1)
	inv_hour_angle = 1
endwhere
where (inv_hour_angle < -1)
    inv_hour_angle = -1
endwhere

hour_angle = acos(inv_hour_angle)

end subroutine calc_hour_angle

!#######################################################################

 subroutine calc_ecc_anomaly( mean_anomaly, ecc, ecc_anomaly)

 real, intent(in) :: mean_anomaly, ecc
 real, intent(out) :: ecc_anomaly
 real :: dE, d
 integer, parameter :: maxiter = 30
 real, parameter :: tol = 1.d-10
 integer :: k

 ecc_anomaly = mean_anomaly
 d = ecc_anomaly - ecc*sin(ecc_anomaly) - mean_anomaly
 do k=1,maxiter
        dE = d/(1 - ecc*cos(ecc_anomaly))
        ecc_anomaly = ecc_anomaly - dE
        d = ecc_anomaly - ecc*sin(ecc_anomaly) - mean_anomaly
        if (abs(d) < tol) then
                exit
        endif
 enddo

 if (k > maxiter) then
        if (abs(d) > tol) then
                print *, '*** Warning: eccentric anomaly has not converged'
        endif
 endif

 end subroutine calc_ecc_anomaly

!###################################################################

subroutine top_down_newtonian_damping ( Time, lat, ps, p_full, p_half, t, tdt, teq, dt, h_trop, zfull, mask )

!-----------------------------------------------------------------------
!
!   routine to compute thermal forcing a top-down, tropopause defined model
!
!-----------------------------------------------------------------------

type(time_type), intent(in)         :: Time
real, intent(in)                    :: dt
real, intent(in),  dimension(:,:)   :: lat, ps
real, intent(in),  dimension(:,:,:) :: p_full, t, p_half, zfull
real, intent(out), dimension(:,:,:) :: tdt, teq
real, intent(out), dimension(:,:)   :: h_trop
real, intent(in),  dimension(:,:,:), optional :: mask

!-----------------------------------------------------------------------

          real, dimension(size(t,1),size(t,2)) :: &
     sin_lat, sin_lat_2, cos_lat, cos_lat_2, cos_lat_4, &
     tstr, sigma, the, tfactr, rps, p_norm, sin_sublon_2, &
     coszen, fracday, t_trop, s, hour_angle, t_surf, tg, t_radbal

       real, dimension(size(t,1),size(t,2),size(t,3)) :: tdamp, heights
       real, dimension(size(t,2),size(t,3)) :: tz
       real :: rrsun

       integer :: k, i, j
       real    :: tcoeff, pref,  dec, orb_dist
       integer :: days, seconds, dt_integer


!-----------------------------------------------------------------------
!------------find out the time------------------------------------------

       call get_time(Time, seconds, days)
       dt_integer = 86400*days + seconds

!-----------------------------------------------------------------------
!------------latitudinal constants--------------------------------------

      sin_lat  (:,:) = sin(lat(:,:))
      cos_lat  (:,:) = cos(lat(:,:))
      sin_lat_2(:,:) = sin_lat(:,:)*sin_lat(:,:)
      cos_lat_2(:,:) = 1.0-sin_lat_2(:,:)
      cos_lat_4(:,:) = cos_lat_2(:,:)*cos_lat_2(:,:)

!-----------------------------------------------------------------------
!----------- orbital calculations --------------------------------------

    call update_orbit(dt_integer, dec, orb_dist)


    call calc_hour_angle(lat, dec, hour_angle)

    s(:,:) = solar_const/pi*(hour_angle*sin_lat*sin(dec) + cos_lat*cos(dec)*sin(hour_angle))

    t_radbal = ((1-albedo)*s(:,:)/stefan)**0.25


    ! --- compute tropopause height (h_trop) and apply heat capacity
    t_trop(:,:) = t_radbal(:,:)/(2**0.25)
    h_trop = 1.0/(16*lapse)*(1.3863*t_trop + sqrt((1.3863*t_trop)**2 + 32*lapse*tau_s*h_a*t_trop))

	t_surf = t_trop(:,:) + h_trop*lapse
	tg(:,:) = stefan*dt/(ml_depth*heat_capacity)*(t_surf**4 - tg_prev**4) + tg_prev
	tg_prev = tg
	t_trop(:,:) = tg(:,:) - h_trop*lapse


	!----- stratosphere temperature ------------
      tstr  (:,:) = t_strat - eps*sin_lat(:,:)


    ! ---- relaxation coefficient ----
      tcoeff = (tks-tka)/(1.0-sigma_b)
      pref = P00
      rps  = 1./ps


    do k = 1, size(t,3)

!  ----- compute equilibrium temperature (teq) -----


        teq(:,:,k) = t_trop + lapse*(h_trop-zfull(:,:,k)/1000)
        if (stratosphere_t_option == 'c_above_tp') then
            do i=1, size(t,1)
            do j=1, size(t,2)
                if(zfull(i,j,k)/1000 >= h_trop(i,j)) then
                 	teq(i,j,k) = tstr(i,j)
                endif
            enddo
            enddo
        elseif (stratosphere_t_option == 'hs_like') then
                teq(:,:,k) = max(teq(:,:,k), tstr(:,:))
		elseif (stratosphere_t_option == 'extend_tp') then
			do i=1,size(t,1)
			do j=1,size(t,1)
                if (zfull(i,j,k)/1000 >= h_trop(i,j)) then
                    teq(i,j,k) = t_trop(i,j)
                endif
			enddo
			enddo
		else
			teq(:,:,k) = max(teq(:,:,k), 0.)
        endif
!  ----- compute damping -----
! ------ is symmetric about the equator, change this?? -----
        sigma(:,:) = p_full(:,:,k)*rps(:,:)
        where (sigma(:,:) <= 1.0 .and. sigma(:,:) > sigma_b)
            tfactr(:,:) = tcoeff*(sigma(:,:)-sigma_b)
            tdamp(:,:,k) = tka + cos_lat_4(:,:)*tfactr(:,:)
        elsewhere
            tdamp(:,:,k) = tka
        endwhere

    enddo

      do k=1,size(t,3)
         tdt(:,:,k) = -tdamp(:,:,k)*(t(:,:,k)-teq(:,:,k))
      enddo

      !print *, maxval(tdt), maxval(teq), maxval(h_trop)

      if (present(mask)) then
         tdt = tdt * mask
         teq = teq * mask
      endif

!-----------------------------------------------------------------------

 end subroutine top_down_newtonian_damping

end module hs_forcing_mod
