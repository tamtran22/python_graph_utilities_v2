#!/bin/bash

#******************************************************************************
# Solve 1-D pulmonary airflow model for pressure and flow rate distribution. 
# -----------------------------------------------------------------------------
# Version 2 (Shinjiro Miyawaki, 7/20/2016)
# Original version (Sanghun Choi?)
#******************************************************************************

# Paths
# -----
#CODE_DIR=$HOME/mbin/Codes/fluid1d/exes
CODE_DIR=/data3/common/yjhong/exes
PREP_EXE=$CODE_DIR/prep_pres_flow_lung.exe
PRES_EXE=$CODE_DIR/pres_flow_lung.exe
PRES_SH=pres_flow_lung.sh
INP_DIR=data_input
OUTP_DIR=data_output
MESH1D_DIR=data_mesh1D
PLT_ND_DIR=data_plt_nd
XFLX_DIR=data_xflx

# Set up the job
# --------------
# Global
get_stats=1  # Get stats. of diameter ratio (1:yes, 0:no)
hetero_All=2  # Heterogeneity applied to 0:no, 1:'E'-'T', 2:all branches
lobar_hetero=1  # Heterogeneity applied 0:globally, 1:lobe by lobe
fileStatApollo="10081_TLC_Central_1D_withDiameter_rf.dat" # 1D mesh for <get_stats> = 1
BCond=1  # read(*,*) i_p_type 1,inQ & outP, 2, inP & outQ
ixflx=2  ## Boundary condition testing

# 1D mesh
file=Output_10081_Amount_St_whole_WT.dat  # 1D mesh
scale=0.001  # Scaling factor
igen_lim=21  # Highest generation number

# Gas properties
rhog=1.12 # Fluid density (kg/m^3)
visg=1.64d-05  # Kinematic viscosity (m^2/s)

# Simulation parameters
i_max=400  # Total number of time steps
idump=2  # Number of time steps between output files
i_solver=3  # Matrix solver (1:BCG, 2:GMRES, 3:DAGMG)
npicard=20   # Total number of nonlinear iteration
i_kinetic=1  # Kinetic energy effect (0:off, 1:on)
i_disp=1  # Dynamic displacement (0:off, 1:on)
i_diam=1  # Dynamic diameter (0:off, 1:on)
map=1  # diameter & displacement mapping w.r.t volume (0:off, 1:on)
i_compl=1 # Wall compliance (0:off, 1:on)
i_inert=1 # Inertance model (0:off, 1:on)
i_Swan=0  # Swan et al. acinar compliance model (0:off, 1:on)

# Simulation parameters
res_freq=5.8 # Resonant frequency (cps = Hz)
Eaw=3300
kin_seg=1

# Breathing pattern
tperiod=4.0  # Breathing period (s)
v_tidal=1.000d-03  # Tidal volume (m^3)
v_FRC=2.007d-03  # Get FRC global lung volume (m^3)
v_TLC=4.007d-03  # Get TLC global lung volume (m^3)
i_wform=1  # 1:sin, 2:Longest breathing waveform
i_wave=0   # 0:sin, 1:Oscilatory sin flow rate for <i_wform> = 1
time_ein=2.5  # Time to end insp. for <i_wform> = 2
xtime_pin=0.167  # Fraction of time to peak insp. for <i_wform> = 2
time_lvol=time_lvol.in

# Clinical information
gender=1  # Gender (0:male, 1:female) for <get_stats> = 1
age=61.0  # Age (yr) for <get_stats> = 1
height=161.0  # Height (cm) for <get_stats> = 1
ResistModel=0 # 0, SChoi's Model; 1, Pedley's Model
min_rdiam=0.25
max_rdiam=1.5
acoef=0.4215962909348021
bcoef=-0.35404874159884814

# Before the job
# --------------
# Copy input files and codes to $INP_DIR
mkdir -p ./$INP_DIR
cp $PRES_SH ./$INP_DIR  # W/o time stamp to record start time
cp -p  $file ./$INP_DIR

# Get input files
$PREP_EXE $file $scale $BCond $get_stats \
	$gender $age $height $fileStatApollo $i_diam $map

# Run the job
# -----------
$PRES_EXE $rhog $visg $tperiod $v_tidal $i_max $idump \
	$scale $BCond $ixflx $igen_lim $i_solver $npicard $i_kinetic \
        $hetero_All $lobar_hetero $min_rdiam $ResistModel $max_rdiam\
	$i_wform $i_disp $i_diam $map $i_compl $i_inert $i_Swan \
	$v_FRC $v_TLC $res_freq $Eaw $acoef $bcoef $kin_seg $i_wave $file $time_lvol $acoef $bcoef

# After the job
# -------------
mkdir -p ./$OUTP_DIR ./$MESH1D_DIR ./$PLT_ND_DIR ./$XFLX_DIR

# Files from $PREP_EXE
mv flag_summary.dat single_001.dat kind_ne000001.dat \
	tec_whole.dat ./$OUTP_DIR/
if [ $get_stats -eq 1 ]; then
	mv Apollo_Statistics.dat ./$OUTP_DIR/
fi

# Files from $PRES_EXE
mv timehis.dat Nodal_Distance_from_Trachea.dat ./$OUTP_DIR/
mv mesh1D_******.dat ./$MESH1D_DIR/
mv plt_nd_******.dat ./$PLT_ND_DIR/
mv xflx_******.dat ./$XFLX_DIR/
if [ $hetero_All -gt 0 ]; then
	mv Diameter_Heterogeneity.dat ./$OUTP_DIR/
fi
