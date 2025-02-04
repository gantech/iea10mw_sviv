#!/bin/bash
#SBATCH --nodes=7
#SBATCH --time=48:00:00
#SBATCH --account=sviv
#SBATCH --switches=1@00:01:00
#SBATCH --job-name=iea10mw_sviv
#SBATCH --output=out.%x_%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xuepeng.fu@nrel.gov
#SBATCH --array=1-40

source /nopt/nrel/apps/cpu_stack/env_cpe23.sh
module load cray-python
module load PrgEnv-intel
module use /projects/sviv/exawind-manager/environments/sviv/modules
module load exawind

export LD_PRELOAD=/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel/libmpi_intel.so.12:/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel/libmpifort_intel.so.12


export MPICH_VERSION_DISPLAY=1
export MPICH_OFI_CQ_STALL=1
#export MPICH_OFI_CQ_STALL_USECS=3
export MPICH_OFI_CQ_STALL_USECS=16
export MPICH_OFI_CXI_COUNTER_REPORT=2
export MPICH_ENV_DISPLAY=1

BINDW='--cpu_bind=map_cpu:0,52,13,65,26,78,39,91,1,53,14,66,27,79,40,92,2,54,15,67,28,80,41,93,3,55,16,68,29,81,42,94,4,56,17,69,30,82,43,95,5,57,18,70,31,83,44,96,6,58,19,71,32,84,45,97,7,59,20,72,33,85,46,98,8,60,21,73,34,86,47,99,9,61,22,74,35,87,48,100,10,62,23,75,36,88,49,101,11,63,24,76,37,89,50,102,12,64,25,77,38,90,51,103'

export FI_MR_CACHE_MONITOR=memhooks
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export FI_CXI_RX_MATCH_MODE=software


cd `sed -n '${SLURM_ARRAY_TASK_ID}p' list_of_cases`
cd cfd_run
cp ../openfast_run/turb_00_output.nc .
srun -N 7 -n 672 --ntasks-per-node=96 --distribution=block:cyclic $BINDW exawind --nwind 288 --awind 384 iea10mw.yaml &> log_7nodes.out
