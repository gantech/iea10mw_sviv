#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --reservation=hfm
#SBATCH --account=hfm
#SBATCH --switches=1@00:01:00
#SBATCH --job-name=iea10mw_sviv_openfast
#SBATCH --output=out.%x_%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xuepeng.fu@nrel.gov

source /nopt/nrel/apps/cpu_stack/env_cpe23.sh
module load cray-python
module load PrgEnv-intel
module use /projects/sviv/exawind-manager/environments/openfast/modules/openfast-develop
module load oneapi-2023.2.0

for i in `cat list_of_cases`
do
    cd $i/openfast_run
    srun -n 1 --exclusive openfastcpp inp.yaml &> log &
    cd -
done

wait
