# Mesh refinement studies for forced motion

To pull the required meshes, do the following

1. Add this to your `~/.bashrc`

```
function anacondaenv() {
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/nopt/nrel/apps/cpu_stack/utilities/06-24/conda_22/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/nopt/nrel/apps/cpu_stack/utilities/06-24/conda_22/etc/profile.d/conda.sh" ]; then
        . "/nopt/nrel/apps/cpu_stack/utilities/06-24/conda_22/etc/profile.d/conda.sh"
    else
        export PATH="/nopt/nrel/apps/cpu_stack/utilities/06-24/conda_22/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
}
```

2. Type this into your terminal. Replace `USER_NAME` with yours on kestrel.

```
anacondaenv
conda activate /projects/sviv/turbine_mesh/miniconda3
dvc remote add kestrel ssh://USER_NAME@kl3.hpc.nrel.gov:/projects/sviv/iea10mw_sviv
dvc pull -r kestrel
```

3. Submit OpenFAST job first `sbatch run_openfast.slurm`. 

4. Submit all decomposition cases.

5. Submit all forced motion cases once the jobs in #3 and #4 have been successful.



