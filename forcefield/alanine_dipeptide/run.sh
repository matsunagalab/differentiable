#!/bin/bash
#SBATCH -p all
#SBATCH -J sim_target # job name
#SBATCH -n 1  # num of total mpi processes
#SBATCH -c 1  # num of threads per mpi processes
#SBATCH -o run.log
#SBATCH -w n4

#python sim.py alanine-dipeptide-nowater.pdb GB99dms_new.xml traj_new
python sim.py alanine-dipeptide-nowater.pdb GB99dms_target.xml traj_target 100000