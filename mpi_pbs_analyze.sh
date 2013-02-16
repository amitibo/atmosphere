#!/bin/sh
#
# job name (default is the name of pbs script file)
#---------------------------------------------------
#PBS -N job_name
#
# Submit the job to the queue "queue_name" (minerva_h_p, gpu_l_p)
#---------------------------------------------------
#PBS -q  minerva_h_p
#
# Send the mail messages (see below) to the specified user address 
#-----------------------------------------------------------------
#PBS -M amitibo@tx.technion.ac.il
#
# send me mail when the job begins
#---------------------------------------------------
#PBS -mb
# send me mail when the job ends  
#---------------------------------------------------
#PBS -me
# send me mail when the job aborts (with an error)
#---------------------------------------------------
#PBS -ma
#
# Comment: if you want more than one message to be sent, you must group 
# the flags in one line, otherwise only the last flag executes.
#-------------------------------------------------------------
#PBS -mbea
#
#
# resource limits: number and distribution of parallel processes 
#------------------------------------------------------------------ 
#PBS -l select=8:ncpus=12:mpiprocs=12
#
# comment: this select statement means: use M chunks (nodes), 
# use N (=< 12) CPUs for N mpi tasks on each of M nodes. 
# "scatter" will use exactly N CPUs from each node, while omitting 
# "-l place" statement will fill all available CPUs of M nodes 
#  
#  resource limits: max. wall clock time during which job can run
#-----------------------------------------------------------------
# #PBS -l walltime=3:20:00
#
#  specifying working directory
#------------------------------------------------------
PBS_O_WORKDIR=$HOME/code/atmosphere
cd $PBS_O_WORKDIR

#
# running MPI executable with M*N processes  
#------------------------------------------------------
mpirun -np 96 python $HOME/.local/bin/analyzeAtmo3D.py --ref_images /u/amitibo/data/Low_Density

# comment: the "np" must be equal the number of chunks multiplied by the number of "ncpus"
