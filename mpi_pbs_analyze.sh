#!/bin/sh
#
# job name (default is the name of pbs script file)
#---------------------------------------------------
#PBS -N job_name
#
# Submit the job to the queue "queue_name" (minerva_h_p, gpu_l_p)
#---------------------------------------------------
# Minerva  queue      High	      minerva_h_p         wall time limit=168 h             Minerva, Training
#       						  av. hosts n029 - n080
#							  user CPU number limit=160 
# General queue    	  Low          all_l_p              wall time limit=24 h		    
#							   av. hosts n001-n026
#                                                              n029-n076	                 All Users
#	                                                temporary:   n101-n108      
#							   user CPU number limit = 160                                             
# Non-prime time       Low         np_weekend       active from Thu 17:00 to Sun 8:00   
#     queue                                                maximum 63 h
#                                                       av. hosts n001-n080                    All Users
#                                                   user CPU number limit = 200
#                                                   job  CPU number limit = 120    
# Graphical Processing   High        gpu_l_p	  	wall time limit= 72 h	        
#     Units (GPU)						av. hosts gn01 - gn04                 All users
#							        CPU number limit=12
#   General		Low	      general_ld	    wall time limit=24 h            
#  Large Disk						    av. hosts n097 - n100                    All users
#
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
#PBS -l select=9:ncpus=12:mpiprocs=12
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
mpirun -np 108 python $HOME/.local/bin/analyzeAtmo3D.py --use_simulated --job_id $PBS_JOBID $HOME/code/atmosphere/atmotomo/data/configurations/two_clouds_km_low_density/configuration.ini
#mpirun -np 96 python $HOME/.local/bin/analyzeAtmo3D.py --use_simulated --ref_mc /u/amitibo/data/New_Low_Density --job_id $PBS_JOBID $HOME/code/atmosphere/atmotomo/data/configurations/two_clouds_low_density/configuration.ini
#mpirun -np 96 python $HOME/.local/bin/analyzeAtmo3D.py --mcarats /u/amitibo/data/mcarats/3
#mpirun -np 24 python $HOME/.local/bin/analyzeAtmo3D.py --ref_images /u/amitibo/data/20Images_mc_limited --no_air

# comment: the "np" must be equal the number of chunks multiplied by the number of "ncpus"
