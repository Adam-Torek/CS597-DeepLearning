#!/bin/bash

#SBATCH -J Latent_Attention_Pretrainer                  # job name
#SBATCH -o slurm_logs/attention_pretraining.o%j         # output and error file name (%j expands to jobID)                             # number of tasks per node to run on
#SBATCH --ntasks-per-node 4				# number of tasks to run per node
#SBATCH -N 1			                                # Number of nodes to run on
#SBATCH --gres=gpu:L40:4	                            # request a gpu
#SBATCH -p gpu-l40                                      # queue (partition)
#SBATCH -t 24:00:00                                     # run time (hh:mm:ss)

source activate $1

model_name=bert
latent_size=0

srun python3 wikitext_trainer.py --model_name $model_name \
                                 --dataset_name Salesforce/wikitext \
                                 --dataset_subname wikitext-103-v1 \
                                 --model_save_name ajtorek/$model_name-wikitext-$latent_size \
                                 --learning_rate 1e-4 \
                                 --batch_size 64 \
                                 --accelerator gpu \
                                 --num_nodes 1 \
                                 --num_devices 4 \
                                 --latent_dimension $latent_size
				 --max_training_steps 1000000000 \
				 --num_epochs 10 
