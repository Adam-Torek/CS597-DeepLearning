#!/bin/bash

#SBATCH -J Electra_MoE_Pretrainer                  # job name
#SBATCH -o slurm_logs/electra_moe_pretraining.o%j         # output and error file name (%j expands to jobID)                             # number of tasks per node to run on
#SBATCH --ntasks-per-node 2				# number of tasks to run per node
#SBATCH -N 1			                                # Number of nodes to run on
#SBATCH --gres=gpu:L40:2	                            # request a gpu
#SBATCH -p gpu-l40                                      # queue (partition)
#SBATCH -t 12:00:00                                     # run time (hh:mm:ss)

source activate $1



srun python3 wikitext_trainer.py --dataset_name Salesforce/wikitext \
                            --dataset_subname wikitext-2-v1 \
                            --model_save_name ajtorek/electra-num_experts-1-top_k-1-capacity_factor-1.0 \
                            --learning_rate 1e-4 \
                            --batch_size 64 \
                            --accelerator gpu \
                            --num_nodes 1 \
                            --num_devices 2 \
                            --num_experts 1 \
                            --top_k 1 \
                            --capacity_factor 1.0 \
                            --max_training_steps 1000000000 \
                            --num_epochs 10

num_experts=("2" "4" "8")
top_k=("1" "2")
capacity_factor=1.0

for expert_num in $num_experts 
do
	for k in $top_k 
	do
		srun python3 wikitext_trainer.py --dataset_name Salesforce/wikitext \
                                 --dataset_subname wikitext-2-v1 \
				                 --model_save_name ajtorek/electra-num_experts-$expert_num-top_k-$k-capacity_factor-$capacity_factor \
                                 --learning_rate 1e-4 \
                                 --batch_size 64 \
                                 --accelerator gpu \
                                 --num_nodes 1 \
                                 --num_devices 2 \
                                 --num_experts $expert_num \
                                 --top_k $k \
                                 --capacity_factor $capacity_factor \
                                 --max_training_steps 1000000000 \
                                 --num_epochs 10

	done
done	
