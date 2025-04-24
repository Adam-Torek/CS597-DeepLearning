#!/bin/bash

#SBATCH -J Electra_MoE_Pretrainer                  # job name
#SBATCH -o slurm_logs/electra_moe_pretraining.o%j         # output and error file name (%j expands to jobID)                             # number of tasks per node to run on
#SBATCH --ntasks-per-node 4				# number of tasks to run per node
#SBATCH -N 1			                                # Number of nodes to run on
#SBATCH --gres=gpu:L40:4	                            # request a gpu
#SBATCH -p gpu-l40                                      # queue (partition)
#SBATCH -t 24:00:00                                     # run time (hh:mm:ss)

source activate $1

run_training(){
    num_exps=$1
    k_active=$2
    cap_fac=$3

    srun python3 wikitext_trainer.py --dataset_name Salesforce/wikitext \
                                 --dataset_subname wikitext-2-v1 \
				 --model_save_name ajtorek/electra-num_experts-$num_exps-top_k-$k_active-capacity_factor-$cap_fac \
                                 --learning_rate 1e-4 \
                                 --batch_size 64 \
                                 --accelerator gpu \
                                 --num_nodes 1 \
                                 --num_devices 4 \
                                 --num_experts $num_exps \
                                 --top_k $k_active \
                                 --capacity_factor $cap_fac \
                                 --max_training_steps 10000000000 \
                                 --num_epochs 10
}

run_training 1 1 1.0

num_experts=("2" "4" "8")
top_k=("1" "2")
capacity_factor=1.0

for expert_num in "${num_experts[@]}" 
do
	for k in "${top_k[@]}" 
	do
		run_training $expert_num $k $capacity_factor 

	done
done	
