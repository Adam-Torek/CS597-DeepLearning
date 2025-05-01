#!/bin/bash

#SBATCH -J Electra_GLUE_Evaluations                     # job name
#SBATCH -o slurm_logs/electra_moe_glue.o%j       # output and error file name (%j expands to jobID)                             # number of tasks per node to run on
#SBATCH --ntasks-per-node 4				                # number of tasks to run per node
#SBATCH -N 1			                                # Number of nodes to run on
#SBATCH --gres=gpu:L40:4	                            # request a gpu
#SBATCH -p gpu-l40                                      # queue (partition)
#SBATCH -t 24:00:00                                     # run time (hh:mm:ss)

source activate $1

num_experts=("8")
top_k=("1" "2")
glue_tasks=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "stsb" "wnli")
capacity_factor=1.0

for expert_num in "${num_experts[@]}"
do
	for k in "${top_k[@]}"
	do
        echo "Running evaluation for model with $expert_num experts and $k top choices"
        if [ $k != "1" ] && [ $expert_num == "1" ];
        then
            echo "Skipping evaluation for model with $expert_num experts and $k top choices"
            continue
        fi
        for task in "${glue_tasks[@]}"
        do
            model_name=ajtorek/electra-num_experts-$expert_num-top_k-$k-capacity_factor-$capacity_factor
            echo "Running task $task"
            srun python3 run_glue.py --model_name_or_path $model_name \
                                --task_name $task \
                                --do_train True \
                                --do_eval True \
                                --output_dir evaluation_results/$model_name/$task \
                                --tokenizer_name google/electra-base-discriminator \
                                --ignore_mismatched_sizes True \
                                --overwrite_output_dir True \
                                --save_strategy no 

            echo "Completed task $task for model $model_name"
            rm -rf evaluation_results/ajtorek/**/**/checkpoint*
        done
	done
done	
