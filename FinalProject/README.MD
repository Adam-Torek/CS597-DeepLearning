# CS 597: Special Topics In Deep Learning Final Project
# Exploring Mixture of Experts with ELECTRA
## Author: Adam Torek
## Professor: Jun Zhuang
## Spring 2025 Semester

### Project Overview

This repository contains the code I used to explore mixture of expert (MoE) configurations with ELECTRA, a transformer-based language model. The purpose of this project is to test different mixture of expert configurations, if any, would enhance performance on GLUE, a natural language understanding benchmark. Mixture of Expert (MoE) models are created by putting many smaller networks, each called "experts" side by side. Each expert is selectively activated based on the output of a routing network. My experiments focused on altering the number of experts and active experts inside an MoE layer that replaced the FFN layer. I tested different six MoE variations and ELECTRA-Base by itself by pretraining on WikiText-103 before fine-tuning and evaluating on the GLUE benchmark. Unfortunately, the configurations I tested did not yield any improved results over the base model. This is probably because I did not use a large enough number of experts or active experts to make a difference. This shows that more experts and active experts are needed to get improved performance.

### Project Files

 - modeling_electra.py: Code for modified ELECTRA-MoE model with replaced FFN layer
 - modeling_electra_config.py: Configuration for setting number of experts and active experts
 - wikitext_trainer.py: Running script for pretraining ELECTRA-MoE and ELECTRA-Base
 - run_glue.py: Running script for fine-tuning and evaluating ELECTRA-MoE and ELECTRA-Base on GLUE
 - gather_glue_results.py: Helper script for putting GLUE evaluation results into a single file
 - glue_evaluation_results_visualizer.ipynb: Jupyter notebook used to visualize GLUE results
 - Electra_MoE_Pretrainer.sh: Shell script for running wikitext_trainer.py using Slurm
 - Run_GLUE_Evaluation.sh: Shell script for running run_glue.py using Slurm

### Running Instructions

To pretrain ELECTRA-MoE using a particular experts or active experts, use wikitext_trainer.py. To fine-tune and evaluate a pretrained ELECTRA-MoE model on GLUE, use run_glue.py. The run_glue.py script must be given a pretrained model. To collect results from various GLUE evaluations into a single file, use gather_glue_results.py, and to visualize those results, use the glue_evaluation_results_visualizer.ipynb notebook file and point it to the CSV containing all of your GLUE results. Each script contains a help mode detailing their arguments and purpose. 

You will likely need or benefit from a HuggingFace account. Here is their documentation and account page: [HuggingFace Documentation](https://huggingface.co/docs)

Note: To install dependencies, run the following command below in a terminal and a new Python environment:

`conda env create --name electra_moe --file electra_moe_env.yml`

Note: The bash scripts are designed to submit jobs on a Slurm computing cluster. They were specifically written to run on the Borah Research Computing Cluster
owned by Boise State University on their GPU nodes. Here is the documentation to run on Borah: [Borah Documentation](https://bsu-docs.readthedocs.io/en/latest/)

