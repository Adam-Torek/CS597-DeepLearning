import argparse
import os
import json
import pandas as pd

def main():
    glue_tasks = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
    parser = argparse.ArgumentParser(prog="gather_glue_results.py", 
                                     description="Helper script used to gather data from GLUE evaluation results in JSON" \
                                     "format.")
    
    parser.add_argument("--evaluation_directory", 
                            required=True, 
                            type=str, 
                            help="Directory that contains output of evaluation files from GLUE.")
    
    parsed_arguments = parser.parse_args()
    top_level_directory = parsed_arguments.evaluation_directory

    directories = os.listdir(top_level_directory)

    model_configurations = []
    
    if len(directories) == 1:
        top_level_directory = os.path.join(top_level_directory, directories[0])
    
    for directory in os.listdir(top_level_directory):
        model_configurations.append(directory)

    glue_evaluation_results = {}
    for configuration in model_configurations:
        glue_evaluation_results[configuration] = {}
        for task in glue_tasks:
            config_task_file_path = os.path.join(top_level_directory, configuration, task, "eval_results.json")
            try:
                with open(config_task_file_path, "r") as config_task_results:
                    config_task_results_dict = json.load(config_task_results)
                    for metric, value in config_task_results_dict.items():
                        if "accuracy" in metric or "correlation" in metric:
                            glue_evaluation_results[configuration][task] = value

            except Exception as e:
                print("unable to load results for {0}".format(config_task_file_path))
                continue
            
    glue_results_dataframe = pd.DataFrame.from_dict(glue_evaluation_results, orient="index")

    glue_results_dataframe.to_csv(os.path.join(top_level_directory, "all_glue_results.csv"))

    
if __name__ == "__main__":
    main()
