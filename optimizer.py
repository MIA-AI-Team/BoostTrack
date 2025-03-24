import optuna
import subprocess
import shutil

# Define the objective function
def objective(trial):
    params = {
        "min_hits": trial.suggest_int("min_hits", 2, 6),
        "min_box_area": trial.suggest_int("min_box_area", 10, 150),
        "max_age": trial.suggest_int("max_age", 15, 50),
        "det_thresh": trial.suggest_float("det_thresh", 0.1, 0.8),
        "iou_threshold": trial.suggest_float("iou_threshold", 0.1, 0.6),
        "lambda_iou": trial.suggest_float("lambda_iou", 0.1, 1.3),
        "lambda_mhd": trial.suggest_float("lambda_mhd", 0.1, 0.7),
        "lambda_shape": trial.suggest_float("lambda_shape", 0.1, 0.7),
        "dlo_boost_coef": trial.suggest_float("dlo_boost_coef", 0.3, 0.9)
    }

    # Construct command
    command = f"""python main.py --dataset mot20 --exp_name {trial.number} --test_dataset \
        --conf {params["det_thresh"]} \
        --iou_thresh {params["iou_threshold"]} \
        --min_hits {params["min_hits"]} \
        --max_age {params["max_age"]} \
        --lambda_iou {params["lambda_iou"]} \
        --lambda_mhd {params["lambda_mhd"]} \
        --min_box_area {params["min_box_area"]} \
        --lambda_shape {params["lambda_shape"]} \
        --dlo_boost_coef {params["dlo_boost_coef"]} \
        --detector "yolox"
    """

    mv = f"""mv results/trackers/MOT20-val/* results/trackers/MOT20-test/"""

    print(f"Running trial {trial.number} with params: {params}")
    
    # Run tracking script
    shutil.rmtree("cache", ignore_errors=True)
    subprocess.run(command, shell=True)
    subprocess.run(mv, shell=True)

    # Evaluate performance
    eval_cmd = f"""python external/TrackEval/scripts/run_mot_challenge.py \
        --SPLIT_TO_EVAL test --GT_FOLDER results/gt/ \
        --TRACKERS_FOLDER results/trackers/ --BENCHMARK MOT20 \
        --TRACKERS_TO_EVAL {trial.number}_post_gbi
    """
    subprocess.run(eval_cmd, shell=True)

    # Read HOTA score
    with open(f"results/trackers/MOT20-test/{trial.number}_post_gbi/pedestrian_summary.txt", "r") as file:
        last_line = file.readlines()[-1].strip()
        hota_score = float(last_line.split()[0])  # Extract first number (HOTA)

    print(f"Trial {trial.number} - HOTA Score: {hota_score}")
    return hota_score

# Run Bayesian Optimization
study = optuna.create_study(direction="maximize")

study.enqueue_trial({
    "min_hits": 5,  
    "min_box_area": 59,  
    "max_age": 20,  
    "det_thresh": 0.3,  
    "iou_threshold": 0.1,  
    "lambda_iou": 0.9,  
    "lambda_mhd": 0.4,  
    "lambda_shape": 0.3,  
    "dlo_boost_coef": 0.7  
})

study.optimize(objective, n_trials=30)

print(f"Best HOTA Score: {study.best_value}")
print(f"Best Parameters: {study.best_params}")
