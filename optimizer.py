import optuna
import subprocess
import shutil
import re

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

    
    mv = f"""mkdir -p /kaggle/working/BoostTrack/results/trackers/MOT20-test && \
        mv /kaggle/working/BoostTrack/results/trackers/MOT20-val/* /kaggle/working/BoostTrack/results/trackers/MOT20-test/
    """

    print(f"Running trial {trial.number} with params: {params}")
    
    # Run tracking script
    shutil.rmtree("/kaggle/working/BoostTrack/cache", ignore_errors=True)
    subprocess.run(command, shell=True)
    subprocess.run(mv, shell=True)

    # Evaluate performance
    eval_cmd = f"""python /kaggle/working/BoostTrack/external/TrackEval/scripts/run_mot_challenge.py \
        --SPLIT_TO_EVAL test --GT_FOLDER results/gt/ \
        --TRACKERS_FOLDER results/trackers/ --BENCHMARK MOT20 \
        --TRACKERS_TO_EVAL {trial.number}_post_gbi
    """
    process = subprocess.run(eval_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output_lines = [line.strip() for line in process.stdout.split("\n") if line.strip()]

    if len(output_lines) >= 12:
        hota_line = output_lines[-12]  # Get the required line

        # Normalize spaces within the line (convert multiple spaces into a single space)
        hota_line_cleaned = " ".join(hota_line.split())

        # Extract the second word (first numerical value)
        hota_words = hota_line_cleaned.split()
        if len(hota_words) > 1:
            try:
                hota_score = float(hota_words[1])  # Convert to float
                print("HOTA Score:", hota_score)
            except ValueError:
                print("Error: HOTA score is not a valid number.")
        else:
            print("Error: Unexpected format in HOTA line.")
    else:
        print("Error: Output does not have enough lines.")
    print(f"Trial {trial.number} - HOTA Score: {hota_score}")
    return hota_score

# Run Bayesian Optimization
study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=70)

print(f"Best HOTA Score: {study.best_value}")
print(f"Best Parameters: {study.best_params}")
