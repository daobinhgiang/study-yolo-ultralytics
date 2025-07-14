import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil

def get_latest_run(run_base_dir):
    subdirs = [os.path.join(run_base_dir, d) for d in os.listdir(run_base_dir) if os.path.isdir(os.path.join(run_base_dir, d))]
    if not subdirs:
        raise Exception("No runs found.")
    latest_run = max(subdirs, key=os.path.getmtime)
    return latest_run

def plot_results(csv_path):
    results = pd.read_csv(csv_path)
    epochs = results.index + 1

    plt.figure(figsize=(10, 5))
    if 'train/loss' in results.columns and 'metrics/accuracy' in results.columns:
        plt.subplot(1,2,1)
        plt.plot(epochs, results['train/loss'], label='Train Loss')
        plt.plot(epochs, results['val/loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')

        plt.subplot(1,2,2)
        plt.plot(epochs, results['metrics/accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Validation Accuracy')
        plt.tight_layout()
        plt.show()
    else:
        print("No 'train/loss' or 'metrics/accuracy' columns in results.csv.")

def print_final_metrics(csv_path):
    results = pd.read_csv(csv_path)
    final_row = results.iloc[-1]
    print("\n=== Final Epoch Metrics ===")
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', shutil.get_terminal_size().columns)
    pd.set_option('display.width', 500)
    pd.set_option('display.max_colwidth', None)
    # print(final_row)
    print (results)

if __name__ == "__main__":
    # Set this to your runs base directory
    RUNS_DIR = "/home/giangdb/Documents/ETC/runs/classify"
    latest_run = get_latest_run(RUNS_DIR)
    print(f"Latest run directory: {latest_run}")

    csv_path = os.path.join(latest_run, "results.csv")
    if not os.path.isfile(csv_path):
        print(f"No results.csv found in {latest_run}")
    else:
        print_final_metrics(csv_path)
        # plot_results(csv_path)

    # Optionally open result images (confusion matrix, F1 curve, etc.)
    from glob import glob
    img_files = glob(os.path.join(latest_run, "*.png"))
    if img_files:
        print("Available plots in results directory:")
        for f in img_files:
            print(f"- {f}")

from ultralytics import YOLO

model = YOLO('/home/giangdb/Documents/ETC/runs/classify/train8/weights/best.pt')  # or your model path
print(model.model)  # <--- This will print the architecture
