from pathlib import Path
dataset="scx2"
best_model_path = f"saved_model/{dataset}_best.pth"
print(f"Saving best model to {best_model_path}")
# Path(best_model_path.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
# print(best_model_path.rsplit('/')[0])
