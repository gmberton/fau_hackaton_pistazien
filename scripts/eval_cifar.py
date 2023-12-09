
from pathlib import Path


CSV_PATH = "output.csv"
CIFAR_PATH = "cifar"
CIFAR_PATH = "../../../cifar"
TOP = 1


with open(CSV_PATH, "r") as file:
    lines = file.read().splitlines()

lines = lines[1:]  # Remove header

cnt_corrects = 0
for line in lines:
    splits = line.split(",")
    query_path = Path(splits[0])
    query_class = query_path.parent.name
    
    preds_paths = splits[1:]
    
    for pred_path in preds_paths[:TOP]:
        pred_path = Path(pred_path)
        pred_class = pred_path.parent.name
        is_correct = query_class == pred_class
        if is_correct:
            cnt_corrects += 1
            break

accuracy = cnt_corrects / len(lines)
print(f"Top-{TOP} accuracy {accuracy * 100:.1f}")

