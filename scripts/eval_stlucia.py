
import math
from pathlib import Path


def get_distance(coords_A, coords_B):
    return math.sqrt((float(coords_B[0])-float(coords_A[0]))**2 + (float(coords_B[1])-float(coords_A[1]))**2)


def compute_accuracy(csv_path, top=5):
    with open(csv_path, "r") as file:
        lines = file.read().splitlines()
    
    lines = lines[1:]  # Remove header
    
    cnt_corrects = 0
    for line in lines:
        splits = line.split(",")
        query_path = Path(splits[0])
        query_utm = query_path.name.split("@")[1:3]
        
        preds_paths = splits[1:]
        
        for pred_path in preds_paths[:top]:
            pred_path = Path(pred_path)
            pred_utm = pred_path.name.split("@")[1:3]
            
            # Distance between query and prediction, in meters
            distance = get_distance(query_utm, pred_utm)
            
            is_correct = distance < 25
            if is_correct:
                cnt_corrects += 1
                break
    
    accuracy = cnt_corrects / len(lines)
    print(f"Top-{top} accuracy {accuracy * 100:.1f}")
    return accuracy

