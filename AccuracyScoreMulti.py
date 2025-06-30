import os
import json
import glob
from collections import defaultdict

def calculate_accuracy():
    # Path to prediction files
    pred_dir = "outputs_vehicle_results_Brand_Finetune_Diff_trivial_negatives/_max5/preds"
    
    # Get all JSON files
    json_files = glob.glob(os.path.join(pred_dir, "*.json"))
    
    # Initialize counters
    total_files = len(json_files)
    
    # Define top-n values and score thresholds to evaluate
    top_n_values = [1, 2] # Adjust as needed!!!
    score_thresholds = [0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025]
    
    # Initialize results dictionary
    results = defaultdict(lambda: defaultdict(int))
    
    print(f"Processing {total_files} files...")
    
    # Process each file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Get all labels and scores
            labels = data["labels"]
            scores = data["scores"]
            
            # Check for each top-n and threshold combination
            for top_n in top_n_values:
                top_labels = labels[:top_n]
                top_scores = scores[:top_n]
                
                for threshold in score_thresholds:
                    is_correct = False
                    for label, score in zip(top_labels, top_scores):
                        if label == 0 and score > threshold:
                            is_correct = True
                            break
                    
                    if is_correct:
                        results[f"top{top_n}"][threshold] += 1
                
        except Exception as e:
            print(f"Error processing {os.path.basename(json_file)}: {e}")
    
    # Print results
    print("\n=== ACCURACY RESULTS ===")
    print(f"Total files evaluated: {total_files}")
    print("\n")
    
    # Create a formatted table for results
    print(f"{'Top-N':<10} | {'Threshold':<10} | {'Correct':<10} | {'Accuracy':<10}")
    print("-" * 47)
    
    for top_n in top_n_values:
        for threshold in score_thresholds:
            correct = results[f"top{top_n}"][threshold]
            accuracy = correct / total_files if total_files > 0 else 0
            print(f"Top-{top_n:<7} | {threshold:<10.3f} | {correct:<10} | {accuracy*100:<7.2f}%")
    
    # Print original metric for compatibility
    original_correct = results["top5"][0.1]
    original_accuracy = original_correct / total_files if total_files > 0 else 0
    print("\nOriginal metric (Top-5, threshold=0.1):")
    print(f"Correct predictions: {original_correct}")
    print(f"Accuracy: {original_accuracy:.4f} ({original_accuracy*100:.2f}%)")
    
    return results

if __name__ == "__main__":
    calculate_accuracy()