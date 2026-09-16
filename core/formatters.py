
#core/formatters.py
import csv
import os
import pandas as pd
from datetime import datetime


def export_results_to_csv(all_predictions: dict, output_dir: str = ".", output_filename: str = None) -> str:
    """
    Exports evaluation results and confidence scores to a CSV file.

    Args:
        all_predictions (dict): Dictionary mapping image names to model predictions.
                                Format: { 'image_name.jpg': { 'Model_Name': ('Species', confidence_score) } }
        output_dir (str): Directory to save the CSV file.
        output_filename (str): Target CSV filename. Auto-generates with a timestamp if None.

    Returns:
        str: The full path to the saved CSV file.
    """
    if not output_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"evaluation_results_{timestamp}.csv"

    output_path = os.path.join(output_dir, output_filename)

    # Flatten the nested dictionary into a list of row dictionaries
    rows = []
    for img_name, models_data in all_predictions.items():
        for model_name, (species, confidence) in models_data.items():
            rows.append({
                "Image Name": img_name,
                "Model": model_name,
                "Predicted Species": species,
                "Confidence (%)": round(confidence, 2)
            })

    # Convert to DataFrame for robust CSV formatting and escaping
    df = pd.DataFrame(rows)

    # Export to CSV
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n[+] Results successfully exported to {output_path}")

    return output_path


def check_low_confidence_alternatives(model_name, probs, idx_to_name_func, confidence_threshold=90.0,
                                      min_alt_confidence=30.0):
    """
    Shared utility to inspect PyTorch probability tensors and print alternatives
    if confidence falls below a threshold.
    """
    import torch

    top_probs, top_idxs = torch.topk(probs, k=min(5, len(probs)))
    top_conf = top_probs[0].item() * 100

    if top_conf < confidence_threshold:
        viable_alternatives = []
        for i in range(1, len(top_probs)):
            conf = top_probs[i].item() * 100
            if conf > min_alt_confidence:
                idx = top_idxs[i].item()
                name = idx_to_name_func(idx)
                viable_alternatives.append((name, conf))

        if viable_alternatives:
            print(
                f"\n   [{model_name}] Low confidence ({top_conf:.2f}% < {confidence_threshold}%). Other likely alternatives (>{min_alt_confidence}%):")
            for name, conf in viable_alternatives:
                print(f"      - {name} ({conf:.2f}%)")
            print("-" * 50)

def print_results(image_name: str, predictions: dict):
    """Formats the inferences clearly into the console."""
    print(f"\nResults for '{image_name}':")
    print("-" * 65)
    for model_name, (species, confidence) in predictions.items():
        print(f"[{model_name:<18}] Species: {species:<22} | Confidence: {confidence:>5.2f}%")
    print("-" * 65)