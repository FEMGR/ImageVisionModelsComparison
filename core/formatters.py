# core/formatters.py

import os
import pandas as pd


def export_evaluation_results(
    evaluation_results: list,
    output_dir: str = "./results",
    output_filename: str = "evaluation_results.csv",
) -> str:
    """Export ground-truth evaluation results to CSV."""

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        output_filename,
    )

    df = pd.DataFrame(evaluation_results)

    df.to_csv(
        output_path,
        index=False,
    )

    print(f"Evaluation results successfully exported to: " f"{output_path}")

    return output_path


def export_results_to_csv(
    all_predictions: dict,
    output_dir: str = ".",
    output_filename: str = None,
) -> str:
    """Export ranked Top-K predictions to CSV."""

    os.makedirs(output_dir, exist_ok=True)

    if output_filename is None:
        output_filename = "plant_model_predictions.csv"

    output_path = os.path.join(
        output_dir,
        output_filename,
    )

    rows = []

    for image_name, models_data in all_predictions.items():

        for model_name, ranked_predictions in models_data.items():

            for rank, (species, confidence) in enumerate(
                ranked_predictions,
                start=1,
            ):
                rows.append(
                    {
                        "Image Name": image_name,
                        "Model": model_name,
                        "Rank": rank,
                        "Predicted Species": species,
                        "Confidence (%)": round(
                            confidence,
                            2,
                        ),
                    }
                )

    df = pd.DataFrame(rows)

    df.to_csv(
        output_path,
        index=False,
    )

    print(f"\nPrediction results successfully exported to: " f"{output_path}")

    return output_path


def export_evaluation_summary(
    evaluation_summary: list,
    output_dir: str = "./results",
    output_filename: str = "evaluation_summary.csv",
) -> str:
    """Export aggregate experiment metrics to CSV."""

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        output_filename,
    )

    df = pd.DataFrame(evaluation_summary)

    df.to_csv(
        output_path,
        index=False,
    )

    print(f"\nEvaluation summary successfully exported to: " f"{output_path}")

    return output_path


def check_low_confidence_alternatives(
    model_name,
    probs,
    idx_to_name_func,
    confidence_threshold=90.0,
    min_alt_confidence=30.0,
):
    """
    Inspect probability tensors and print alternative predictions
    when the top prediction has low confidence.
    """
    import torch

    top_probs, top_idxs = torch.topk(
        probs,
        k=min(5, len(probs)),
    )

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
                f"\n   [{model_name}] Low confidence "
                f"({top_conf:.2f}% < {confidence_threshold}%). "
                f"Other likely alternatives "
                f"(>{min_alt_confidence}%):"
            )

            for name, conf in viable_alternatives:
                print(f"      - {name} ({conf:.2f}%)")

            print("-" * 50)


def print_results(image_name: str, predictions: dict):
    """Print ranked Top-K predictions for each model."""

    print(f"\nResults for '{image_name}':")
    print("-" * 65)

    for model_name, ranked_predictions in predictions.items():
        print(f"\n[{model_name}]")

        if not ranked_predictions:
            print("  No prediction available.")
            continue

        for rank, (species, confidence) in enumerate(
            ranked_predictions,
            start=1,
        ):
            print(f"  Top-{rank}: " f"{species:<35} " f"{confidence:>6.2f}%")
