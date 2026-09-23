# core/evaluation.py

"""
Utilities for evaluating plant identification model predictions.

This module compares model predictions against a known ground-truth
species and calculates Top-1, Top-3, and Top-5 correctness.
"""
import re


def normalize_species_name(name: str) -> str:
    """
    Normalize a species name for comparison.

    This provides a basic normalization so that names such as:

        Tagetes erecta
        Tagetes erecta L.

    can be compared as the same species.

    Args:
        name: Species name.

    Returns:
        Normalized species name.
    """
    if not name:
        return ""

        # ---------------------------------------------------------
        # 1. Basic whitespace and case normalization
        # ---------------------------------------------------------

    name = name.strip().lower()

    # Replace repeated whitespace with a single space.
    name = " ".join(name.split())

    # ---------------------------------------------------------
    # 2. Normalize common punctuation
    # ---------------------------------------------------------

    name = name.replace(",", " ")
    name = " ".join(name.split())

    # ---------------------------------------------------------
    # 3. Remove common botanical author citation: "L."
    #
    # Examples:
    #   "tagetes erecta l."
    #   "tagetes erecta l"
    # ---------------------------------------------------------

    name = re.sub(
        r"\s+l\.?$",
        "",
        name,
    ).strip()

    # ---------------------------------------------------------
    # 4. Handle missing whitespace between genus and species
    #
    # Example:
    #   "hydrangeamacrophylla"
    #       -> "hydrangea macrophylla"
    #
    # This is handled only for known genus names rather than
    # inserting spaces blindly into arbitrary species names.
    # ---------------------------------------------------------

    known_genera = [
        "hibiscus",
        "hydrangea",
        "lansium",
        "passiflora",
        "tagetes",
    ]

    for genus in known_genera:

        if (
            name.startswith(genus)
            and name != genus
            and not name.startswith(genus + " ")
        ):
            name = genus + " " + name[len(genus) :]
            break

    # ---------------------------------------------------------
    # 5. Final whitespace normalization
    # ---------------------------------------------------------

    return " ".join(name.split())


def species_match(
    predicted_species: str,
    ground_truth: str,
) -> bool:
    """
    Determine whether a predicted species matches the ground truth.

    Args:
        predicted_species: Predicted species name.
        ground_truth: Correct species name.

    Returns:
        True if the normalized species names match.
    """
    return normalize_species_name(predicted_species) == normalize_species_name(
        ground_truth
    )


def top_k_correct(
    predictions: list,
    ground_truth: str,
    k: int,
) -> bool:
    """
    Determine whether the ground-truth species appears within Top-K.

    Args:
        predictions:
            Ranked predictions in descending confidence order.

            Example:
            [
                ("Tagetes erecta", 89.64),
                ("Tagetes patula", 7.21),
                ("Tagetes tenuifolia", 2.13),
            ]

        ground_truth:
            Correct species name.

        k:
            Number of predictions to consider.

    Returns:
        True if the ground truth appears within the Top-K predictions.
    """
    if not predictions:
        return False

    top_k_predictions = predictions[:k]

    return any(species_match(species, ground_truth) for species, _ in top_k_predictions)


def evaluate_prediction(
    predictions: list,
    ground_truth: str,
) -> dict:
    """
    Evaluate a ranked prediction list against ground truth.

    Args:
        predictions:
            Ranked list of (species, confidence) tuples.

        ground_truth:
            Correct species name.

    Returns:
        Dictionary containing Top-1, Top-3, and Top-5 evaluation results.
    """
    if not predictions:
        return {
            "ground_truth": ground_truth,
            "predicted_species": None,
            "confidence": 0.0,
            "top1_correct": False,
            "top3_correct": False,
            "top5_correct": False,
        }

    top1_species, top1_confidence = predictions[0]

    return {
        "ground_truth": ground_truth,
        "predicted_species": top1_species,
        "confidence": round(top1_confidence, 2),
        "top1_correct": species_match(
            top1_species,
            ground_truth,
        ),
        "top3_correct": top_k_correct(
            predictions,
            ground_truth,
            k=3,
        ),
        "top5_correct": top_k_correct(
            predictions,
            ground_truth,
            k=5,
        ),
    }


def create_evaluation_record(
    image_name: str,
    model_name: str,
    predictions: list,
    ground_truth: str,
) -> dict:
    """
    Create a complete evaluation record for one image and model.

    Args:
        image_name:
            Name of the evaluated image.

        model_name:
            Name of the model producing the predictions.

        predictions:
            Ranked list of (species, confidence) tuples.

        ground_truth:
            Correct species name.

    Returns:
        Dictionary suitable for CSV export and later analysis.
    """
    evaluation = evaluate_prediction(
        predictions=predictions,
        ground_truth=ground_truth,
    )

    return {
        "Image Name": image_name,
        "Model": model_name,
        "Predicted Species": evaluation["predicted_species"],
        "Ground Truth": evaluation["ground_truth"],
        "Confidence (%)": evaluation["confidence"],
        "Correct": evaluation["top1_correct"],
        "Top-3 Correct": evaluation["top3_correct"],
        "Top-5 Correct": evaluation["top5_correct"],
    }


def calculate_accuracy(
    evaluation_results: list,
    metric: str,
) -> float:
    """
    Calculate accuracy for a specific evaluation metric.

    Args:
        evaluation_results: Evaluation records for the current experiment.
        metric: CSV/display field containing the correctness value.

    Returns:
        Accuracy as a percentage.
    """

    if not evaluation_results:
        return 0.0

    correct_count = sum(1 for result in evaluation_results if result.get(metric, False))

    return (correct_count / len(evaluation_results)) * 100


def summarize_evaluation_results(
    evaluation_results: list,
) -> list:
    """
    Calculate aggregate Top-1, Top-3, and Top-5 accuracy
    for each model.
    """

    if not evaluation_results:
        return []

    models = sorted({result["Model"] for result in evaluation_results})

    summaries = []

    for model_name in models:

        model_results = [
            result for result in evaluation_results if result["Model"] == model_name
        ]

        summaries.append(
            {
                "Model": model_name,
                "Number of Images": len(model_results),
                "Top-1 Accuracy (%)": round(
                    calculate_accuracy(
                        model_results,
                        "Correct",
                    ),
                    2,
                ),
                "Top-3 Accuracy (%)": round(
                    calculate_accuracy(
                        model_results,
                        "Top-3 Correct",
                    ),
                    2,
                ),
                "Top-5 Accuracy (%)": round(
                    calculate_accuracy(
                        model_results,
                        "Top-5 Correct",
                    ),
                    2,
                ),
            }
        )

    return summaries


def print_evaluation_summary(
    evaluation_summary: list,
) -> None:
    """Print aggregate evaluation metrics for the current experiment."""

    if not evaluation_summary:
        print("\nNo evaluation results available.")
        return

    print("\n")
    print("=" * 80)
    print("EXPERIMENT EVALUATION SUMMARY")
    print("=" * 80)

    for summary in evaluation_summary:
        print(f"\nModel: {summary['Model']}")
        print(f"Images evaluated: {summary['Number of Images']}")
        print(f"Top-1 Accuracy: " f"{summary['Top-1 Accuracy (%)']:.2f}%")
        print(f"Top-3 Accuracy: " f"{summary['Top-3 Accuracy (%)']:.2f}%")
        print(f"Top-5 Accuracy: " f"{summary['Top-5 Accuracy (%)']:.2f}%")

    print("=" * 80)
