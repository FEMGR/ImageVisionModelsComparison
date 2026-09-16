# main.py

import os

from models.juppy_model import JuppyModel
from models.plantnet_model import PlantNetModel

from core.analytics import plot_confidence_trends

from core.evaluation import (
    create_evaluation_record,
    print_evaluation_summary,
    summarize_evaluation_results,
)

from core.formatters import (
    export_evaluation_results,
    export_evaluation_summary,
    export_results_to_csv,
)

from core.ui_utils import (
    prompt_for_custom_model,
    prompt_for_images,
)

# from models.custom_model import CustomModel


def initialize_models():
    """Initialize the pretrained models used in the comparison."""

    models = [
        JuppyModel(),
        PlantNetModel(),
    ]

    return models


def load_models(models):
    """Load all active models before evaluation begins."""

    for model in models:
        print(f"Loading {model.name}...")
        model.load()

    return models


def process_images(image_paths, models):
    """
    Collect ground truth, run predictions, and evaluate selected images.

    Ground truth is entered once for every selected image before
    model inference begins.
    """

    ground_truths = {}

    # ---------------------------------------------------------
    # 1. Collect ground-truth labels
    # ---------------------------------------------------------

    print("\nEnter ground-truth species for each image:")
    print("-" * 65)

    for index, img_path in enumerate(image_paths, start=1):
        img_name = os.path.basename(img_path)

        ground_truth = input(
            f"{index}. {img_name}\n" f"   Ground truth species: "
        ).strip()

        ground_truths[img_name] = ground_truth

    # ---------------------------------------------------------
    # 2. Run model predictions
    # ---------------------------------------------------------

    print("\n" + "=" * 65)
    print("Running model predictions...")
    print("=" * 65)

    all_predictions = {}
    all_evaluation_results = []

    for img_path in image_paths:

        img_name = os.path.basename(img_path)
        ground_truth = ground_truths[img_name]

        predictions = {}

        for model in models:

            try:
                # Generate ranked Top-K predictions
                ranked_predictions = model.predict(
                    img_path,
                    top_k=5,
                )

                predictions[model.name] = ranked_predictions

                # Evaluate predictions against ground truth
                evaluation_record = create_evaluation_record(
                    image_name=img_name,
                    model_name=model.name,
                    predictions=ranked_predictions,
                    ground_truth=ground_truth,
                )

                all_evaluation_results.append(evaluation_record)

            except Exception as exc:
                print(f"Error with {model.name} on {img_name}: " f"{exc}")

                predictions[model.name] = []

        all_predictions[img_name] = predictions

    return all_predictions, all_evaluation_results


def main():
    """Run the plant identification model comparison."""

    print("Initializing Plant Model Evaluator...")
    print("-" * 65)

    # =========================================================
    # 1. Initialize pretrained models
    # =========================================================

    models = initialize_models()

    # =========================================================
    # 2. Optional custom model
    # =========================================================

    print("\nDo you want to include a custom model in the comparison?")

    use_custom = (
        input("Enter 'y' to pick a custom model, " "or press Enter to skip: ")
        .strip()
        .lower()
    )

    if use_custom == "y":

        print("Waiting for user to select custom model " "via file dialog...")

        custom_model_path = prompt_for_custom_model()

        if custom_model_path:

            print(f"Selected custom model: " f"{custom_model_path}")

            # Enable this when CustomModel is implemented:
            #
            # custom_model = CustomModel(
            #     name="My Custom Model",
            #     model_path=custom_model_path,
            # )
            #
            # models.append(custom_model)

        else:

            print(
                "File dialog closed without selection. "
                "Proceeding with Juppy and PlantNet only."
            )

    else:

        print("Skipping custom model. " "Proceeding with Juppy and PlantNet only.")

    print("-" * 65)

    # =========================================================
    # 3. Load models
    # =========================================================

    load_models(models)

    # =========================================================
    # 4. Continuous evaluation loop
    # =========================================================

    while True:

        print("\nWaiting for user to select images " "via file dialog...")

        image_paths = prompt_for_images()

        if not image_paths:

            print("No images selected.")

        else:

            print(f"\nProcessing {len(image_paths)} image(s)...")

            (
                all_predictions,
                all_evaluation_results,
            ) = process_images(
                image_paths=image_paths,
                models=models,
            )

            # =================================================
            # Export and summarize current batch
            # =================================================

            if all_evaluation_results:

                # ---------------------------------------------
                # Raw Top-K predictions
                # ---------------------------------------------

                export_results_to_csv(
                    all_predictions,
                    output_dir="./results",
                )

                # ---------------------------------------------
                # Per-image evaluation results
                # ---------------------------------------------

                export_evaluation_results(
                    all_evaluation_results,
                    output_dir="./results",
                )

                # ---------------------------------------------
                # Aggregate evaluation metrics
                # ---------------------------------------------

                evaluation_summary = summarize_evaluation_results(
                    all_evaluation_results
                )

                # Display aggregate accuracy
                print_evaluation_summary(evaluation_summary)

                # Export aggregate accuracy
                export_evaluation_summary(
                    evaluation_summary,
                    output_dir="./results",
                )

            else:

                print("\nNo evaluation results were generated " "for this batch.")

        # =====================================================
        # 5. Continue or finish
        # =====================================================

        choice = (
            input("\nWould you like to predict another " "batch of images? (y/n): ")
            .strip()
            .lower()
        )

        if choice != "y":

            print(
                "\nExiting evaluation loop. "
                "Generating final confidence trend analytics..."
            )

            break

    # =========================================================
    # 6. Generate historical confidence analytics
    # =========================================================

    plot_confidence_trends()


if __name__ == "__main__":
    main()
