# main.py

import os
from typing import List

from models.juppy_model import JuppyModel
from models.plantnet_model import PlantNetModel
from models.juppy_extended_model import JuppyExtendedModel
from models.plantnet_extended_model import PlantNetExtendedModel
from models.custom_model import CustomModel
from models.base_model import BasePlantModel
from core.generate_report import build_word_report
from core.analytics import (
    log_prediction_result,
    plot_confidence_trends,
)

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

from core.benchmarking import (
    measure_inference_speed,
    print_benchmark_summary,
    export_benchmark_results,
)


def initialize_models() -> List[BasePlantModel]:
    """Initialize the default pretrained baseline models."""
    return [
        JuppyModel(),
        PlantNetModel(),
        JuppyExtendedModel(),
        PlantNetExtendedModel(),
    ]


def setup_custom_models(models: List[BasePlantModel]) -> List[BasePlantModel]:
    """
    Prompt the user to append custom local or Hugging Face models
    (ResNet, ViT, ConvNeXt, etc.) to the comparison pipeline.
    """
    print("\n" + "=" * 65)
    print("CUSTOM MODEL SELECTION (ResNet, ViT, ConvNeXt, Swin, etc.)")
    print("=" * 65)

    while True:
        use_custom = (
            input("Would you like to add a custom model to the evaluation? (y/n): ")
            .strip()
            .lower()
        )

        if use_custom != "y":
            break

        print("\nSelect custom model source type:")
        print("1. Local model folder (via File Dialog)")
        print(
            "2. Hugging Face Hub Model ID (e.g., 'microsoft/resnet-18', 'google/vit-base-patch16-224')"
        )

        choice = input("Enter choice (1/2): ").strip()

        if choice == "1":
            print("Opening file dialog to pick custom model directory...")
            custom_path = prompt_for_custom_model()

            if custom_path:
                display_name = os.path.basename(custom_path) or "Custom Local Model"
                custom_model = CustomModel(
                    name=f"Custom ({display_name})",
                    model_path_or_repo=custom_path,
                )
                models.append(custom_model)
                print(f"Added custom local model: {display_name}")
            else:
                print("No directory selected.")

        elif choice == "2":
            repo_id = input("Enter Hugging Face Repository ID: ").strip()
            if repo_id:
                model_name = repo_id.split("/")[-1]
                custom_model = CustomModel(
                    name=f"HF ({model_name})",
                    model_path_or_repo=repo_id,
                )
                models.append(custom_model)
                print(f"Added Hugging Face model: {repo_id}")
            else:
                print("Invalid Repository ID provided.")
        else:
            print("Invalid selection.")

        # Ask if the user wants to add another model architecture
        add_more = input("\nAdd another custom model? (y/n): ").strip().lower()
        if add_more != "y":
            break

    return models


def load_models(models: List[BasePlantModel]) -> List[BasePlantModel]:
    """Load all active models before evaluation begins."""
    print("\n" + "=" * 65)
    print("LOADING ALL ACTIVE MODELS")
    print("=" * 65)

    loaded_models = []
    for model in models:
        try:
            print(f"Loading {model.name}...")
            model.load()
            loaded_models.append(model)
        except Exception as exc:
            print(f"Failed to load {model.name}: {exc}")

    return loaded_models


def run_benchmarks(models: List[BasePlantModel], sample_image_path: str):
    """Run latency, FPS, and memory benchmarks across all loaded models."""
    print("\n" + "=" * 65)
    print("RUNNING HARDWARE & SPEED BENCHMARKS")
    print("=" * 65)

    benchmark_results = []
    for model in models:
        print(f"Benchmarking {model.name}...")
        results = measure_inference_speed(
            model=model,
            image_path=sample_image_path,
            warmup_runs=2,
            test_runs=10,
        )
        benchmark_results.append(results)

    # Print summary table to CLI
    print_benchmark_summary(benchmark_results)

    # Optional: Save benchmark metrics to CSV
    export_benchmark_results(
        benchmark_results,
        output_dir="./results",
    )

    return benchmark_results


def process_images(image_paths: List[str], models: List[BasePlantModel]):
    """
    Collect ground truth, run predictions across all selected architectures,
    and evaluate the predictions.
    """
    ground_truths = {}

    # ---------------------------------------------------------
    # 1. Collect ground-truth labels
    # ---------------------------------------------------------
    print("\nEnter ground-truth species for each image:")
    print("-" * 65)

    for index, img_path in enumerate(image_paths, start=1):
        img_name = os.path.basename(img_path)
        ground_truth = input(f"{index}. {img_name}\n   Ground truth species: ").strip()
        ground_truths[img_name] = ground_truth

    # ---------------------------------------------------------
    # 2. Run model predictions across all loaded backbones
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

                if ranked_predictions:
                    top_species, top_confidence = ranked_predictions[0]
                    log_prediction_result(
                        image_name=img_name,
                        model_name=model.name,
                        species=top_species,
                        confidence=top_confidence,
                    )

            except Exception as exc:
                print(f"Error with {model.name} on {img_name}: {exc}")
                predictions[model.name] = []

        all_predictions[img_name] = predictions

    return all_predictions, all_evaluation_results


def main():
    """Run the plant identification model comparison and benchmarking."""
    print("Initializing Plant Model Evaluator...")
    print("-" * 65)

    # 1. Initialize baseline models
    models = initialize_models()

    # 2. Add custom models (ResNet, ViT, ConvNeXt, etc.)
    models = setup_custom_models(models)

    # 3. Load all configured models
    models = load_models(models)

    if not models:
        print("No models were loaded successfully. Exiting program.")
        return

    benchmark_completed = False

    # 4. Continuous evaluation loop
    while True:
        print("\nWaiting for user to select images via file dialog...")
        image_paths = prompt_for_images()

        if not image_paths:
            print("No images selected.")
        else:
            # -------------------------------------------------
            # Run Benchmarks (Once, on the first batch)
            # -------------------------------------------------
            if not benchmark_completed:
                # Use the first image from the selected batch as reference input
                sample_image = image_paths[0]
                run_benchmarks(models=models, sample_image_path=sample_image)
                benchmark_completed = True

            # -------------------------------------------------
            # Process Image Batch Predictions & Accuracy
            # -------------------------------------------------
            print(f"\nProcessing {len(image_paths)} image(s)...")

            (
                all_predictions,
                all_evaluation_results,
            ) = process_images(
                image_paths=image_paths,
                models=models,
            )

            # Export and summarize batch accuracy results
            if all_evaluation_results:
                export_results_to_csv(
                    all_predictions,
                    output_dir="./results",
                )

                export_evaluation_results(
                    all_evaluation_results,
                    output_dir="./results",
                )

                evaluation_summary = summarize_evaluation_results(
                    all_evaluation_results
                )

                print_evaluation_summary(evaluation_summary)

                export_evaluation_summary(
                    evaluation_summary,
                    output_dir="./results",
                )
            else:
                print("\nNo evaluation results were generated for this batch.")

        # 5. Continue or finish prompt
        choice = (
            input("\nWould you like to predict another batch of images? (y/n): ")
            .strip()
            .lower()
        )

        if choice != "y":
            print(
                "\nExiting evaluation loop. "
                "Generating final confidence trend analytics..."
            )
            break

    # 6. Generate historical confidence analytics
    plot_confidence_trends()
    print("Generating Word document summary...")
    build_word_report()


if __name__ == "__main__":
    main()
