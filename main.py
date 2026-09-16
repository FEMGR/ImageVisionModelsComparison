"""

"""
# main.py
import os
from core.ui_utils import prompt_for_custom_model, prompt_for_images
from core.formatters import print_results, export_results_to_csv
from models.juppy_model import JuppyModel
from models.plantnet_model import PlantNetModel


# from models.custom_model import CustomModel

def main():
    print("Initializing Plant Model Evaluator...")

    # 1. Initialize the base list with our standard models
    models = [
        JuppyModel(),
        PlantNetModel()
    ]

    # 2. Ask the user if they want to add a third custom model
    print("\nDo you want to include a custom model in the comparison?")
    use_custom = input("Enter 'y' to pick a custom model, or press Enter to skip: ").strip().lower()

    if use_custom == 'y':
        print("Waiting for user to select custom model via file dialog...")
        custom_model_path = prompt_for_custom_model()

        if custom_model_path:
            print(f"Selected custom model: {custom_model_path}")
            # Uncomment below once your CustomModel wrapper is built
            # custom_model = CustomModel(name="My Custom Model")
            # models.append(custom_model)
        else:
            print("File dialog closed without selection. Proceeding with Juppy and PlantNet only.")
    else:
        print("Skipping custom model. Proceeding with Juppy and PlantNet only.")

    print("-" * 65)

    # 3. Load all active models in the list
    for model in models:
        print(f"Loading {model.name}...")
        # If the custom model was added, give it the path. Otherwise, load normally.
        if "Custom" in model.name and 'custom_model_path' in locals():
            model.load(model_path=custom_model_path)
        else:
            model.load()

    # 4. Prompt user for images
    print("\nWaiting for user to select images via file dialog...")
    image_paths = prompt_for_images()
    if not image_paths:
        print("No images selected. Exiting.")
        return

    # 5. Run Inference Pipeline
    print(f"\nProcessing {len(image_paths)} image(s)...")
    all_evaluation_results = {}

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        predictions = {}

        for model in models:
            try:
                species, confidence = model.predict(img_path)
                predictions[model.name] = (species, confidence)
            except Exception as e:
                print(f"Error with {model.name} on {img_name}: {e}")
                predictions[model.name] = ("Error", 0.0)

        # Console output
        print_results(img_name, predictions)

        # Save to batch dictionary for CSV
        all_evaluation_results[img_name] = predictions

    # 6. Export to CSV automatically
    if all_evaluation_results:
        export_results_to_csv(all_evaluation_results, output_dir="./results")


if __name__ == "__main__":
    main()