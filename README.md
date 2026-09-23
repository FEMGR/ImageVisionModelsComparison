# ImageVisionModelComparison 🌿🤖

A modular, extensible Python system for benchmarking and comparing pre-trained plant identification models side-by-side.

This repository implements the **Strategy Pattern** to support multiple underlying model architectures, including Hugging Face Vision Transformers and PyTorch ResNets, through a unified prediction interface.

The system provides an experimental workflow for:

* running multiple plant identification models on the same images,
* generating ranked Top-K species predictions,
* comparing predictions against user-provided ground truth,
* calculating Top-1, Top-3, and Top-5 accuracy,
* analyzing model confidence and alternative predictions,
* exporting detailed evaluation results as CSV files, and
* generating graphical visualizations for comparing model confidence.

The models currently included in the comparison are:

* **juppy44/plant-identification-2m-vit-b** — a ViT-Base plant identification model trained on approximately 2 million plant images representing around 14,000 species.
* **Pl@ntNet-300K ResNet-18** — a ResNet-18 model trained using the Pl@ntNet-300K dataset covering 1,081 plant species.

These models serve as **pre-trained baselines** for evaluating plant image classification within the broader Smart Urban Farming research project.

---

## 🚀 Key Features

* **Modular Strategy Architecture:** Plug in, test, and compare different plant identification models through a unified wrapper interface.

* **Multiple Model Architectures:** Supports both Vision Transformer and ResNet-based plant identification models.

* **Top-K Prediction:** Generates ranked Top-1, Top-3, and Top-5 species predictions with associated confidence percentages.

* **Ground-Truth Evaluation:** Allows the user to provide the expected species for each test image and automatically compares model predictions against the ground truth.

* **Species Name Normalization:** Handles common formatting differences in botanical names during evaluation, such as author abbreviations and missing whitespace.

* **Accuracy Analysis:** Calculates Top-1, Top-3, and Top-5 accuracy for each model in the current experiment.

* **Confidence Analysis:** Records prediction confidence and identifies alternative species when the highest-confidence prediction does not meet the configured confidence threshold.

* **Historical Confidence Tracking:** Maintains a historical record of model confidence across evaluation sessions for subsequent analysis.

* **Graphical Comparison:** Generates a confidence comparison graph showing the prediction confidence produced by each model for the evaluated images.

* **Descriptive Statistics:** Calculates confidence statistics including mean, median, minimum, maximum, and standard deviation for each model.

* **Batch Processing & File Dialogs:** Provides interactive image selection for evaluating multiple local images in a single experiment.

* **Automated Results Export:** Exports raw predictions, per-image evaluation results, aggregate accuracy summaries, and historical confidence data to the `./results/` directory.

* **Extensible Research Workflow:** Separates model implementations from prediction, evaluation, and analytics components so that additional models can be incorporated into future experiments.

---

## 📁 Project Directory Structure

```text
ImageVisionModelComparison/
│
├── core/
│   ├── analytics.py           # Historical confidence logging,
│   │                          # statistics, and graph generation
│   ├── evaluation.py          # Ground-truth comparison and
│   │                          # Top-K accuracy calculations
│   ├── formatters.py          # CSV export and result formatting
│   ├── ui_utils.py            # Interactive image/model selection
│   └── config.py              # Centralized configuration and
│                              # WEIGHTS_DIR paths
│
├── models/
│   ├── base_model.py          # Abstract strategy interface
│   ├── juppy_model.py         # Hugging Face ViT-B wrapper
│   └── plantnet_model.py      # Pl@ntNet-300K ResNet-18 wrapper
│
├── weights/
│   ├── juppy44/               # Local Hugging Face model weights
│   │                          # and configuration
│   └── plantnet300k/          # ResNet weights and species
│                              # metadata mapping files
│
├── results/
│   ├── plant_model_predictions.csv
│   ├── evaluation_results.csv
│   ├── evaluation_summary.csv
│   ├── model_confidence_history.csv
│   └── confidence_comparison_chart.png
│
└── main.py                    # Main evaluation workflow
```

### Output Files

The `results/` directory contains both **structured experimental data** and **visual analysis outputs**.

| Output                            | Purpose                                                       |
| --------------------------------- | ------------------------------------------------------------- |
| `plant_model_predictions.csv`     | Ranked Top-K predictions and confidence for each model        |
| `evaluation_results.csv`          | Per-image comparison between predictions and ground truth     |
| `evaluation_summary.csv`          | Aggregate Top-1, Top-3, and Top-5 accuracy                    |
| `model_confidence_history.csv`    | Historical prediction confidence across evaluation runs       |
| `confidence_comparison_chart.png` | Graphical comparison of model confidence for evaluated images |

The generated graph provides a visual representation of the confidence values produced by the different models, while the CSV files preserve the underlying experimental results for further analysis.

For example, the confidence comparison workflow can be represented as:

```text
Test Images
     │
     ▼
Model Predictions
     │
     ├───────────────┐
     ▼               ▼
Evaluation       Confidence
Metrics           Analysis
     │               │
     ▼               ▼
CSV Results       Statistics
                     │
                     ▼
              Confidence Graph
```


---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/FEMGR/ImageVisionModelsComparison.git
cd ImageVisionModelComparison
```

### 2. Install Dependencies

Install the project's Python dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Configure Model Weights & Metadata

The project expects model assets to be organized under the centralized `./weights/` directory.

#### Juppy44

Place the local files for:

**`juppy44/plant-identification-2m-vit-b`**

inside:

```text
./weights/juppy44/
```

The model is a Vision Transformer (ViT-Base) fine-tuned for plant species identification. The original model is published by `juppy44` on Hugging Face and uses the Google ViT-Base architecture as its base model.

Model source:

* `juppy44/plant-identification-2m-vit-b`
* Base architecture: `google/vit-base-patch16-224`
* Approximately 97.2M parameters
* Approximately 14,000 plant species

#### Pl@ntNet-300K

Place the pretrained ResNet-18 checkpoint:

```text
resnet18_weights_best_acc.tar
```

and the corresponding metadata files inside:

```text
./weights/plantnet300k/
```

Required mapping files:

```text
class_idx_to_species_id.json
plantnet300K_species_id_2_name.json
```

The official Pl@ntNet-300K repository provides these metadata files together with the pretrained model weights. The first mapping connects the neural-network class index to the Pl@ntNet species ID, while the second maps the species ID to its scientific name.

---

## 🏃 Usage

Run the main evaluation script:

```bash
python main.py
```

The system will:

1. Initialize the configured plant identification models.
2. Load their local pretrained weights.
3. Prompt the user to select one or more test images.
4. Process the images through each active model.
5. Display predicted scientific names and confidence scores.
6. Report alternative predictions when the confidence falls below the configured threshold.
7. Save the evaluation results under `./results/`.

This allows the different model architectures to be evaluated using the same input images and prediction interface.

---

## 📊 Model Comparison

The current implementation provides a common interface for comparing:

| Model                                   | Architecture | Training Data                         | Approx. Classes | Purpose                                     |
| --------------------------------------- | ------------ | ------------------------------------- | --------------: | ------------------------------------------- |
| `juppy44/plant-identification-2m-vit-b` | ViT-Base     | GBIF/iNaturalist-derived plant images |         ~14,000 | Broad plant species identification baseline |
| Pl@ntNet-300K                           | ResNet-18    | Pl@ntNet-300K                         |           1,081 | Plant identification baseline               |

The comparison is intended to support later experimentation with additional architectures and models.

In particular, this repository separates the **model implementation** from the **evaluation interface**, allowing additional vision models to be incorporated without changing the main evaluation workflow.

---

## 🔎 How Pl@ntNet-300K Mapping Works

The Pl@ntNet-300K ResNet-18 model produces numerical class indices. These indices are converted into scientific plant names through a two-step mapping process.

### 1. Class Index → Species ID

```text
class_idx_to_species_id.json
```

maps the model's output class index to the corresponding Pl@ntNet species ID.

### 2. Species ID → Scientific Name

```text
plantnet300K_species_id_2_name.json
```

maps the Pl@ntNet species ID to the corresponding scientific name.

For example:

```text
Model class index
        ↓
Species ID
        ↓
Scientific name
        ↓
Tagetes erecta L.
```

This mapping is necessary because the numerical output indices produced by the neural network do not directly represent scientific plant names.

---

## 🧪 Research Context

This repository is part of a broader research-oriented Smart Urban Farming project investigating the use of **multimodal machine learning and knowledge-based decision support for sustainable small-scale agriculture**.

The plant identification component focuses specifically on the **computer vision modality**. The current pretrained models provide baselines for plant image classification before further experimentation with custom-trained or fine-tuned models.

Future experiments may compare these pretrained baselines with models trained or fine-tuned using project-specific datasets.

The broader system is intended to combine visual information with other modalities such as:

* environmental and sensor measurements,
* weather information,
* plant metadata,
* spatial planting information, and
* knowledge-based relationships between plants, pests, soil, and growing conditions.

---

## 📚 Model & Dataset Attribution

### Juppy44 — Plant Identification 2M ViT-B

This project uses the pretrained:

**`juppy44/plant-identification-2m-vit-b`**

model published on Hugging Face.

The model is based on Google's `vit-base-patch16-224` architecture and was fine-tuned on approximately 2 million curated plant occurrences representing approximately 14,000 species. The model card identifies GBIF-derived, research-grade iNaturalist imagery as its training-data source.

Model:

[juppy44/plant-identification-2m-vit-b on Hugging Face](https://huggingface.co/juppy44/plant-identification-2m-vit-b?utm_source=chatgpt.com)

The associated dataset and its licensing information are documented separately by the dataset publisher:

[juppy44/gbif-plants-raw on Hugging Face](https://huggingface.co/datasets/juppy44/gbif-plants-raw?utm_source=chatgpt.com)

The model repository is listed under the Apache-2.0 license. The training images retain their original GBIF/iNaturalist licensing terms, so those terms should be considered separately when redistributing or creating derivative datasets.

### Pl@ntNet-300K

This project uses the pretrained **ResNet-18** model provided through the official Pl@ntNet-300K project.

Pl@ntNet-300K contains 306,146 plant images covering 1,081 species and was introduced in the NeurIPS 2021 Datasets and Benchmarks track. The official project repository provides the pretrained ResNet-18 weights and the metadata mappings used by this implementation.

Official repository:

[Pl@ntNet-300K — official GitHub repository](https://github.com/plantnet/PlantNet-300K?utm_source=chatgpt.com)

Publication:

> Garcin, C., Joly, A., Bonnet, P., Lombardo, J.-C., Affouard, A., Chouet, M., Servajean, M., Lorieul, T., & Salmon, J. (2021). *Pl@ntNet-300K: a plant image dataset with high label ambiguity and a long-tailed distribution*. NeurIPS 2021 Datasets and Benchmarks.

The official project repository provides the following citation for the work:

```bibtex
@inproceedings{plantnet-300k,
  author    = {Garcin, Camille and Joly, Alexis and Bonnet, Pierre and
               Lombardo, Jean-Christophe and Affouard, Antoine and
               Chouet, Mathias and Servajean, Maximilien and
               Lorieul, Titouan and Salmon, Joseph},
  booktitle = {NeurIPS Datasets and Benchmarks 2021},
  title     = {{Pl@ntNet-300K}: a plant image dataset with high label
               ambiguity and a long-tailed distribution},
  year      = {2021},
}
```

---

## ⚖️ Licensing & Attribution

The models and datasets used by this repository are third-party resources and remain subject to their respective licenses and terms.

This repository does not claim ownership of the pretrained model weights, datasets, source images, or associated metadata.

Users should consult the original model and dataset repositories before redistributing pretrained weights, datasets, or derivative resources.

---

## 🤝 Contributing

Contributions, additional model wrappers, evaluation methods, and performance improvements are welcome.

Potential extensions include:

* additional plant identification architectures,
* custom-trained CNN models,
* additional Vision Transformer models,
* regional or crop-specific models,
* standardized benchmark datasets,
* top-k accuracy evaluation,
* confusion-matrix analysis, and
* integration with the broader Smart Urban Farming AI pipeline.
