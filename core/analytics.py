# core/analytics.py

import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


def log_prediction_result(
    image_name,
    model_name,
    species,
    confidence,
    log_dir="./results",
):
    """Append prediction details to the historical evaluation log."""

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        "model_confidence_history.csv",
    )

    file_exists = os.path.exists(log_file)

    with open(
        log_file,
        mode="a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(
                [
                    "Timestamp",
                    "ImageName",
                    "ModelName",
                    "PredictedSpecies",
                    "ConfidencePercent",
                ]
            )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        writer.writerow(
            [
                timestamp,
                image_name,
                model_name,
                species,
                f"{confidence:.2f}",
            ]
        )


def plot_confidence_trends(
    log_path="./results/model_confidence_history.csv",
):
    """
    Generate a grouped bar chart comparing model confidence
    for each evaluated image.
    """

    if not os.path.exists(log_path):
        print("No historical logs found yet. " "Run some predictions first!")
        return

    df = pd.read_csv(log_path)

    if df.empty:
        print("Historical log is empty.")
        return

    df["ConfidencePercent"] = pd.to_numeric(
        df["ConfidencePercent"],
        errors="coerce",
    )

    # Group by image and model.
    # If an image has been evaluated multiple times,
    # use the mean confidence.
    plot_df = df.pivot_table(
        index="ImageName",
        columns="ModelName",
        values="ConfidencePercent",
        aggfunc="mean",
    )

    print("\n--- Model Confidence Statistics ---")

    stats = df.groupby("ModelName")["ConfidencePercent"].agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
            "std",
        ]
    )

    print(stats.to_string())
    print("-" * 40)

    # -------------------------------
    # Grouped bar chart
    # -------------------------------

    ax = plot_df.plot(
        kind="bar",
        figsize=(12, 6),
        width=0.8,
    )

    ax.set_title(
        "Model Confidence Comparison by Image",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Image")
    ax.set_ylabel("Confidence (%)")

    ax.set_ylim(0, 100)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.6,
    )

    ax.legend(title="Models")

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    chart_path = "./results/confidence_comparison_chart.png"

    plt.savefig(
        chart_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"📊 Confidence comparison chart " f"successfully saved to {chart_path}")

    plt.show()
