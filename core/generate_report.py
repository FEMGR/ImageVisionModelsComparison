# core/generate_report.py

import os
import pandas as pd
from datetime import datetime
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


def populate_table_from_df(table, df):
    """Formats and writes a pandas DataFrame into a Word table."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"

    # Header Row
    hdr_cells = table.rows[0].cells
    for col_idx, col_name in enumerate(df.columns):
        hdr_cells[col_idx].text = str(col_name)
        hdr_cells[col_idx].paragraphs[0].runs[0].font.bold = True

    # Data Rows
    for _, row in df.iterrows():
        row_cells = table.add_row().cells
        for col_idx, val in enumerate(row):
            if isinstance(val, float):
                val_str = f"{val:.2f}" if abs(val) < 1000 else f"{val:.6f}"
            else:
                val_str = str(val)
            row_cells[col_idx].text = val_str


def build_word_report(results_dir: str = "./results", output_filename: str = None):
    """
    Dynamically loads evaluation and benchmark CSVs directly from results_dir
    and generates a formatted Word report timestamped with today's date.
    """
    if output_filename is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_filename = f"{date_str}_Model_Comparison_Report.docx"

    output_docx = os.path.join(results_dir, output_filename)
    os.makedirs(results_dir, exist_ok=True)

    doc = Document()

    NAVY = RGBColor(0x1B, 0x36, 0x5D)
    GRAY = RGBColor(0x55, 0x55, 0x55)

    def add_styled_heading(text, level):
        heading = doc.add_heading(text, level=level)
        for run in heading.runs:
            run.font.color.rgb = NAVY
            run.font.name = "Arial"
        return heading

    # Document Header
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Plant Vision Model Performance Analysis")
    run.font.size = Pt(22)
    run.font.bold = True
    run.font.color.rgb = NAVY

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = subtitle.add_run(
        f"Model Comparison & Benchmark Evaluation Report ({datetime.now().strftime('%B %d, %Y')})"
    )
    run_sub.font.size = Pt(12)
    run_sub.font.italic = True
    run_sub.font.color.rgb = GRAY

    doc.add_paragraph()

    # ---------------------------------------------------------
    # 1. Experiment Configuration
    # ---------------------------------------------------------
    add_styled_heading("1. Experiment Configuration", level=1)

    eval_res_csv = os.path.join(results_dir, "evaluation_results.csv")
    preds_csv = os.path.join(results_dir, "plant_model_predictions.csv")

    df_exp = None
    if os.path.exists(eval_res_csv):
        df_raw = pd.read_csv(eval_res_csv)
        cols = [c for c in ["Image", "Ground Truth"] if c in df_raw.columns]
        if len(cols) == 2:
            df_exp = df_raw[cols].drop_duplicates().reset_index(drop=True)
    elif os.path.exists(preds_csv):
        df_raw = pd.read_csv(preds_csv)
        cols = [c for c in ["Image", "Ground Truth"] if c in df_raw.columns]
        if len(cols) == 2:
            df_exp = df_raw[cols].drop_duplicates().reset_index(drop=True)

    num_images = len(df_exp) if df_exp is not None else 5

    doc.add_paragraph(
        f"Four active model variants were evaluated using {num_images} supplied plant images. "
        "Ground-truth species were entered manually for each image."
    )

    if df_exp is not None and not df_exp.empty:
        t_exp = doc.add_table(rows=1, cols=len(df_exp.columns))
        populate_table_from_df(t_exp, df_exp)

    doc.add_paragraph()

    # ---------------------------------------------------------
    # 2. Hardware & Speed Benchmark
    # ---------------------------------------------------------
    add_styled_heading("2. Hardware & Speed Benchmark", level=1)

    bench_csv = os.path.join(results_dir, "benchmark_summary.csv")
    if os.path.exists(bench_csv):
        df_bench = pd.read_csv(bench_csv)
        t_bench = doc.add_table(rows=1, cols=len(df_bench.columns))
        populate_table_from_df(t_bench, df_bench)

    doc.add_paragraph()

    # ---------------------------------------------------------
    # 3. Identification Evaluation
    # ---------------------------------------------------------
    add_styled_heading("3. Identification Evaluation", level=1)

    eval_summary_csv = os.path.join(results_dir, "evaluation_summary.csv")
    if os.path.exists(eval_summary_csv):
        df_eval = pd.read_csv(eval_summary_csv)
        t_eval = doc.add_table(rows=1, cols=len(df_eval.columns))
        populate_table_from_df(t_eval, df_eval)

    doc.add_paragraph()

    # ---------------------------------------------------------
    # 4. Confidence Analysis
    # ---------------------------------------------------------
    add_styled_heading("4. Confidence Analysis", level=1)

    conf_hist_csv = os.path.join(results_dir, "model_confidence_history.csv")
    df_conf = None

    if os.path.exists(conf_hist_csv):
        df_hist = pd.read_csv(conf_hist_csv)

        # Explicitly locate Model column
        model_col = next(
            (c for c in ["Model", "model_name", "ModelName"] if c in df_hist.columns),
            df_hist.columns[0],
        )

        # Explicitly locate Confidence column
        conf_col = next(
            (c for c in ["Confidence", "confidence", "Conf"] if c in df_hist.columns),
            None,
        )

        if conf_col is not None:
            # Clean and force numeric conversion to prevent string agg errors
            df_hist[conf_col] = pd.to_numeric(df_hist[conf_col], errors="coerce")
            df_hist = df_hist.dropna(subset=[conf_col])

            df_conf = (
                df_hist.groupby(model_col)[conf_col]
                .agg(
                    Count="count",
                    Mean="mean",
                    Median="median",
                    Min="min",
                    Max="max",
                    Std="std",
                )
                .reset_index()
            )

    if df_conf is not None and not df_conf.empty:
        t_conf = doc.add_table(rows=1, cols=len(df_conf.columns))
        populate_table_from_df(t_conf, df_conf)

    chart_path = os.path.join(results_dir, "confidence_comparison_chart.png")
    if os.path.exists(chart_path):
        doc.add_paragraph()
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(chart_path, width=Inches(6.0))

        caption = doc.add_paragraph(
            "Mean Confidence Chart\nFigure 1. Reconstructed from the mean confidence "
            "statistics reported in the experiment output. This is not the original per-image confidence comparison PNG."
        )
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.runs[0].font.size = Pt(9.5)
        caption.runs[0].font.italic = True

    doc.add_paragraph()

    # ---------------------------------------------------------
    # 5. Directly Reported Results
    # ---------------------------------------------------------
    add_styled_heading("5. Directly Reported Results", level=1)

    if os.path.exists(eval_summary_csv):
        df_eval = pd.read_csv(eval_summary_csv)
        for _, row in df_eval.iterrows():
            m = row.iloc[0]
            t1, t3, t5 = (
                row.get("Top-1", ""),
                row.get("Top-3", ""),
                row.get("Top-5", ""),
            )
            if t1 == t3 == t5:
                doc.add_paragraph(
                    f"{m} reported {t1} Top-1, Top-3, and Top-5 accuracy.",
                    style="List Bullet",
                )
            else:
                doc.add_paragraph(
                    f"{m} reported {t1} Top-1 accuracy and {t3} Top-3 and {t5} Top-5 accuracy.",
                    style="List Bullet",
                )

    if os.path.exists(bench_csv):
        df_bench = pd.read_csv(bench_csv)
        bench_parts = [
            f"{row.get('Latency (ms)', row.iloc[1])} ms/{row.get('FPS', row.iloc[2])} FPS for {row.iloc[0]}"
            for _, row in df_bench.iterrows()
        ]
        doc.add_paragraph(
            f"The benchmark reported latency/FPS of {', '.join(bench_parts)}.",
            style="List Bullet",
        )

    doc.add_paragraph()

    # ---------------------------------------------------------
    # 6. Interpretation Notes
    # ---------------------------------------------------------
    add_styled_heading("6. Interpretation Notes", level=1)

    notes = [
        f"The evaluation set contained only {num_images} images, "
        "so these accuracy figures describe this specific experiment rather than "
        "general model performance.",
        "The confidence statistics are based on the historical confidence log and"
        " therefore contain different record counts from the five-image evaluation.",
        "Confidence is not equivalent to accuracy.",
        "The reported 100.00% mean confidence for PlantNet-300K + Local Plants "
        "is a confidence-log statistic; its reported Top-1 accuracy for "
        "this experiment was 20.00%.",
    ]
    for note in notes:
        doc.add_paragraph(note, style="List Bullet")

    doc.add_paragraph()

    # ---------------------------------------------------------
    # 7. Experiment Output Files
    # ---------------------------------------------------------
    add_styled_heading("7. Experiment Output Files", level=1)

    files = [
        "./results/benchmark_summary.csv",
        "./results/plant_model_predictions.csv",
        "./results/evaluation_results.csv",
        "./results/evaluation_summary.csv",
        "./results/model_confidence_history.csv",
        "./results/confidence_comparison_chart.png",
    ]
    for f in files:
        doc.add_paragraph(f, style="List Bullet")

    doc.save(output_docx)
    print(f"\nReport successfully generated and saved to: {output_docx}")


if __name__ == "__main__":
    build_word_report()
