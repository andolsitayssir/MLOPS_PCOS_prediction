import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

FEATURE_COLUMNS = [
    "Follicle No. (L)",
    "Follicle No. (R)",
    "Skin darkening (Y/N)",
    "hair growth(Y/N)",
    "Weight gain(Y/N)",
    "Cycle(R/I)",
    "fast food (Y/N)",
    "Pimples(Y/N)",
    "AMH(ng/mL)",
]  # adjust to match your dataset headers

def generate_drift_report(
    data_path="data/processed/pcos_processed.csv",  # Use processed data
    output_path="reports/monitoring_report.html",
):
    print(" Starting Monitoring Job...")

    if not os.path.exists(data_path):
        print(f" Error: Data file not found at {data_path}")
        return

    try:
        # Load processed data (already clean)
        df = pd.read_csv(data_path)
        
        # Drop target column if present
        if 'PCOS (Y/N)' in df.columns:
            df = df.drop('PCOS (Y/N)', axis=1)
        
        # Drop rows with missing values
        df = df.dropna()

        if len(df) < 2:
            print(f" Error: Not enough data after cleanup (rows={len(df)}).")
            return

        # Split into reference (first 50%) and current (last 50%)
        mid_point = int(len(df) * 0.5)
        reference_data = df.iloc[:mid_point]
        current_data = df.iloc[mid_point:]

        if len(reference_data) == 0 or len(current_data) == 0:
            print(f" Error: reference/current split produced empty batches.")
            return

        print(f"   Reference samples: {len(reference_data)}")
        print(f"   Current samples: {len(current_data)}")

        report = Report(metrics=[DataDriftPreset()])
        print("   Calculating drift metrics...")
        report.run(reference_data=reference_data, current_data=current_data)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report.save_html(output_path)
        print(f"Report successfully generated: {output_path}")

    except Exception as e:
        print(f" Monitoring failed: {str(e)}")

if __name__ == "__main__":
    generate_drift_report()