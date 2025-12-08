import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

def generate_drift_report(data_path="data/PCOS_data_without_infertility.xlsx", output_path="reports/monitoring_report.html"):
    """
    Generates a Data Drift report using Evidently AI.
    Splits the dataset into reference and current batches to simulate production monitoring.
    """
    print("📈 Starting Monitoring Job...")
    
    # Check if data exists
    if not os.path.exists(data_path):
        # Failover to CSV if Excel not found (checking common locations)
        data_path_csv = data_path.replace(".xlsx", ".csv")
        if os.path.exists(data_path_csv):
            data_path = data_path_csv
        else:
            print(f"❌ Error: Data file not found at {data_path}")
            return

    try:
        # Load Data
        if data_path.endswith('.xlsx'):
            df = pd.read_excel(data_path)
        else:
            df = pd.read_csv(data_path)
            
        # Preprocessing matching training (basic cleanup for report)
        # Convert numeric columns loaded as objects if any
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna()

        # Simulate Monitoring Scenario
        # Reference: First 50% of data (Training historic)
        # Current: Last 50% of data (New production data)
        mid_point = int(len(df) * 0.5)
        reference_data = df.iloc[:mid_point]
        current_data = df.iloc[mid_point:]

        print(f"   Reference samples: {len(reference_data)}")
        print(f"   Current samples: {len(current_data)}")

        # Create Report
        report = Report(metrics=[
            DataDriftPreset(),
        ])

        print("   Calculating drift metrics...")
        report.run(reference_data=reference_data, current_data=current_data)

        # Save Report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report.save_html(output_path)
        print(f"✅ Report successfully generated: {output_path}")

    except Exception as e:
        print(f"❌ Monitoring failed: {str(e)}")

if __name__ == "__main__":
    generate_drift_report()
