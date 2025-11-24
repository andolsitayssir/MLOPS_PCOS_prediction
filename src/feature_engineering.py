import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import joblib
import logging
import os
import mlflow

logging.basicConfig(
    filename=os.path.join("logs", "feature_engineering.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

def select_features(data, target_col="PCOS (Y/N)", corr_threshold=0.2, remove_cols=None):
    logger.info("Sélection des features corrélées")
    corrmat = data.corr()
    corr_with_target = corrmat[target_col].abs().sort_values(ascending=False)
    top_features = corr_with_target[corr_with_target > corr_threshold].index.tolist()
    if target_col in top_features:
        top_features.remove(target_col)
    if remove_cols:
        top_features = [col for col in top_features if col not in remove_cols]
    logger.info(f"Features sélectionnées: {top_features}")
    return top_features

def main():
    mlflow.set_experiment("PCOS_Feature_Engineering")
    with mlflow.start_run(run_name="feature_engineering"):
        data = pd.read_csv("data/processed/pcos_processed.csv")
        top_features = select_features(data, remove_cols=["Weight (Kg)"])
        mlflow.log_param("corr_threshold", 0.2)
        mlflow.log_param("removed_columns", ["Weight (Kg)"])
        mlflow.log_param("selected_features", top_features)
        X = data[top_features]
        y = data["PCOS (Y/N)"]
        X = np.nan_to_num(X)
        X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)
        mlflow.log_param("smote_random_state", 42)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_resampled)
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)
        out_path = "data/processed/pcos_proc.csv"
        pd.DataFrame(X_pca).to_csv(out_path, index=False)
        mlflow.log_artifact(out_path)
        joblib.dump({"scaler": scaler, "pca": pca, "top_features": top_features}, "app/models/preprocessing.pkl", compress=3)
        mlflow.log_artifact("app/models/preprocessing.pkl")

if __name__ == "__main__":
    main()