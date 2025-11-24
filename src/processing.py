import pandas as pd
import logging
import os
import mlflow

logging.basicConfig(
    filename=os.path.join("logs", "data_processing.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

def load_data(filepath, sheet_name):
    logger.info("Chargement des données")
    data = pd.read_excel(filepath, sheet_name=sheet_name)
    data.columns = data.columns.str.strip()
    return data

def clean_data(data):
    logger.info("Nettoyage des colonnes inutiles et doublons")
    data.drop(["Unnamed: 44", "Sl. No", "Patient File No."], axis=1, inplace=True, errors="ignore")
    data = data.drop_duplicates()
    return data

def impute_missing(data):
    logger.info("Imputation des valeurs manquantes")
    if data["Marraige Status (Yrs)"].isnull().any():
        data["Marraige Status (Yrs)"] = data["Marraige Status (Yrs)"].fillna(data["Marraige Status (Yrs)"].mean())
    for col in ["AMH(ng/mL)", "II    beta-HCG(mIU/mL)"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data

def categorical_to_numeric(data):
    logger.info("variables catégorielles en numériques")
    categ = data.select_dtypes(include=["object", "bool"]).columns.tolist()
    for col in categ:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data

def main():
    mlflow.set_experiment("PCOS_Data_Processing")
    with mlflow.start_run(run_name="data_cleaning"):
        data = load_data("data/raw/PCOS_data_without_infertility.xlsx", "Full_new")
        data = clean_data(data)
        data = impute_missing(data)
        data = categorical_to_numeric(data)
        os.makedirs("data/processed", exist_ok=True)
        out_path = "data/processed/pcos_processed.csv"
        data.to_csv(out_path, index=False)
        mlflow.log_artifact(out_path)
        mlflow.log_param("missing_imputation", "mean")
        mlflow.log_param("columns_dropped", ["Unnamed: 44", "Sl. No", "Patient File No."])
        logger.info("Préprocessing terminé avec succès")

if __name__ == "__main__":
    main()