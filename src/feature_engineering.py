import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import joblib
import logging
import os
import mlflow
import yaml

logging.basicConfig(
    filename=os.path.join("logs", "feature_engineering.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

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
    params = load_params()
    
    with mlflow.start_run(run_name="feature_engineering_3way"):
        # 1. Charger données nettoyées
        data = pd.read_csv("data/processed/pcos_processed.csv")
        
        # 2. Sélection features
        top_features = select_features(data, remove_cols=["Weight (Kg)"])
        X = data[top_features]
        y = data["PCOS (Y/N)"]
        X = np.nan_to_num(X)
        
        mlflow.log_param("corr_threshold", 0.2)
        mlflow.log_param("selected_features", len(top_features))
        
        # 3. SPLIT 3-WAY (train/val/test) AVANT transformations
        logger.info("Split : train/val/test")
        
        # Premier split: séparer test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=params['feature_engineering']['test_size'],
            random_state=params['feature_engineering']['split_random_state'],
            stratify=y
        )
        
        # Deuxième split: séparer train et validation
        val_ratio = params['feature_engineering']['test_size'] / (1 - params['feature_engineering']['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=params['feature_engineering']['split_random_state'],
            stratify=y_temp
        )
        print(f"✅ Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"✅ Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"✅ Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        print(f" Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f" Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f" Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("test_samples", len(X_test))
        
        # 4. SMOTE SEULEMENT sur train
        # SMOTE sur train, pas sur val et test
        logger.info("Applying SMOTE on training set only")
        smote = SMOTE(random_state=params['feature_engineering']['smote_random_state'])
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        mlflow.log_param("train_after_smote", len(X_train_resampled))
        print(f" Train après SMOTE: {len(X_train_resampled)} samples")
        
        # 5. Scaling: FIT sur train, TRANSFORM sur val et test
        # on fit sur trin pour éviter le data leakage qui est un problème de surapprentissage
        # on transforme sur val et test 
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 6. PCA: FIT sur train, TRANSFORM sur val et test
        pca = PCA(n_components=params['feature_engineering']['pca_variance'])
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        mlflow.log_param("pca_components", pca.n_components_)
        print(f" PCA components: {pca.n_components_}")
        
        # 7. Sauvegarder les 3 sets
        os.makedirs("data/processed", exist_ok=True)
        
        pd.DataFrame(X_train_pca).to_csv("data/processed/pcos_train.csv", index=False)
        print("💾 Sauvegarde des fichiers...")
        pd.DataFrame(X_val_pca).to_csv("data/processed/pcos_val.csv", index=False)
        print("✅ pcos_val.csv sauvegardé!")  
        pd.DataFrame(X_test_pca).to_csv("data/processed/pcos_test.csv", index=False)
        
        pd.DataFrame(y_train_resampled).to_csv("data/processed/pcos_y_train.csv", index=False)
        pd.DataFrame(y_val).to_csv("data/processed/pcos_y_val.csv", index=False)
        pd.DataFrame(y_test).to_csv("data/processed/pcos_y_test.csv", index=False)
        
        # 8. Sauvegarder preprocessing pipeline
        joblib.dump({"scaler": scaler, "pca": pca, "top_features": top_features}, 
                   "app/models/preprocessing.pkl", compress=3)
        
        print(" Feature engineering avec 3-way split terminé!")

if __name__ == "__main__":
    main()