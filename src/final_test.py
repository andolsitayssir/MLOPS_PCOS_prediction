"""
Re-entraîner le meilleur modèle sur TRAIN+VALIDATION
Tester UNE FOIS sur TEST
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import mlflow
import mlflow.sklearn

def main():
    print("🎯 PHASE FINALE: Test sur le test set")
    
    # Charger info meilleur modèle
    with open("metrics/best_model_info.json") as f:
        info = json.load(f)
    
    best_model_name = info['best_model']
    print(f"✅ Meilleur modèle: {best_model_name}")
    
    # Charger le modèle
    best_model = joblib.load(f"models/candidates/{best_model_name}.pkl")
    
    # Combiner train + validation
    X_train = pd.read_csv("data/processed/pcos_train.csv").values
    X_val = pd.read_csv("data/processed/pcos_val.csv").values
    y_train = pd.read_csv("data/processed/pcos_y_train.csv").values.ravel()
    y_val = pd.read_csv("data/processed/pcos_y_val.csv").values.ravel()
    
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    print(f" Données combinées: {len(y_combined)} samples")
    
    # Re-entraîner sur train+val
    print(" Re-training sur train+validation...")
    best_model.fit(X_combined, y_combined)
    
    # Charger test
    X_test = pd.read_csv("data/processed/pcos_test.csv").values
    y_test = pd.read_csv("data/processed/pcos_y_test.csv").values.ravel()
    
    # TEST FINAL
    print("\nTesting sur test set...")
    y_test_pred = best_model.predict(X_test)
    
    final_metrics = {
        'test_accuracy': float(accuracy_score(y_test, y_test_pred)),
        'test_f1_score': float(f1_score(y_test, y_test_pred)),
        'test_roc_auc': float(roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))
    }
    
    print("\n RÉSULTATS FINAUX:")
    print(f"Accuracy: {final_metrics['test_accuracy']:.4f}")
    print(f"F1 Score: {final_metrics['test_f1_score']:.4f}")
    print(f"ROC AUC: {final_metrics['test_roc_auc']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Logger dans MLflow
    mlflow.set_experiment("PCOS_Final_Model")
    with mlflow.start_run(run_name="Final_Test"):
        mlflow.log_param("model_type", best_model_name)
        mlflow.log_metrics(final_metrics)
        mlflow.sklearn.log_model(best_model, "final_model")
    
    # Sauvegarder modèle final
    joblib.dump(best_model, "app/models/best_model.pkl")
    
    # Sauvegarder métriques (Ajouter le nom pour le JSON)
    metrics_to_save = final_metrics.copy()
    metrics_to_save['model_name'] = best_model_name
    
    with open("metrics/final_test_metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=4)
    
    print(f"\n✅ Modèle final sauvegardé: app/models/best_model.pkl")

if __name__ == "__main__":
    main()  