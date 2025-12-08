"""
Entraîner modèles sur TRAIN, évaluer sur VALIDATION
Sélectionner le meilleur modèle basé sur validation
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
import joblib
import json
import os
import yaml

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def load_data():
    X_train = pd.read_csv("data/processed/pcos_train.csv").values
    X_val = pd.read_csv("data/processed/pcos_val.csv").values
    y_train = pd.read_csv("data/processed/pcos_y_train.csv").values.ravel()
    y_val = pd.read_csv("data/processed/pcos_y_val.csv").values.ravel()
    return X_train, X_val, y_train, y_val

def train_and_evaluate_rf(X_train, y_train, X_val, y_val, params):
    print("🔄 Training Random Forest...")
    param_grid = {
        'n_estimators': params['rf_n_estimators'],
        'max_depth': params['rf_max_depth'],
        'min_samples_split': params['rf_min_samples_split'],
        'min_samples_leaf': params['rf_min_samples_leaf']
    }
    rf = RandomForestClassifier(random_state=params['rf_random_state'])
    grid = GridSearchCV(rf, param_grid, cv=params['cv_folds'], scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_rf = grid.best_estimator_
    y_val_pred = best_rf.predict(X_val)
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'f1_score': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, best_rf.predict_proba(X_val)[:, 1])
    }
    
    print(f"✅ RF - F1: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    return best_rf, grid.best_params_, metrics

def train_and_evaluate_lr(X_train, y_train, X_val, y_val, params):
    print("🔄 Training Logistic Regression...")
    lr = LogisticRegression(max_iter=params['lg_max_iter'], random_state=params['lg_random_state'])
    lr.fit(X_train, y_train)
    
    y_val_pred = lr.predict(X_val)
    metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'f1_score': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, lr.predict_proba(X_val)[:, 1])
    }
    
    print(f"✅ LR - F1: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    return lr, metrics

def train_and_evaluate_svm(X_train, y_train, X_val, y_val, params):
    print("🔄 Training SVM...")
    svm = SVC(kernel=params['svm_kernel'], probability=params['svm_probability'], 
              random_state=params['svm_random_state'])
    svm.fit(X_train, y_train)
    
    y_val_pred = svm.predict(X_val)
    metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'f1_score': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, svm.predict_proba(X_val)[:, 1])
    }
    
    print(f"✅ SVM - F1: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    return svm, metrics

def train_and_evaluate_knn(X_train, y_train, X_val, y_val, params):
    print("🔄 Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=params['knn_n_neighbors'])
    knn.fit(X_train, y_train)
    
    y_val_pred = knn.predict(X_val)
    metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'f1_score': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, knn.predict_proba(X_val)[:, 1])
    }
    
    print(f"✅ KNN - F1: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    return knn, metrics

def main():
    params = load_params()['model_training']
    X_train, X_val, y_train, y_val = load_data()
    
    mlflow.set_experiment("PCOS_Model_Selection")
    
    os.makedirs("models/candidates", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    all_results = {}
    
    # Random Forest
    with mlflow.start_run(run_name="RandomForest_Validation"):
        rf, rf_params, rf_metrics = train_and_evaluate_rf(X_train, y_train, X_val, y_val, params)
        mlflow.log_params({f"rf_{k}": v for k, v in rf_params.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in rf_metrics.items()})
        mlflow.sklearn.log_model(rf, "model")
        joblib.dump(rf, "models/candidates/RandomForest.pkl")
        all_results['RandomForest'] = rf_metrics
    
    # Logistic Regression
    with mlflow.start_run(run_name="LogisticRegression_Validation"):
        lr, lr_metrics = train_and_evaluate_lr(X_train, y_train, X_val, y_val, params)
        mlflow.log_metrics({f"val_{k}": v for k, v in lr_metrics.items()})
        mlflow.sklearn.log_model(lr, "model")
        joblib.dump(lr, "models/candidates/LogisticRegression.pkl")
        all_results['LogisticRegression'] = lr_metrics
    
    # SVM
    with mlflow.start_run(run_name="SVM_Validation"):
        svm, svm_metrics = train_and_evaluate_svm(X_train, y_train, X_val, y_val, params)
        mlflow.log_metrics({f"val_{k}": v for k, v in svm_metrics.items()})
        mlflow.sklearn.log_model(svm, "model")
        joblib.dump(svm, "models/candidates/SVM.pkl")
        all_results['SVM'] = svm_metrics
    
    # KNN
    with mlflow.start_run(run_name="KNN_Validation"):
        knn, knn_metrics = train_and_evaluate_knn(X_train, y_train, X_val, y_val, params)
        mlflow.log_metrics({f"val_{k}": v for k, v in knn_metrics.items()})
        mlflow.sklearn.log_model(knn, "model")
        joblib.dump(knn, "models/candidates/KNN.pkl")
        all_results['KNN'] = knn_metrics
    
    # Sauvegarder résultats validation
    with open("metrics/validation_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    # Trouver meilleur modèle
    best_model_name = max(all_results, key=lambda x: all_results[x][params['primary_metric']])
    print(f"\n MEILLEUR MODÈLE: {best_model_name}")
    print(f"   {params['primary_metric']}: {all_results[best_model_name][params['primary_metric']]:.4f}")
    
    # Sauvegarder info du meilleur
    best_info = {
        "best_model": best_model_name,
        "metric": params['primary_metric'],
        "score": all_results[best_model_name][params['primary_metric']],
        "all_results": all_results
    }
    with open("metrics/best_model_info.json", "w") as f:
        json.dump(best_info, f, indent=4)

if __name__ == "__main__":
    main()