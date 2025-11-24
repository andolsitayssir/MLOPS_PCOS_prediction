import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score
import mlflow
import os

def load_data():
    X_train = pd.read_csv("data/processed/pcos_train.csv").values
    X_test = pd.read_csv("data/processed/pcos_test.csv").values
    y_train = pd.read_csv("data/processed/pcos_y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/pcos_y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    cv_score = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='f1').mean()
    return best_rf, grid_search.best_params_, cv_score

def train_logistic_regression(X_train, y_train):
    lg = LogisticRegression(max_iter=1000, random_state=0)
    lg.fit(X_train, y_train)
    cv_score = cross_val_score(lg, X_train, y_train, cv=5, scoring='f1').mean()
    return lg, cv_score

def train_svm(X_train, y_train):
    svm = SVC(kernel='linear', probability=True, random_state=0)
    svm.fit(X_train, y_train)
    cv_score = cross_val_score(svm, X_train, y_train, cv=5, scoring='f1').mean()
    return svm, cv_score

def train_knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(X_train, y_train)
    cv_score = cross_val_score(knn, X_train, y_train, cv=5, scoring='f1').mean()
    return knn, cv_score

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    if hasattr(model, "predict_proba"):
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    else:
        roc_auc = None
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))
    print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f}")
    return acc, f1, roc_auc, mse, rmse

def save_model(model, name):
    os.makedirs("app/models", exist_ok=True)
    path = f"app/models/{name}.pkl"
    joblib.dump(model, path, compress=3)
    return path

def main():
    X_train, X_test, y_train, y_test = load_data()
    mlflow.set_experiment("PCOS_Model_Training")
    with mlflow.start_run(run_name="model_training"):
        # Random Forest
        best_rf, rf_params, rf_cv = train_random_forest(X_train, y_train)
        mlflow.log_params({f"rf_{k}": v for k, v in rf_params.items()})
        mlflow.log_metric("rf_cv_f1", rf_cv)
        acc_rf, f1_rf, roc_auc_rf, mse_rf, rmse_rf = evaluate_model(best_rf, X_test, y_test, "Random Forest")
        mlflow.log_metric("rf_accuracy", acc_rf)
        mlflow.log_metric("rf_f1_score", f1_rf)
        mlflow.log_metric("rf_mse", mse_rf)
        mlflow.log_metric("rf_rmse", rmse_rf)
        if roc_auc_rf is not None:
            mlflow.log_metric("rf_roc_auc", roc_auc_rf)
        rf_path = save_model(best_rf, "best_rf")
        mlflow.log_artifact(rf_path)

        # Logistic Regression
        lg, lg_cv = train_logistic_regression(X_train, y_train)
        mlflow.log_metric("lg_cv_f1", lg_cv)
        acc_lg, f1_lg, roc_auc_lg, mse_lg, rmse_lg = evaluate_model(lg, X_test, y_test, "Logistic Regression")
        mlflow.log_metric("lg_accuracy", acc_lg)
        mlflow.log_metric("lg_f1_score", f1_lg)
        mlflow.log_metric("lg_mse", mse_lg)
        mlflow.log_metric("lg_rmse", rmse_lg)
        if roc_auc_lg is not None:
            mlflow.log_metric("lg_roc_auc", roc_auc_lg)
        lg_path = save_model(lg, "lg")
        mlflow.log_artifact(lg_path)

        # SVM
        svm, svm_cv = train_svm(X_train, y_train)
        mlflow.log_metric("svm_cv_f1", svm_cv)
        acc_svm, f1_svm, roc_auc_svm, mse_svm, rmse_svm = evaluate_model(svm, X_test, y_test, "SVM")
        mlflow.log_metric("svm_accuracy", acc_svm)
        mlflow.log_metric("svm_f1_score", f1_svm)
        mlflow.log_metric("svm_mse", mse_svm)
        mlflow.log_metric("svm_rmse", rmse_svm)
        if roc_auc_svm is not None:
            mlflow.log_metric("svm_roc_auc", roc_auc_svm)
        svm_path = save_model(svm, "svm")
        mlflow.log_artifact(svm_path)

        # KNN
        knn, knn_cv = train_knn(X_train, y_train)
        mlflow.log_metric("knn_cv_f1", knn_cv)
        acc_knn, f1_knn, roc_auc_knn, mse_knn, rmse_knn = evaluate_model(knn, X_test, y_test, "KNN")
        mlflow.log_metric("knn_accuracy", acc_knn)
        mlflow.log_metric("knn_f1_score", f1_knn)
        mlflow.log_metric("knn_mse", mse_knn)
        mlflow.log_metric("knn_rmse", rmse_knn)
        if roc_auc_knn is not None:
            mlflow.log_metric("knn_roc_auc", roc_auc_knn)
        knn_path = save_model(knn, "knn")
        mlflow.log_artifact(knn_path)

if __name__ == "__main__":
    main()