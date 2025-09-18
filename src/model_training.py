import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import joblib
from data_preprocessing import load_and_prepare_data

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    return f'results/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'

def train_logistic_regression(X_train, X_test, y_train, y_test, target_names):
    """Train Logistic Regression model"""
    with mlflow.start_run(run_name="Logistic_Regression"):
        # Model parameters
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Log parameters
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("max_iter", 1000)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Create and log confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_pred, target_names, "Logistic Regression")
        mlflow.log_artifact(cm_path)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = 'models/logistic_regression.pkl'
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Logistic Regression - Accuracy: {accuracy:.4f}")
        return accuracy, model

def train_random_forest(X_train, X_test, y_train, y_test, target_names):
    """Train Random Forest model"""
    with mlflow.start_run(run_name="Random_Forest"):
        # Model parameters
        n_estimators = 100
        max_depth = 10
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                     random_state=42)
        
        # Log parameters
        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", 42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Create and log confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_pred, target_names, "Random Forest")
        mlflow.log_artifact(cm_path)
        
        # Log feature importance
        feature_importance = model.feature_importances_
        mlflow.log_param("top_feature_importance", float(np.max(feature_importance)))
        
        # Save model
        model_path = 'models/random_forest.pkl'
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Random Forest - Accuracy: {accuracy:.4f}")
        return accuracy, model

def train_svm(X_train, X_test, y_train, y_test, target_names):
    """Train SVM model"""
    with mlflow.start_run(run_name="SVM"):
        # Model parameters
        kernel = 'rbf'
        C = 1.0
        model = SVC(kernel=kernel, C=C, random_state=42)
        
        # Log parameters
        mlflow.log_param("model_type", "SVM")
        mlflow.log_param("kernel", kernel)
        mlflow.log_param("C", C)
        mlflow.log_param("random_state", 42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Create and log confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_pred, target_names, "SVM")
        mlflow.log_artifact(cm_path)
        
        # Save model
        model_path = 'models/svm.pkl'
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"SVM - Accuracy: {accuracy:.4f}")
        return accuracy, model

def main():
    """Main training function"""
    # Set MLflow tracking URI
    # mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_tracking_uri("file:///C:/Users/DELL/Desktop/MLOps/MLOps-Assignment-1/mlruns")

    mlflow.set_experiment("Wine_Classification_Experiment")
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_prepare_data()
    
    print("Starting model training...")
    
    # Train models and collect results
    results = {}
    
    # Train Logistic Regression
    lr_accuracy, lr_model = train_logistic_regression(X_train, X_test, y_train, y_test, target_names)
    results['Logistic Regression'] = (lr_accuracy, lr_model)
    
    # Train Random Forest
    rf_accuracy, rf_model = train_random_forest(X_train, X_test, y_train, y_test, target_names)
    results['Random Forest'] = (rf_accuracy, rf_model)
    
    # Train SVM
    svm_accuracy, svm_model = train_svm(X_train, X_test, y_train, y_test, target_names)
    results['SVM'] = (svm_accuracy, svm_model)
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x][0])
    best_accuracy = results[best_model_name][0]
    best_model = results[best_model_name][1]
    
    print(f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    return best_model_name, best_model, best_accuracy

if __name__ == "__main__":
    main()