import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_and_prepare_data():
    """Load and prepare the Wine dataset"""
    # Load wine dataset
    wine = load_wine()
    
    # Create DataFrame
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    
    # Save raw data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/wine_dataset.csv', index=False)
    
    # Prepare features and target
    X = wine.data
    y = wine.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, wine.feature_names, wine.target_names

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_prepare_data()
    print("Data loaded and preprocessed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")