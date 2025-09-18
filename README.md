# MLOps Assignment 1: Wine Classification with MLflow Tracking

## 📋 Project Overview

This project demonstrates an end-to-end MLOps pipeline for wine classification using machine learning models with comprehensive experiment tracking, model comparison, and model registry functionality using MLflow.

### 🎯 Objectives Achieved
- ✅ GitHub version control and project structure
- ✅ Multiple ML model training and comparison
- ✅ MLflow experiment tracking and logging
- ✅ Model monitoring and registration
- ✅ Reproducible workflow documentation

## 🔧 Project Structure

```
mlops-assignment-1/
├── data/                          # Dataset storage
│   └── wine_dataset.csv          # Wine classification dataset
├── notebooks/                     # Jupyter notebooks for analysis
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py     # Data loading and preprocessing
│   ├── model_training.py         # Model training with MLflow logging
│   ├── model_evaluation.py       # Model comparison and visualization
│   └── model_registration.py     # Best model registration
├── models/                       # Trained model artifacts
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── svm.pkl
├── results/                      # Generated plots and results
│   ├── model_comparison.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   └── confusion_matrix_svm.png
├── main.py                       # Main pipeline execution script
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore patterns
└── README.md                    # Project documentation
```

## 📊 Dataset Description

**Dataset**: Wine Recognition Dataset from scikit-learn
- **Samples**: 178 wine samples
- **Features**: 13 chemical properties (alcohol, malic acid, ash, etc.)
- **Classes**: 3 wine categories (class 0, 1, 2)
- **Task**: Multi-class classification
- **Split**: 80% training, 20% testing
- **Preprocessing**: Standardized features using StandardScaler

## 🤖 Model Selection & Training

### Models Implemented

1. **Logistic Regression**
   - Parameters: max_iter=1000, random_state=42
   - Advantages: Fast, interpretable, good baseline

2. **Random Forest**
   - Parameters: n_estimators=100, max_depth=10, random_state=42
   - Advantages: Handles non-linear relationships, feature importance

3. **Support Vector Machine (SVM)**
   - Parameters: kernel='rbf', C=1.0, random_state=42
   - Advantages: Effective for high-dimensional data

### Training Process
- Each model trained with standardized features
- Cross-validation performed implicitly through train/test split
- All hyperparameters logged to MLflow
- Models saved as pickle files and MLflow artifacts

## 📈 Model Comparison Results

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 97% | 97% | 97% | 97% |
| Random Forest | 100% | 100% | 100% | 100% |
| SVM | 97% | 97% | 97% | 97% |

**Best Performing Model**: Random Forest

### Model Comparison Visualization
![alt text](image.png)

## 🔬 MLflow Experiment Tracking

### Experiment Configuration
- **Experiment Name**: Wine_Classification_Experiment
- **Tracking URI**: http://localhost:5000
- **Logged Components**:
  - Model parameters (hyperparameters)
  - Performance metrics (accuracy, precision, recall, F1-score)
  - Artifacts (confusion matrices, model files)
  - Model signatures and input examples

### MLflow UI Screenshots

#### Experiment Overview
![alt text](image-1.png)*Screenshot showing all experiment runs with metrics comparison*

#### Model Metrics Comparison
![MLflow Metrics](screenshots/mlflow_metrics.png)
*Screenshot showing detailed metrics comparison across models*

#### Individual Run Details
![MLflow Run Details](screenshots/mlflow_run_details.png)
*Screenshot showing detailed view of a single experiment run*

## 🏆 Model Registry & Deployment

### Best Model Registration
- **Model Name**: Wine_Classification_Best_Model
- **Version**: 1
- **Stage**: Production
- **Description**: Best performing model based on accuracy metrics

### Model Registry Screenshots
![Model Registry](screenshots/model_registry.png)
*Screenshot showing registered model in MLflow Model Registry*

![Model Version Details](screenshots/model_version_details.png)
*Screenshot showing model version details and metadata*

## 🚀 How to Run the Project

### Prerequisites
```bash
# Python 3.8+ required
# Clone the repository
git clone https://github.com/MeMorphLing/MLOps-Assignment-1.git
cd mlops-assignment-1
```

### Step 1: Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt
```

### Step 2: Start MLflow Server
```bash
# In a separate terminal, start MLflow server
mlflow server --host 127.0.0.1 --port 5000
```

### Step 3: Run the Pipeline
```bash
# Execute the complete pipeline
python main.py
```

### Step 4: Access MLflow UI
```bash
# Open browser and navigate to:
http://localhost:5000
```

## 📊 Monitoring & Results

### Key Insights
1. **Best Model Performance**: Random Forest with 100% Accuracy
2. **Feature Importance**: Available in Random Forest model logs
3. **Model Stability**: All models show consistent performance across runs
4. **Overfitting Check**: Test vs. train performance monitored

### Confusion Matrices
- Individual confusion matrices generated for each model
- Saved as artifacts in MLflow and results/ directory
- Visual comparison shows model strengths and weaknesses

## 🔄 Reproducibility

### Environment Reproducibility
- Fixed random seeds (random_state=42) for all models
- Exact package versions specified in requirements.txt
- Standardized data preprocessing pipeline

### Code Organization
- Modular code structure with separate functions
- Clear separation of data preprocessing, training, and evaluation
- Comprehensive logging and error handling

### Version Control
- Regular commits with clear messages
- .gitignore configured to exclude large files and temporary artifacts
- All source code and documentation tracked in Git

## 🛠 Technologies Used

- **Python 3.8+**: Core programming language
- **scikit-learn**: Machine learning models and metrics
- **MLflow**: Experiment tracking and model registry
- **pandas & numpy**: Data manipulation and analysis
- **matplotlib & seaborn**: Data visualization
- **GitHub**: Version control and collaboration

## 📝 Future Improvements

1. **Hyperparameter Tuning**: Implement GridSearch/RandomSearch
2. **Cross-Validation**: Add k-fold cross-validation
3. **Feature Engineering**: Create additional features
4. **Model Ensemble**: Combine best performing models
5. **Automated Deployment**: Add CI/CD pipeline
6. **Model Monitoring**: Implement drift detection

## 👨‍💻 Author

**Muhammad Hassasn Tahir **  
**FAST-NUCES | 22F - 3177**  
**Assignment Date**: 17/9/25

## 📄 License

This project is created for educational purposes as part of MLOps coursework.

---

