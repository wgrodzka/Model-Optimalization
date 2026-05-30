# Model Optimization Project

## Overview

This project demonstrates comprehensive machine learning pipeline optimization using artificial datasets. The primary goal is to build, tune, and evaluate classification models with a focus on achieving robust performance through systematic hyperparameter optimization and feature engineering.

## Project Structure

```
Model-Optimalization/
├── README.md                 # Project documentation
├── report.pdf                # Project report and findings
├── .git/                     # Git version control
└── code/                     # Main project folder
    ├── code.ipynb                      # Main Jupyter notebook with complete pipeline
    ├── artificial_train_data.csv       # Training features
    ├── artificial_train_labels.csv     # Training labels
    ├── artificial_test_data.csv        # Test features
    └── artificial_prediction.txt       # Model predictions on test data
```

## Key Features

### Data Preprocessing
- **Scaling**: Comparison of StandardScaler and RobustScaler for feature normalization
- **Feature Selection**: SelectKBest with mutual information scoring to identify the most relevant features
- **Feature Engineering**: Advanced dimensionality reduction and variance-based filtering

### Model Training & Optimization

The pipeline incorporates multiple machine learning algorithms and optimization techniques:

1. **Gradient Boosting Classifier**
   - Multi-stage hyperparameter tuning
   - Optimization of n_estimators, learning_rate, max_depth, and subsample parameters

2. **Random Forest Classifier**
   - Systematic parameter optimization
   - Feature importance analysis

3. **Class Imbalance Handling**
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - ADASYN (Adaptive Synthetic Sampling)
   - SMOTETomek (combined approach)

4. **Cross-Validation**
   - Stratified K-Fold with 5 splits
   - Balanced accuracy scoring for imbalanced datasets

### Performance Metrics

- **Balanced Accuracy Score**: Primary evaluation metric for imbalanced classification
- **ROC-AUC Score**: Additional performance evaluation
- **Train/Test Split**: 85% training, 15% testing with stratification

## Technologies & Libraries

- **Python 3.x**
- **scikit-learn**: Machine learning algorithms and utilities
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **matplotlib & seaborn**: Data visualization
- **imbalanced-learn**: Handling imbalanced datasets
- **joblib**: Parallel processing for GridSearchCV

## Workflow Steps

### Stage 1: Initial Optimization
Select the best scaler and number of features (k) for SelectKBest using GridSearchCV on Gradient Boosting.

### Stage 2: GradientBoosting Fine-tuning
Optimize core hyperparameters:
- Number of estimators
- Learning rate
- Maximum depth
- Subsample ratio

### Subsequent Stages
Advanced optimization including:
- Feature selection method comparison (RFE, SequentialFeatureSelector)
- Additional hyperparameter refinement
- Model ensemble techniques
- Validation on test set

## Running the Project

### Prerequisites
```bash
pip install pandas scikit-learn imbalanced-learn matplotlib seaborn numpy joblib
```

### Execution
1. Navigate to the `code/` folder
2. Open `code.ipynb` in Jupyter Notebook or JupyterLab
3. Execute cells sequentially from top to bottom
4. Review interim results and score comparisons at each stage
5. Models are evaluated on both training and test sets
6. Predictions are generated and saved to `artificial_prediction.txt`

## Results

### Performance Achieved
- **Final Result on External Test Data**: **0.87** (Balanced Accuracy Score)
- **Best Model Configuration**: Optimized Gradient Boosting Classifier with:
  - RobustScaler for feature normalization
  - SelectKBest with 40 selected features
  - n_estimators: 500
  - learning_rate: 0.05
  - max_depth: 5
  - subsample: 0.8

### Output Files
- **Predictions**: Model outputs stored in `code/artificial_prediction.txt`
- **Performance Metrics**: Balanced accuracy scores tracked across optimization stages
- **Detailed Report**: See `report.pdf` for comprehensive findings and analysis

## Key Insights

- Feature selection significantly impacts model performance by reducing noise and computational overhead
- Robust scaling helps handle outliers in artificial datasets
- Balanced accuracy is essential for imbalanced classification problems
- Strategic combination of oversampling techniques improves minority class detection
- Systematic hyperparameter tuning yields substantial performance improvements

## Author

Wiktoria Grodzka

## Notes

This project uses **artificial data** for demonstration purposes. The pipeline is fully reproducible and can be adapted to real-world datasets by modifying the data loading section and adjusting parameters as needed.
