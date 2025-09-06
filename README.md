# Titanic Survival Prediction using K-Nearest Neighbors

A machine learning project that predicts passenger survival on the Titanic using K-Nearest Neighbors (KNN) algorithm 
with cross-validation and oversampling techniques to prevent overfitting and improve model performance.

## 📊 Project Overview

This project analyzes the famous Titanic dataset to predict passenger survival using the K-Nearest Neighbors algorithm. 
The implementation focuses on robust model evaluation through cross-validation and addresses class imbalance using oversampling techniques 
to create a reliable and generalizable predictive model.

## 🎯 Objectives

- **Survival Prediction**: Build a KNN classifier to predict Titanic passenger survival
- **Data Preprocessing**: Clean and prepare the dataset for optimal model performance
- **Overfitting Prevention**: Implement cross-validation and oversampling to improve generalization
- **Model Evaluation**: Assess performance using multiple metrics and visualization techniques
- **Class Separation Analysis**: Visualize how the model separates different classes

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms and evaluation
- **Matplotlib/Seaborn** - Data visualization
- **Imbalanced-learn** - Oversampling techniques (SMOTE)
- **Jupyter Notebook** - Interactive development environment

## 📁 Dataset

**Source**: `train.csv` from the Kaggle Titanic competition

**Dataset Details**:
- **Size**: 891 passenger records
- **Target Variable**: Survived (0 = No, 1 = Yes)
- **Key Features**:
  - **Age**: Passenger age in years
  - **Fare**: Ticket fare paid
  - **Sex**: Gender (male/female)
  - **Pclass**: Passenger class (1st, 2nd, 3rd)
  - **Embarked**: Port of embarkation (C, Q, S)
  - Additional features: SibSp, Parch, Ticket, Cabin, Name

## 🔧 Methodology

### 1. Data Preprocessing
- **Missing Value Handling**: Imputation strategies for Age, Embarked, and Cabin
- **Feature Engineering**: Create new features from existing ones
- **Categorical Encoding**: Convert categorical variables to numerical format
- **Feature Scaling**: Normalize numerical features for KNN algorithm
- **Outlier Detection**: Identify and handle extreme values

### 2. Class Imbalance Handling
- **Oversampling**: SMOTE (Synthetic Minority Oversampling Technique)
- **Class Distribution Analysis**: Before and after oversampling comparison
- **Stratified Sampling**: Maintain class proportions in train/test splits

### 3. K-Nearest Neighbors Implementation
- **Algorithm**: KNN classifier with optimized hyperparameters
- **Distance Metric**: Euclidean distance (with alternatives tested)
- **K-Value Optimization**: Grid search to find optimal number of neighbors
- **Weighted Voting**: Distance-based weighting for predictions

### 4. Cross-Validation Strategy
- **K-Fold Cross-Validation**: 5-fold or 10-fold validation
- **Stratified Cross-Validation**: Preserve class distribution in each fold
- **Performance Stability**: Reduce variance in model evaluation
- **Hyperparameter Tuning**: Cross-validated grid search

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn jupyter
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/titanic-survival-knn.git
cd titanic-survival-knn
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Titanic dataset (`train.csv`) and place it in the `data/` directory

### Usage

1. **Data Exploration and Preprocessing**:
```python
python scripts/data_preprocessing.py
```

2. **Model Training with Cross-Validation**:
```python
python scripts/knn_model.py
```

3. **Oversampling and Model Evaluation**:
```python
python scripts/model_evaluation.py
```

4. **Run Complete Pipeline**:
```python
python main.py
```

5. **Interactive Analysis**:
```bash
jupyter notebook notebooks/titanic_analysis.ipynb
```

## 📈 Results

### Model Performance
Classification Report:
              precision    recall  f1-score   support

Not Survived       0.85      0.85      0.85       110
    Survived       0.76      0.75      0.76        69

    accuracy                           0.82       179
   macro avg       0.81      0.80      0.80       179
weighted avg       0.82      0.82      0.82       179

Accuracy: 0.8156
Precision: 0.7647
Recall: 0.7536
F1-Score: 0.7591


### Key Insights
- **Most Important Features**: Fare, Age, Sex, Pclass
- **Optimal K Value**: X neighbors
- **Class Separation**: Clear decision boundaries in feature space
- **Feature Relationships**: Strong correlation between fare and survival

## 📊 Visualizations

The project generates comprehensive visualizations:
- **Data Distribution**: Histograms and box plots of key features
- **Class Separation**: 2D scatter plots showing decision boundaries
- **Confusion Matrix**: Model prediction accuracy breakdown
- **ROC Curve**: True positive vs false positive rates
- **Cross-Validation Scores**: Performance consistency across folds
- **Feature Importance**: Impact of different features on predictions

## 📁 Project Structure

```
titanic-survival-knn/
│
├── data/
│   ├── train.csv            # Original Titanic dataset
│   └── processed/           # Cleaned and preprocessed data
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── preprocessing.ipynb
│   ├── model_development.ipynb
│   └── results_analysis.ipynb
│
├── scripts/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── knn_model.py
│   ├── model_evaluation.py
│   └── visualization.py
│
├── models/
│   ├── knn_classifier.pkl   # Trained model
│   ├── scaler.pkl          # Feature scaler
│   └── oversampler.pkl     # SMOTE oversampler
│
├── results/
│   ├── plots/              # Generated visualizations
│   ├── reports/            # Performance reports
│   └── cv_results.json     # Cross-validation results
│
├── requirements.txt
├── main.py
└── README.md
```

## 🔍 Key Features

### Robust Data Preprocessing
- Intelligent missing value imputation
- Feature engineering from existing variables
- Proper categorical variable encoding
- Feature scaling for KNN algorithm

### Overfitting Prevention
- **Cross-Validation**: K-fold stratified cross-validation
- **Oversampling**: SMOTE to balance class distribution
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Performance Monitoring**: Training vs validation accuracy tracking

### Model Interpretability
- Decision boundary visualization
- Feature importance analysis
- Prediction confidence assessment
- Class separation plots

### Performance Evaluation
- Multiple evaluation metrics
- Statistical significance testing
- Cross-validation stability analysis
- Comprehensive reporting

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
jupyter>=1.0.0
```

## 🎯 Model Optimization Techniques

### Cross-Validation Strategy
```python
# Stratified K-Fold Cross-Validation
cv_scores = cross_val_score(knn_model, X_scaled, y, 
                           cv=StratifiedKFold(n_splits=10), 
                           scoring='accuracy')
```

### Oversampling Implementation
```python
# SMOTE Oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
```

### Hyperparameter Tuning
```python
# Grid Search with Cross-Validation
param_grid = {'n_neighbors': range(1, 31),
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan']}
```



## 🙏 Acknowledgments

- Kaggle for providing the Titanic dataset
- Scikit-learn community for machine learning tools
- Imbalanced-learn contributors for oversampling techniques
- Open source community for visualization libraries



*This project demonstrates best practices in machine learning including proper validation techniques, handling class imbalance, and preventing overfitting while maintaining model interpretability.*
