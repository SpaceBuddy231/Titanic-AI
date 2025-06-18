# Titanic AI 🚢

An artificial intelligence model to predict passenger survival on the Titanic disaster using machine learning.

## Overview

This project implements a complete machine learning pipeline to analyze the Titanic passenger dataset and predict survival outcomes. The model achieves **84% accuracy** using Random Forest classification with comprehensive data preprocessing and feature engineering.

## Dataset

The project uses the famous Titanic dataset containing:
- **PassengerId**: Unique identifier for each passenger
- **Survived**: Survival outcome (0 = No, 1 = Yes) - *Target variable*
- **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Sex**: Gender of passenger
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Passenger fare
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Machine Learning Pipeline

### 1. Data Preprocessing
- **Missing Value Imputation**: 
  - Age: filled with median value (28.0 years)
  - Embarked: filled with most frequent port (Southampton)
- **Feature Removal**: 
  - Cabin: dropped due to 77% missing values
  - Name & Ticket: removed as non-predictive features
- **Categorical Encoding**:
  - Sex: binary encoding (male=0, female=1)
  - Embarked: one-hot encoding with drop_first=True

### 2. Model Training
- **Algorithm**: Random Forest Classifier
- **Parameters**: 100 estimators, random_state=42
- **Train/Test Split**: 80/20 ratio
- **Cross-validation**: Built-in bootstrap sampling

### 3. Model Performance
- **Accuracy**: 84% on test set
- **Robustness**: Consistent performance across different runs
- **Interpretability**: Feature importance analysis available

## Requirements

```bash
pip install pandas scikit-learn
```

**Dependencies:**
- Python 3.13+
- pandas 2.3.0+
- scikit-learn 1.7.0+

## Usage

```bash
# Clone the repository
git clone https://github.com/SpaceBuddy231/Titanic-AI.git
cd Titanic-AI

# Run the model
python main.py
```

## Project Structure

```
Titanic-AI/
├── main.py          # Complete ML pipeline
├── data/
│   ├── train.csv    # Training dataset (891 passengers)
│   ├── test.csv     # Test dataset (418 passengers)
│   └── gender_submission.csv # Sample submission format
└── README.md        # Project documentation
```

## Key Features

✅ **Complete Data Preprocessing Pipeline**  
✅ **Missing Value Handling**  
✅ **Feature Engineering & Encoding**  
✅ **Machine Learning Model Training**  
✅ **Model Evaluation & Metrics**  
✅ **Clean, Documented Code**  

## Results

The model successfully processes the raw Titanic dataset and produces:
- Clean numerical features suitable for ML algorithms
- Zero missing values in the final dataset
- **84% prediction accuracy** on survival outcomes
- Robust performance with Random Forest classification

## Future Enhancements

Potential improvements for higher accuracy:
- Advanced feature engineering (family size, titles from names)
- Hyperparameter tuning with GridSearch
- Ensemble methods (Voting, Stacking)
- Cross-validation optimization
