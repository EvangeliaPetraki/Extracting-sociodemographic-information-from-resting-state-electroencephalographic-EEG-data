# Extracting sociodemographic information from resting state electroencephalographic EEG data

This Python script conducts a machine learning analysis of EEG data using Scikit-Learn, Seaborn, Matplotlib, and Pandas. It processes data stored in MATLAB .mat files, performing feature selection, exploratory data analysis, and classification with SVM, k-NN, and Random Forest classifiers. The script also implements cross-validation and grid search for model evaluation and tuning.

**Libraries**: scipy, numpy, pandas, scikit-learn, matplotlib, seaborn, tabulate

## Dataset

The script processes EEG data stored in MATLAB .mat files. It uses:
* mat1.mat: Contains gender-labeled EEG data.
* all_epochs_final_LABELED.mat: Contains age-labeled EEG data.

Each dataset includes EEG features and respective labels.

## Script Breakdown

1. **Data Loading and Preparation**: 
This loads the data from a .mat file into a DataFrame for analysis. The labels are stored in the last row of the DataFrame (y), while the features are in all preceding rows (X).

2. **Feature Selection**:
The SelectKBest method selects the k best features based on ANOVA F-values to reduce dimensionality:

3. **Exploratory Data Analysis (EDA)**: 
Box plots are generated to compare the selected features across gender or age groups
Statistical summaries (mean, median, etc.) are also provided for each feature by group.

4. **Model Training and Evaluation**: 
The script trains and evaluates SVM, k-NN, and Random Forest classifiers
    * SVM: Support Vector Machine with linear kernel
    * k-NN: k-Nearest Neighbors
    * Random Forest: Ensemble classifier

    Each model’s performance is evaluated using accuracy, precision, recall, and F1-score

5. **Cross-Validation**: 
To ensure model robustness, 5-fold cross-validation is performed
The script reports the average accuracy, precision, recall, and F1-score across folds.

6. **Hyperparameter Tuning**: 
Grid search with cross-validation is applied to find the optimal hyperparameters for SVM and k-NN

7. **Principal Component Analysis (PCA)**: 
PCA reduces the dimensionality of the selected features, retaining five principal components. The reduced features are used to re-train SVM and k-NN models

8. **Confusion Matrices**: 
Confusion matrices are displayed to visualize true positives, false positives, true negatives, and false negatives for each model

## Output
The script outputs performance metrics, cross-validation results, and the best parameters identified by grid search. Box plots, confusion matrices, and descriptive statistics are also provided.
