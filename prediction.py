import scipy.io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from tabulate import tabulate

data = scipy.io.loadmat("/content/mat1.mat")

print(data.keys())

mat1 = data['mat1']
mat1.shape
print(mat1)

df = pd.DataFrame(mat1)
print(df)

X = df.iloc[:-1]
y = df.iloc[-1]
X = X.T
print(X.shape)
print(y.shape)
print(X)
print(y)

k = 7
selector = SelectKBest(score_func=f_classif, k=k)
X_selected = selector.fit_transform(X, y)

selected_indices = selector.get_support(indices=True)
print("Selected features indices:", selected_indices)



df = pd.DataFrame(X_selected)
df['gender'] = np.where(y == 0, 'Women', 'Men')

descriptive_stats = df.describe()
descriptive_stats_gender = df.groupby('gender').describe()

plt.figure(figsize=(12, 8))
for i, feature in enumerate(df.columns[:-1]):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x='gender', y=feature, data=df, showfliers=False)
    plt.title(f'{feature}')
plt.tight_layout()
plt.show()

selected_features_table = tabulate(descriptive_stats, headers='keys', tablefmt='fancy_grid', floatfmt=".2f")
gender_table = tabulate(descriptive_stats_gender, headers='keys', tablefmt='fancy_grid', floatfmt=".2f")

print("Descriptive Statistics for Selected Features:")
print(selected_features_table)

print("\nDescriptive Statistics for Selected Features by Gender:")
print(gender_table)


X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

print(y_train.shape)
print(X_train.shape)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)
print("SVM Metrics:")
print("SVM Accuracy:", svm_accuracy)

print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1-score:", svm_f1)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)
print("KNN Metrics:")
print("KNN Accuracy:", knn_accuracy)

print("Precision:", knn_precision)
print("Recall:", knn_recall)
print("F1-score:", knn_f1)

svm_cm = confusion_matrix(y_test, svm_pred)

knn_cm = confusion_matrix(y_test, knn_pred)

print("Confusion Matrix for SVM:")
print(svm_cm)

print("\nConfusion Matrix for k-NN:")
print(knn_cm)

svm_cm = confusion_matrix(y_test, svm_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(svm_cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['positive', 'negative'],
            yticklabels=['positive', 'negative'])
plt.title('Confusion Matrix for SVM')
plt.show()

knn_cm = confusion_matrix(y_test, knn_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(knn_cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['positive', 'negative'],
            yticklabels=['positive', 'negative'])
plt.title('Confusion Matrix for k-NN')
plt.show()

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

svm_cv_scores = cross_val_score(svm, X_selected, y, cv=kf, scoring='accuracy')
svm_cv_accuracy = svm_cv_scores.mean()

knn_cv_scores = cross_val_score(knn, X_selected, y, cv=kf, scoring='accuracy')
knn_cv_accuracy = knn_cv_scores.mean()

svm_cv_precision = cross_val_score(svm, X_selected, y, cv=kf, scoring='precision').mean()
svm_cv_recall = cross_val_score(svm, X_selected, y, cv=kf, scoring='recall').mean()
svm_cv_f1 = cross_val_score(svm, X_selected, y, cv=kf, scoring='f1').mean()


knn_cv_precision = cross_val_score(knn, X_selected, y, cv=kf, scoring='precision').mean()
knn_cv_recall = cross_val_score(knn, X_selected, y, cv=kf, scoring='recall').mean()
knn_cv_f1 = cross_val_score(knn, X_selected, y, cv=kf, scoring='f1').mean()

#cross valiation SVM metrics
print("Cross validation SVM Metrics:")
print("Average Accuracy:", svm_cv_accuracy)
print("Average Precision:", svm_cv_precision)
print("Average Recall:", svm_cv_recall)
print("Average F1-score:", svm_cv_f1)

#cross valiation KNN metrics
print("\nCross validation KNN metrics:")
print("Average Accuracy:", knn_cv_accuracy)
print("Average Precision:", knn_cv_precision)
print("Average Recall:", knn_cv_recall)
print("Average F1-score:", knn_cv_f1)



svm_param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 10]
}

svm_grid_search = GridSearchCV(SVC(kernel='linear'), svm_param_grid, cv=5)
svm_grid_search.fit(X_selected, y)

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_selected)

svm_pca = SVC(kernel='linear')
svm_pca.fit(X_pca, y)

knn_param_grid = {
    'n_neighbors': [5, 10, 15],
    'weights': ['uniform', 'distance']
}

knn_grid_search = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5)
knn_grid_search.fit(X_selected, y)

knn_pca = KNeighborsClassifier(n_neighbors=10)
knn_pca.fit(X_pca, y)

print("Best parameters for SVM:", svm_grid_search.best_params_)
print("Best score for SVM:", svm_grid_search.best_score_)
print("Best parameters for KNN:", knn_grid_search.best_params_)
print("Best score for KNN:", knn_grid_search.best_score_)


data = scipy.io.loadmat('/content/all_epochs_final_LABELED.mat')
print(data.keys())

mat2 = data['all_epochs_final']
mat2.shape
print(mat2)

df2 = pd.DataFrame(mat2)
print(df2)

X = df2.iloc[:-1]
y = df2.iloc[-1]
X = X.T
print(X.shape)
print(y.shape)
print(X)
print(y)


k = 9
selector = SelectKBest(score_func=f_classif, k=k)
X_selected = selector.fit_transform(X, y)

selected_indices = selector.get_support(indices=True)
print("Selected features indices:", selected_indices)

df2 = pd.DataFrame(X_selected)
df2['Age'] = np.where(y == 0, 'Young', 'Old')

descriptive_stats2 = df2.describe()
descriptive_stats_age = df2.groupby('Age').describe()

plt.figure(figsize=(12, 8))
for i, feature in enumerate(df2.columns[:-1]):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x='Age', y=feature, data=df2, showfliers=False)
    plt.title(f'{feature}')
plt.tight_layout()
plt.show()


selected_features_table = tabulate(descriptive_stats, headers='keys', tablefmt='fancy_grid', floatfmt=".2f")
age_group_table = tabulate(descriptive_stats_age, headers='keys', tablefmt='fancy_grid', floatfmt=".2f")
print("Descriptive Statistics for Selected Features:")
print(selected_features_table)

print("\nDescriptive Statistics for Selected Features by Age Group:")
print(age_group_table)


X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

print(y_train.shape)
print(X_train.shape)


knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)
print("KNN Metrics:")
print("KNN Accuracy:", knn_accuracy)

print("Precision:", knn_precision)
print("Recall:", knn_recall)
print("F1-score:", knn_f1)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("Random Forest Metrics:")
print("Random Forest Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1-score:", rf_f1)

rf_cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Positive', 'Negative'],
            yticklabels=['Positive', 'Negative'])
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

knn_cm = confusion_matrix(y_test, knn_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(knn_cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Positive', 'Negative'],
            yticklabels=['Positive', 'Negative'])
plt.title('Confusion Matrix for k-NN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

knn = KNeighborsClassifier(n_neighbors=10)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

knn_cv_accuracy = cross_val_score(knn, X_selected, y, cv=5, scoring='accuracy').mean()
knn_cv_precision = cross_val_score(knn, X_selected, y, cv=5, scoring='precision').mean()
knn_cv_recall = cross_val_score(knn, X_selected, y, cv=5, scoring='recall').mean()
knn_cv_f1 = cross_val_score(knn, X_selected, y, cv=5, scoring='f1').mean()

rf_cv_accuracy = cross_val_score(rf, X_selected, y, cv=5, scoring='accuracy').mean()
rf_cv_precision = cross_val_score(rf, X_selected, y, cv=5, scoring='precision').mean()
rf_cv_recall = cross_val_score(rf, X_selected, y, cv=5, scoring='recall').mean()
rf_cv_f1 = cross_val_score(rf, X_selected, y, cv=5, scoring='f1').mean()

print("5-fold Cross-Validation Metrics:")
print("\nKNN Metrics:")
print("KNN Accuracy:", knn_cv_accuracy)
print("Precision:", knn_cv_precision)
print("Recall:", knn_cv_recall)
print("F1-score:", knn_cv_f1)

print("\nRandom Forest Metrics:")
print("Random Forest Accuracy:", rf_cv_accuracy)
print("Precision:", rf_cv_precision)
print("Recall:", rf_cv_recall)
print("F1-score:", rf_cv_f1)
