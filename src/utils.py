import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


"""
Cerinta 2: sa se preproceseze 1,5 pct preprocesari si vizualizari + explicatie concluzii vizualizari
"""


def visualize_data(X_train_scaled, y_train, activity_names):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)

    plt.figure(figsize=(10, 8))
    for i, activity in enumerate(activity_names):
        mask = y_train == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=activity, alpha=0.7)

    plt.title('PCA Visualization of Human Activity Data')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('har_pca_visualization.png')

    # 2. Feature correlation heatmap (for first 20 features)
    plt.figure(figsize=(12, 10))
    corr_matrix = np.corrcoef(X_train_scaled[:, :20], rowvar=False)
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix (First 20 Features)')
    plt.tight_layout()
    plt.savefig('har_feature_correlation.png')

    # 3. Class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(y=y_train)
    plt.yticks(ticks=np.arange(len(activity_names)), labels=activity_names)
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.savefig('har_class_distribution.png')

    print("Data visualizations saved to disk.")


def preprocess_data(X_train, y_train, X_test, y_test, activity_names):
    # 1. Check for missing values
    print(f"Missing values in training data: {np.isnan(X_train).sum()}")

    # 2. Data scaling
    # Standard scaler for feature normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Check class distribution
    class_counts = np.bincount(y_train)
    class_distribution = pd.DataFrame({
        'Activity': activity_names,
        'Count': class_counts
    })
    print("Class distribution:")
    print(class_distribution)

    # 4. Create visualizations
    visualize_data(X_train_scaled, y_train, activity_names)

    return X_train_scaled, X_test_scaled, scaler


def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    """Plot confusion matrix with seaborn"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    return cm
