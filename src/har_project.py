"""

Ensemble Learning Project: Human Activity Recognition (HAR)
https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones

"""


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import recall_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data import load_har_data, HAR_Dataset
from src.experiments import experiment_with_batch_size, experiment_with_optimizers, experiment_with_learning_rates
from models import LogisticRegression, MLPModel
from trainers import train_pytorch_model, train_decision_tree, bagging_same_model, voting_ensemble, \
    calculate_model_uncertainty
from utils import preprocess_data, plot_confusion_matrix

CONFIG = {
    'data_dir': '../',
    'batch_sizes': [32, 64, 128],
    'learning_rates': [1e-2, 1e-3, 1e-4],
    'optimizers': ['Adam', 'SGD'],
    'scheduler': 'ReduceLROnPlateau',
    'k_folds': 5,
    'num_epochs': 50,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_classes': 6,
    'seed': 42,
    'log_dir': './runs/har_ensemble',
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])


# Main routine
def main():
    print(f"Using device: {CONFIG['device']}")

    # Load data
    print("Loading data...")
    X_train, y_train, X_test, y_test, activity_names = load_har_data(CONFIG['data_dir'])

    # Preprocess data
    print("Preprocessing data...")
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, y_train, X_test, y_test, activity_names)

    # Split training data into train and validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=CONFIG['seed']
    )

    input_dim = X_train.shape[1]
    num_classes = len(activity_names)
    print(f"Input dimensions: {input_dim}, Number of classes: {num_classes}")

    # Create TensorBoard writer
    writer = SummaryWriter(logdir=CONFIG['log_dir'])

    # Experiment with batch sizes
    print("\nExperimenting with batch sizes...")
    batch_results = experiment_with_batch_size(
        X_tr, y_tr, X_val, y_val, input_dim, num_classes,
        CONFIG['device'], CONFIG['batch_sizes'], writer
    )

    # Experiment with optimizers
    print("\nExperimenting with optimizers...")
    optimizer_results = experiment_with_optimizers(
        X_tr, y_tr, X_val, y_val, input_dim, num_classes, CONFIG['device'], writer
    )

    # Experiment with learning rates
    print("\nExperimenting with learning rates...")
    lr_results = experiment_with_learning_rates(
        X_tr, y_tr, X_val, y_val, input_dim, num_classes,
        CONFIG['device'], CONFIG['learning_rates'], writer
    )

    # Select best batch size based on experiments
    best_batch_size = max(batch_results, key=lambda bs: batch_results[bs]['val_acc'])
    print(f"\nBest batch size: {best_batch_size}")

    # Create datasets and dataloaders with best batch size
    train_ds = HAR_Dataset(X_tr, y_tr)
    val_ds = HAR_Dataset(X_val, y_val)
    test_ds = HAR_Dataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_ds, batch_size=best_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=best_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=best_batch_size, shuffle=False)

    # Train logistic regression model
    print("\nTraining Logistic Regression...")
    logreg_model = LogisticRegression(input_dim, num_classes).to(CONFIG['device'])
    logreg_criterion = nn.CrossEntropyLoss()
    logreg_optimizer = optim.Adam(logreg_model.parameters(), lr=1e-3)
    logreg_scheduler = ReduceLROnPlateau(logreg_optimizer, mode='min', factor=0.1, patience=5)

    logreg_model = train_pytorch_model(
        logreg_model, train_loader, val_loader, logreg_criterion,
        logreg_optimizer, logreg_scheduler, CONFIG['device'],
        CONFIG['num_epochs'], writer, 'LogisticRegression'
    )

    # Train MLP model
    print("\nTraining MLP...")
    mlp_model = MLPModel(input_dim, [128, 64], num_classes)
    mlp_criterion = nn.CrossEntropyLoss()
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=1e-3)
    mlp_scheduler = ReduceLROnPlateau(mlp_optimizer, mode='min', factor=0.1, patience=5)

    mlp_model = train_pytorch_model(
        mlp_model, train_loader, val_loader, mlp_criterion,
        mlp_optimizer, mlp_scheduler, CONFIG['device'],
        CONFIG['num_epochs'], writer, 'MLP'
    )

    # Train decision tree
    print("\nTraining Decision Tree...")
    tree_model = train_decision_tree(X_tr, y_tr, max_depth=8)

    # Evaluate decision tree on validation set
    tree_preds = tree_model.predict(X_val)
    tree_acc = accuracy_score(y_val, tree_preds)
    tree_recall = recall_score(y_val, tree_preds, average='macro')
    print(f"Decision Tree - Val Accuracy: {tree_acc:.4f} | Val Recall: {tree_recall:.4f}")

    # Log decision tree metrics
    writer.add_scalar('DecisionTree/Accuracy', tree_acc, 0)
    writer.add_scalar('DecisionTree/Recall', tree_recall, 0)

    # Implement bagging with decision trees
    print("\nImplementing bagging with decision trees...")
    bagged_trees = bagging_same_model(
        X_tr, y_tr, n_estimators=10, model_type='decision_tree', max_depth=5
    )

    # Evaluate bagged trees on validation set
    bagged_tree_preds, bagged_tree_probs = voting_ensemble(bagged_trees, X_val)
    bagged_tree_acc = accuracy_score(y_val, bagged_tree_preds)
    bagged_tree_recall = recall_score(y_val, bagged_tree_preds, average='macro')
    print(f"Bagged Trees - Val Accuracy: {bagged_tree_acc:.4f} | Val Recall: {bagged_tree_recall:.4f}")

    # Log bagged tree metrics
    writer.add_scalar('BaggedTrees/Accuracy', bagged_tree_acc, 0)
    writer.add_scalar('BaggedTrees/Recall', bagged_tree_recall, 0)

    # Create ensemble of different models
    print("\nCreating ensemble of different models...")

    # Prepare all models for ensemble
    models = [logreg_model, mlp_model, tree_model]

    # Evaluate each model on test set
    print("\nEvaluating individual models on test set...")
    model_accuracies = []
    model_predictions = []

    # Logistic Regression evaluation
    logreg_test_preds = logreg_model.predict(X_test_scaled)
    logreg_test_acc = accuracy_score(y_test, logreg_test_preds)
    logreg_test_recall = recall_score(y_test, logreg_test_preds, average='macro')
    print(f"Logistic Regression - Test Accuracy: {logreg_test_acc:.4f} | Test Recall: {logreg_test_recall:.4f}")
    model_accuracies.append(logreg_test_acc)
    model_predictions.append(logreg_test_preds)

    # MLP evaluation
    mlp_test_preds = mlp_model.predict(X_test_scaled)
    mlp_test_acc = accuracy_score(y_test, mlp_test_preds)
    mlp_test_recall = recall_score(y_test, mlp_test_preds, average='macro')
    print(f"MLP - Test Accuracy: {mlp_test_acc:.4f} | Test Recall: {mlp_test_recall:.4f}")
    model_accuracies.append(mlp_test_acc)
    model_predictions.append(mlp_test_preds)

    # Decision Tree evaluation
    tree_test_preds = tree_model.predict(X_test_scaled)
    tree_test_acc = accuracy_score(y_test, tree_test_preds)
    tree_test_recall = recall_score(y_test, tree_test_preds, average='macro')
    print(f"Decision Tree - Test Accuracy: {tree_test_acc:.4f} | Test Recall: {tree_test_recall:.4f}")
    model_accuracies.append(tree_test_acc)
    model_predictions.append(tree_test_preds)

    # Equal-weight voting ensemble
    print("\nEvaluating equal-weight voting ensemble...")
    equal_ensemble_preds, equal_ensemble_probs = voting_ensemble(models, X_test_scaled)
    equal_ensemble_acc = accuracy_score(y_test, equal_ensemble_preds)
    equal_ensemble_recall = recall_score(y_test, equal_ensemble_preds, average='macro')
    print(f"Equal-weight Ensemble - Test Accuracy: {equal_ensemble_acc:.4f} | Test Recall: {equal_ensemble_recall:.4f}")

    # Weighted voting ensemble (weighted by validation accuracy)
    print("\nEvaluating weighted voting ensemble...")
    # Normalize accuracies to use as weights
    weights = np.array(model_accuracies) / np.sum(model_accuracies)
    print(f"Model weights: {weights}")

    weighted_ensemble_preds, weighted_ensemble_probs = voting_ensemble(models, X_test_scaled, weights=weights)
    weighted_ensemble_acc = accuracy_score(y_test, weighted_ensemble_preds)
    weighted_ensemble_recall = recall_score(y_test, weighted_ensemble_preds, average='macro')
    print(
        f"Weighted Ensemble - Test Accuracy: {weighted_ensemble_acc:.4f} | Test Recall: {weighted_ensemble_recall:.4f}")

    # Calculate model uncertainty
    print("\nCalculating model uncertainty...")
    uncertainty = calculate_model_uncertainty(weighted_ensemble_probs)
    avg_uncertainty = np.mean(uncertainty)
    print(f"Average model uncertainty: {avg_uncertainty:.4f}")

    # Create confusion matrices
    print("\nGenerating confusion matrices...")
    cm_logreg = plot_confusion_matrix(y_test, logreg_test_preds, activity_names, "Logistic Regression Confusion Matrix")
    plt.savefig("confusion_matrix_logreg.png")

    cm_mlp = plot_confusion_matrix(y_test, mlp_test_preds, activity_names, "MLP Confusion Matrix")
    plt.savefig("confusion_matrix_mlp.png")

    cm_tree = plot_confusion_matrix(y_test, tree_test_preds, activity_names, "Decision Tree Confusion Matrix")
    plt.savefig("confusion_matrix_tree.png")

    cm_ensemble = plot_confusion_matrix(y_test, weighted_ensemble_preds, activity_names,
                                        "Weighted Ensemble Confusion Matrix")
    plt.savefig("confusion_matrix_ensemble.png")

    # Log confusion matrices to TensorBoard as images
    def plot_to_tensor(cm, class_names, title):
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return fig

    # Create figures for TensorBoard
    fig_logreg = plot_to_tensor(cm_logreg, activity_names, "Logistic Regression Confusion Matrix")
    fig_mlp = plot_to_tensor(cm_mlp, activity_names, "MLP Confusion Matrix")
    fig_tree = plot_to_tensor(cm_tree, activity_names, "Decision Tree Confusion Matrix")
    fig_ensemble = plot_to_tensor(cm_ensemble, activity_names, "Weighted Ensemble Confusion Matrix")

    # Log figures to TensorBoard
    writer.add_figure('ConfusionMatrix/LogisticRegression', fig_logreg)
    writer.add_figure('ConfusionMatrix/MLP', fig_mlp)
    writer.add_figure('ConfusionMatrix/DecisionTree', fig_tree)
    writer.add_figure('ConfusionMatrix/WeightedEnsemble', fig_ensemble)

    # Close TensorBoard writer
    writer.close()

    print("\nExperiment completed successfully!")


if __name__ == '__main__':
    main()
