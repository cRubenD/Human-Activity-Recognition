import numpy as np
import torch
from sklearn.metrics import recall_score

from src.models import DecisionTreeClassifierSimple

"""
Cerinta 4: sa se implementeze diverse metode de ensemble
"""


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y_batch.size(0)
        preds = outputs.argmax(dim=1)

        matches = torch.eq(preds, y_batch).sum().item()
        correct += matches
        total += y_batch.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def eval_model(model, loader, criterion, optimizer, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            running_loss += loss.item() * y_batch.size(0)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

            correct += torch.eq(preds, y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    recall = recall_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, recall, np.array(all_preds), np.array(all_labels)


def train_pytorch_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                        device, num_epochs, writer, model_name):
    """Train a PyTorch model with logging"""
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_recall, val_preds, val_labels = eval_model(model, val_loader, criterion, optimizer,
                                                                          device)

        # Update learning rate
        if scheduler:
            scheduler.step(val_loss)

        # Log to tensorboard
        writer.add_scalars(f'{model_name}/Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars(f'{model_name}/Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalar(f'{model_name}/Recall', val_recall, epoch)

        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Recall: {val_recall:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Could save model here if needed

    return model


def train_decision_tree(X_train, y_train, max_depth=5):
    """Train a decision tree classifier"""
    tree = DecisionTreeClassifierSimple(max_depth=max_depth)
    tree.fit(X_train, y_train)
    return tree


# Ensemble methods
def bagging_same_model(X_train, y_train, n_estimators=5, model_type='decision_tree', **model_params):
    """Bagging ensemble with the same model type"""
    models = []
    n_samples = len(y_train)

    for _ in range(n_estimators):
        # Bootstrap sampling
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap, y_bootstrap = X_train[idxs], y_train[idxs]

        # Create and train model
        if model_type == 'decision_tree':
            model = DecisionTreeClassifierSimple(**model_params)
            model.fit(X_bootstrap, y_bootstrap)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        models.append(model)

    return models


def bagging_different_model(model_classes, X_train, y_train):
    models = []
    for cls in model_classes:
        model = cls()
        model.fit(X_train, y_train)
        models.append(model)
    return models


def voting_ensemble(models, X, weights=None):
    all_probs = []

    for model in models:
        model_probs = model.predict_proba(X)
        all_probs.append(model_probs)

    # Apply weights if provided
    if weights is not None:
        for i, w in enumerate(weights):
            all_probs[i] = all_probs[i] * w

    # Average probabilities
    avg_probs = np.mean(all_probs, axis=0)

    # Get class predictions
    y_pred = np.argmax(avg_probs, axis=1)

    return y_pred, avg_probs


def calculate_model_uncertainty(probabilities):
    """Calculate model uncertainty (entropy) from probability distributions"""
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)

    # Calculate entropy: -sum(p_i * log(p_i))
    entropy = -np.sum(probabilities * np.log(probabilities), axis=1)

    # Normalize to [0, 1]
    max_entropy = -np.log(1.0 / probabilities.shape[1])  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy

    return normalized_entropy
