from torch import nn, optim
from torch.utils.data import DataLoader

from src.data import HAR_Dataset
from models import LogisticRegression, MLPModel
from trainers import train_one_epoch, eval_model

"""
Cerinta 5: sa se experimenteze cu batch-size-ul, mica explicatie ce observati daca il mariti sau micsorati pentru fiecare model= 0.5 pct
sa se experimenteze pentru Logistic si MLPs cu diversi optimizatori 1 pct (minim sgd si adam)
sa se experimenteze cu diverse LR + explicatii (LR+LR scheduler clasa din pytorch pentru Logistic regression si MLPs, la decision nu aveti ce face) 1 pct
"""


def experiment_with_batch_size(X_train, y_train, X_val, y_val, input_dim, num_classes, device, batch_sizes, writer):
    """Experiment with different batch sizes"""
    results = {}

    for batch_size in batch_sizes:
        print(f"\nExperimenting with batch size: {batch_size}")

        # Create dataloaders
        train_ds = HAR_Dataset(X_train, y_train)
        val_ds = HAR_Dataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Create model
        model = LogisticRegression(input_dim, num_classes).to(device)

        # Training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Train for a few epochs
        num_epochs = 10
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_recall, _, _ = eval_model(model, val_loader, criterion, optimizer, device)

            # Log to tensorboard
            writer.add_scalars(f'BatchSize/{batch_size}/Loss', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalars(f'BatchSize/{batch_size}/Accuracy', {'train': train_acc, 'val': val_acc}, epoch)

        # Record final results
        results[batch_size] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss
        }

        print(f"Final results - Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return results


def experiment_with_optimizers(X_train, y_train, X_val, y_val, input_dim, num_classes, device, writer):
    """Experiment with different optimizers"""
    optimizers = {
        'SGD': optim.SGD,
        'Adam': optim.Adam
    }

    results = {}

    # Create dataloaders
    train_ds = HAR_Dataset(X_train, y_train)
    val_ds = HAR_Dataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # Experiment with each optimizer
    for opt_name, opt_class in optimizers.items():
        print(f"\nExperimenting with optimizer: {opt_name}")

        # Create model
        model = MLPModel(input_dim, [100, 50], num_classes).to(device)

        # Training parameters
        criterion = nn.CrossEntropyLoss()

        # Create optimizer with appropriate parameters
        if opt_name == 'SGD':
            optimizer = opt_class(model.parameters(), lr=1e-2, momentum=0.9)
        else:
            optimizer = opt_class(model.parameters(), lr=1e-3)

        # Train for a few epochs
        num_epochs = 10
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_recall, _, _ = eval_model(model, val_loader, criterion, optimizer, device)

            # Log to tensorboard
            writer.add_scalars(f'Optimizer/{opt_name}/Loss', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalars(f'Optimizer/{opt_name}/Accuracy', {'train': train_acc, 'val': val_acc}, epoch)

        # Record final results
        results[opt_name] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss
        }

        print(f"Final results - Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return results


def experiment_with_learning_rates(X_train, y_train, X_val, y_val, input_dim, num_classes, device, learning_rates,
                                   writer):
    """Experiment with different learning rates"""
    results = {}

    # Create dataloaders
    train_ds = HAR_Dataset(X_train, y_train)
    val_ds = HAR_Dataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # Experiment with each learning rate
    for lr in learning_rates:
        print(f"\nExperimenting with learning rate: {lr}")

        # Create model
        model = LogisticRegression(input_dim, num_classes).to(device)

        # Training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train for a few epochs
        num_epochs = 10
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_recall, _, _ = eval_model(model, val_loader, criterion, optimizer, device)

            # Log to tensorboard
            writer.add_scalars(f'LearningRate/{lr}/Loss', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalars(f'LearningRate/{lr}/Accuracy', {'train': train_acc, 'val': val_acc}, epoch)

        # Record final results
        results[lr] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss
        }

        print(f"Final results - Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return results
