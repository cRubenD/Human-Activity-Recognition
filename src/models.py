import torch
import torch.nn as nn
import numpy as np

"""
Cerinta 3: sa se implementeze >= 3 clasificatoare (nu cu clase gata existente din librarii)
"""


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

    def predict_proba(self, X):
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.forward(X_tensor)
            probs = torch.softmax(logits, dim=1)
            return probs.numpy()

    def predict(self, X):
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.forward(X_tensor)
            return torch.argmax(logits, dim=1).numpy()


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_classes):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_layer in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_layer))
            layers.append(nn.ReLU())
            prev_dim = hidden_layer
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, X):
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.forward(X_tensor)
            probs = torch.softmax(logits, dim=1)
            return probs.numpy()

    def predict(self, X):
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.forward(X_tensor)
            return torch.argmax(logits, dim=1).numpy()


# Decision Tree Implementation
class DecisionTreeClassifierSimple:
    class Node:
        def __init__(self, feature_idx=None, threshold=None, left=None, right=None, *, value=None):
            self.feature_idx = feature_idx
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def __init__(self, max_depth=5, min_samples_split=2):
        self.n_classes = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        counts = np.bincount(y)
        probs = counts / m
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        m, n = X.shape
        best_idx, best_thr, best_gain = None, None, 0
        parent_gini = self._gini(y)

        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes
            num_right = np.bincount(classes, minlength=self.n_classes).tolist()

            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                if thresholds[i] == thresholds[i - 1]:
                    continue

                gini_left = 1 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes))
                gini_right = 1 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes))
                gini = (i * gini_left + (m - i) * gini_right) / m
                gain = parent_gini - gini

                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # stopping criteria
        if depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        feat_idx, threshold = self._best_split(X, y)
        if feat_idx is None:
            return self.Node(value=self._most_common_label(y))

        # split dataset
        left_idx = X[:, feat_idx] < threshold
        X_l, y_l = X[left_idx], y[left_idx]
        X_r, y_r = X[~left_idx], y[~left_idx]

        left = self._grow_tree(X_l, y_l, depth + 1)
        right = self._grow_tree(X_r, y_r, depth + 1)
        return self.Node(feature_idx=feat_idx, threshold=threshold, left=left, right=right)

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def predict_proba(self, X):
        # Get predictions
        y_pred = self.predict(X)

        # Convert to one-hot encoding (pseudo-probabilities)
        probs = np.zeros((len(X), self.n_classes))
        for i, pred in enumerate(y_pred):
            probs[i, pred] = 1.0

        return probs

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
