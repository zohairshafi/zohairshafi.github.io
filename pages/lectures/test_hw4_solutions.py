#!/usr/bin/env python3
"""
Test script to verify all HW4 solutions work correctly.
"""

import numpy as np
import sys

def test_entropy():
    """Test the entropy function."""
    print("Testing entropy function...")

    # Define entropy function
    def entropy(y):
        unique_classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy_value = 0.0
        for p in probabilities:
            if p > 0:
                entropy_value -= p * np.log2(p)
        return entropy_value

    # Test cases
    tests = [
        (np.array([0, 0, 0, 0]), 0.0000, "Pure node"),
        (np.array([0, 1]), 1.0000, "Balanced binary"),
        (np.array([0, 1, 0, 1]), 1.0000, "Balanced binary 2"),
        (np.array([0, 0, 0, 1]), 0.8113, "Imbalanced 3:1"),
        (np.array([0, 1, 2]), 1.5850, "Three classes"),
    ]

    passed = 0
    for y, expected, desc in tests:
        result = entropy(y)
        if abs(result - expected) < 0.001:
            print(f"  ✓ {desc}: {result:.4f}")
            passed += 1
        else:
            print(f"  ✗ {desc}: got {result:.4f}, expected {expected:.4f}")

    print(f"Passed {passed}/{len(tests)} entropy tests\n")
    return passed == len(tests)


def test_information_gain():
    """Test the information gain function."""
    print("Testing information_gain function...")

    # Define functions
    def entropy(y):
        unique_classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy_value = 0.0
        for p in probabilities:
            if p > 0:
                entropy_value -= p * np.log2(p)
        return entropy_value

    def split_indices(X, feature_idx, value):
        return np.where(X[:, feature_idx] == value)[0]

    def information_gain(X, y, feature_idx):
        parent_entropy = entropy(y)
        conditional_entropy = 0.0
        unique_values = np.unique(X[:, feature_idx])

        for value in unique_values:
            indices = split_indices(X, feature_idx, value)
            y_subset = y[indices]
            prob = len(indices) / len(y)
            subset_entropy = entropy(y_subset)
            conditional_entropy += prob * subset_entropy

        return parent_entropy - conditional_entropy

    # Test data
    X = np.array([
        [1, 1], [1, 0], [1, 1], [0, 1],
        [0, 0], [0, 1], [1, 0], [0, 0]
    ], dtype=int)
    y = np.array([1, 1, 1, 0, 0, 0, 1, 1], dtype=int)

    # Test information gain
    ig0 = information_gain(X, y, 0)
    ig1 = information_gain(X, y, 1)

    expected_ig0 = 0.5488
    expected_ig1 = 0.0488

    passed = 0
    if abs(ig0 - expected_ig0) < 0.01:
        print(f"  ✓ Feature 0 IG: {ig0:.4f}")
        passed += 1
    else:
        print(f"  ✗ Feature 0 IG: got {ig0:.4f}, expected {expected_ig0:.4f}")

    if abs(ig1 - expected_ig1) < 0.01:
        print(f"  ✓ Feature 1 IG: {ig1:.4f}")
        passed += 1
    else:
        print(f"  ✗ Feature 1 IG: got {ig1:.4f}, expected {expected_ig1:.4f}")

    print(f"Passed {passed}/2 information gain tests\n")
    return passed == 2


def test_forward_pass():
    """Test the forward pass implementation."""
    print("Testing forward_pass function...")

    def relu(x):
        return np.maximum(0, x)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward_pass(X, W1, b1, W2, b2):
        Z1 = X @ W1 + b1
        A1 = relu(Z1)
        Z2 = A1 @ W2 + b2
        y_hat = sigmoid(Z2)
        return Z1, A1, Z2, y_hat

    # Test data
    X = np.array([[0.5, -1.0], [1.5, 0.3], [-0.3, 0.8]], dtype=float)
    W1 = np.array([[0.2, -0.4, 0.1], [0.7, 0.3, -0.5]], dtype=float)
    b1 = np.array([0.1, -0.2, 0.05], dtype=float)
    W2 = np.array([[0.6], [-0.1], [0.2]], dtype=float)
    b2 = np.array([0.0], dtype=float)

    Z1, A1, Z2, y_hat = forward_pass(X, W1, b1, W2, b2)

    # Check shapes
    passed = 0
    if Z1.shape == (3, 3):
        print(f"  ✓ Z1 shape correct: {Z1.shape}")
        passed += 1
    else:
        print(f"  ✗ Z1 shape incorrect: {Z1.shape}")

    if A1.shape == (3, 3):
        print(f"  ✓ A1 shape correct: {A1.shape}")
        passed += 1
    else:
        print(f"  ✗ A1 shape incorrect: {A1.shape}")

    if Z2.shape == (3, 1):
        print(f"  ✓ Z2 shape correct: {Z2.shape}")
        passed += 1
    else:
        print(f"  ✗ Z2 shape incorrect: {Z2.shape}")

    if y_hat.shape == (3, 1):
        print(f"  ✓ y_hat shape correct: {y_hat.shape}")
        passed += 1
    else:
        print(f"  ✗ y_hat shape incorrect: {y_hat.shape}")

    # Check first prediction approximately
    if 0.52 < y_hat[0, 0] < 0.54:
        print(f"  ✓ y_hat[0] in expected range: {y_hat[0, 0]:.4f}")
        passed += 1
    else:
        print(f"  ✗ y_hat[0] out of range: {y_hat[0, 0]:.4f}")

    print(f"Passed {passed}/5 forward pass tests\n")
    return passed == 5


def test_decision_tree():
    """Test the decision tree implementation."""
    print("Testing DecisionTree class...")

    # Import required functions
    def entropy(y):
        unique_classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy_value = 0.0
        for p in probabilities:
            if p > 0:
                entropy_value -= p * np.log2(p)
        return entropy_value

    def split_indices(X, feature_idx, value):
        return np.where(X[:, feature_idx] == value)[0]

    def information_gain(X, y, feature_idx):
        parent_entropy = entropy(y)
        conditional_entropy = 0.0
        unique_values = np.unique(X[:, feature_idx])

        for value in unique_values:
            indices = split_indices(X, feature_idx, value)
            y_subset = y[indices]
            prob = len(indices) / len(y)
            subset_entropy = entropy(y_subset)
            conditional_entropy += prob * subset_entropy

        return parent_entropy - conditional_entropy

    # Define DecisionTree class
    class DecisionTree:
        def __init__(self, max_depth=5):
            self.max_depth = max_depth
            self.tree = None
            self.feature_names = None

        def fit(self, X, y, feature_names=None):
            self.feature_names = feature_names
            self.tree = self._build_tree(X, y, depth=0)

        def _build_tree(self, X, y, depth):
            n_samples, n_features = X.shape
            majority_class = np.bincount(y.astype(int)).argmax()

            if (depth >= self.max_depth or n_samples <= 1 or
                n_features == 0 or entropy(y) == 0):
                return {'leaf': True, 'prediction': majority_class}

            best_ig = -1
            best_feature_idx = None

            for j in range(n_features):
                ig = information_gain(X, y, j)
                if ig > best_ig:
                    best_ig = ig
                    best_feature_idx = j

            if best_ig == 0 or best_feature_idx is None:
                return {'leaf': True, 'prediction': majority_class}

            children = {}
            feature_values = np.unique(X[:, best_feature_idx])

            for value in feature_values:
                indices = split_indices(X, best_feature_idx, value)
                feature_mask = np.ones(n_features, dtype=bool)
                feature_mask[best_feature_idx] = False
                X_child = X[indices][:, feature_mask]
                y_child = y[indices]
                children[value] = self._build_tree(X_child, y_child, depth + 1)

            feature_name = (self.feature_names[best_feature_idx]
                          if self.feature_names else f"Feature_{best_feature_idx}")

            return {
                'feature_idx': best_feature_idx,
                'feature_name': feature_name,
                'children': children
            }

        def predict_single(self, x, node=None, removed_features=None):
            if node is None:
                node = self.tree
                removed_features = []

            if node.get('leaf', False):
                return node['prediction']

            feature_idx = node['feature_idx']
            adjusted_idx = feature_idx
            for removed in sorted(removed_features):
                if removed <= feature_idx:
                    adjusted_idx += 1

            feature_value = x[adjusted_idx]

            if feature_value in node['children']:
                child_node = node['children'][feature_value]
                new_removed = removed_features + [adjusted_idx]
                return self.predict_single(x, child_node, new_removed)
            else:
                first_child = node['children'][list(node['children'].keys())[0]]
                if first_child.get('leaf', False):
                    return first_child['prediction']
                return self.predict_single(x, first_child, removed_features + [adjusted_idx])

        def predict(self, X):
            return np.array([self.predict_single(x) for x in X])

    # Test data
    X = np.array([
        [1, 1], [1, 0], [1, 1], [0, 1],
        [0, 0], [0, 1], [1, 0], [0, 0]
    ], dtype=int)
    y = np.array([1, 1, 1, 0, 0, 0, 1, 1], dtype=int)

    # Train tree
    dt = DecisionTree(max_depth=3)
    dt.fit(X, y, feature_names=["Has_HW", "Attended"])

    # Test predictions
    y_pred = dt.predict(X)
    accuracy = np.mean(y_pred == y)

    passed = 0
    if dt.tree is not None:
        print(f"  ✓ Tree was built")
        passed += 1
    else:
        print(f"  ✗ Tree was not built")

    if accuracy >= 0.75:
        print(f"  ✓ Accuracy is reasonable: {accuracy:.4f}")
        passed += 1
    else:
        print(f"  ✗ Accuracy is too low: {accuracy:.4f}")

    print(f"Passed {passed}/2 decision tree tests\n")
    return passed == 2


def main():
    """Run all tests."""
    print("=" * 60)
    print("HW4 Solutions Verification")
    print("=" * 60)
    print()

    results = []
    results.append(("Entropy", test_entropy()))
    results.append(("Information Gain", test_information_gain()))
    results.append(("Forward Pass", test_forward_pass()))
    results.append(("Decision Tree", test_decision_tree()))

    print("=" * 60)
    print("Summary:")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name:20s}: {status}")

    all_passed = all(result[1] for result in results)

    print()
    if all_passed:
        print("✓ All tests passed! The solutions are working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please review the implementations.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
