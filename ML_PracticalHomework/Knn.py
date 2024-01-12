import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter


class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

def load_data(folder):
    texts, labels = [], []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.txt'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    labels.append(1 if 'spmsg' in file else 0)
                    texts.append(text)
    return texts, labels

def perform_cross_validation(base_folder, classifier):
    parts = [f'part{i}' for i in range(1, 11)]
    accuracies = []

    for test_part in parts:
        print(f'Cross-validation: Testing on {test_part}')
        train_parts = [part for part in parts if part != test_part]

        train_texts, train_labels = [], []
        for part in train_parts:
            part_path = os.path.join(base_folder, part)
            texts, labels = load_data(part_path)
            train_texts += texts
            train_labels += labels

        test_part_path = os.path.join(base_folder, test_part)
        test_texts, test_labels = load_data(test_part_path)

        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(train_texts).toarray()
        X_test_tfidf = vectorizer.transform(test_texts).toarray()
        y_train = np.array(train_labels)
        y_test = np.array(test_labels)

        classifier.fit(X_train_tfidf, y_train)
        y_pred = classifier.predict(X_test_tfidf)
        accuracies.append(accuracy_score(y_test, y_pred))

    return accuracies

def calculate_statistics(accuracies):
    max_accuracy = max(accuracies)
    min_accuracy = min(accuracies)
    average_accuracy = np.mean(accuracies)
    std_dev_accuracy = np.std(accuracies)
    return max_accuracy, min_accuracy, average_accuracy, std_dev_accuracy

def plot_accuracies(accuracies, title):
    plt.plot(range(1, len(accuracies) + 1), accuracies)
    plt.xlabel('Iteration (Part used as Test Set)')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.show()

preprocessing_types = ['bare', 'lemm', 'stop', 'lemm_stop']
base_path = 'lingspam_public'

for preprocessing in preprocessing_types:
    print(f"\nPerforming Cross-Validation for '{preprocessing}'")
    base_folder = os.path.join(base_path, preprocessing)
    knn_classifier = KNearestNeighbors(k=3)
    accuracies = perform_cross_validation(base_folder, knn_classifier)
    max_acc, min_acc, avg_acc, std_acc = calculate_statistics(accuracies)

    print(f"\nStatistics for '{preprocessing}' preprocessing:")
    print(f"Maximum Accuracy: {max_acc:.2f}")
    print(f"Minimum Accuracy: {min_acc:.2f}")
    print(f"Average Accuracy: {avg_acc:.2f}")
    print(f"Standard Deviation of Accuracy: {std_acc:.2f}")

    plot_accuracies(accuracies, f'LOOCV k-NN Performance - {preprocessing}')
