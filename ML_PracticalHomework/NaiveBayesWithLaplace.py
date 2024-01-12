import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_stats = {}
        n_features = X.shape[1]
        self.alpha = 1  # Smoothing parameter

        for c in self.classes:
            X_c = X[y == c]
            total_count_c = np.sum(X_c) + self.alpha * n_features
            word_counts_c = np.sum(X_c, axis=0) + self.alpha
            self.class_stats[c] = {
                'prior': len(X_c) / len(X),
                'word_counts': word_counts_c,
                'total_count': total_count_c
            }

    def predict(self, X):
        predictions = []
        for x in X:
            class_probs = {}
            for c in self.classes:
                log_prob_c = np.log(self.class_stats[c]['prior'])
                word_counts_c = self.class_stats[c]['word_counts']
                total_count_c = self.class_stats[c]['total_count']
                log_prob_c += np.sum(np.log(word_counts_c / total_count_c) * x)
                class_probs[c] = log_prob_c
            predictions.append(max(class_probs, key=class_probs.get))
        return np.array(predictions)

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

def perform_cross_validation(base_folder):
    parts = [f'part{i}' for i in range(1, 11)]
    accuracies = []

    for test_part in parts:
        print(f'Cross-validation: Testing on {test_part}')
        train_parts = [part for part in parts if part != test_part]

        # Load training data
        train_texts, train_labels = [], []
        for part in train_parts:
            part_path = os.path.join(base_folder, part)
            texts, labels = load_data(part_path)
            train_texts += texts
            train_labels += labels

        # Check if training texts are empty
        if not train_texts or all(len(text.strip()) == 0 for text in train_texts):
            print(f"Warning: No valid training data found in {train_parts}")
            continue

        # Load testing data
        test_part_path = os.path.join(base_folder, test_part)
        test_texts, test_labels = load_data(test_part_path)

        # Check if testing texts are empty
        if not test_texts or all(len(text.strip()) == 0 for text in test_texts):
            print(f"Warning: No valid testing data found in {test_part}")
            continue

        # Vectorize text data
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(train_texts).toarray()
        X_test_tfidf = vectorizer.transform(test_texts).toarray()
        y_train = np.array(train_labels)
        y_test = np.array(test_labels)

        # Train and predict with Naive Bayes Classifier
        classifier = NaiveBayesClassifier()
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
    accuracies = perform_cross_validation(base_folder)
    max_acc, min_acc, avg_acc, std_acc = calculate_statistics(accuracies)

    print(f"\nStatistics for '{preprocessing}' preprocessing:")
    print(f"Maximum Accuracy: {max_acc:.2f}")
    print(f"Minimum Accuracy: {min_acc:.2f}")
    print(f"Average Accuracy: {avg_acc:.2f}")
    print(f"Standard Deviation of Accuracy: {std_acc:.2f}")

    plot_accuracies(accuracies, f'LOOCV Naive Bayes Performance - {preprocessing}')
