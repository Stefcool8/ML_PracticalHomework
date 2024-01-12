import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_stats = {}
        for c in self.classes:
            X_c = X[y == c]
            self.class_stats[c] = {
                'prior': len(X_c) / len(X),
                'word_counts': np.sum(X_c, axis=0),
                'total_count': np.sum(np.sum(X_c, axis=0))
            }

    def predict(self, X):
        predictions = []
        for x in X:
            class_probs = {}
            for c in self.classes:
                log_prob_c = np.log(self.class_stats[c]['prior'])
                total_count_c = self.class_stats[c]['total_count']
                word_counts_c = self.class_stats[c]['word_counts']
                log_prob_c += np.sum(np.log((word_counts_c + 1) / (total_count_c + len(word_counts_c))) * x)
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
                    texts.append(text)
                    labels.append(1 if 'spmsg' in file else 0)
    return texts, labels

folder_path = 'lingspam_public/bare'
texts, labels = load_data(folder_path)

df = pd.DataFrame({'text': texts, 'label': labels})

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['text']).toarray()
y = df['label'].values

kf = KFold(n_splits=100)
accuracies = []

for train_index, test_index in kf.split(X_tfidf):
    print(f'Fold {len(accuracies) + 1}/100')
    X_train, X_test = X_tfidf[train_index], X_tfidf[test_index]
    y_train, y_test = y[train_index], y[test_index]

    classifier = NaiveBayesClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.plot(range(1, 101), accuracies)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('100-Fold Cross-Validation Naive Bayes Performance')
plt.show()
