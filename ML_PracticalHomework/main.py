from util import Util
import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        self.train_parts_texts = []
        self.train_parts_labels = []

        self.test_parts_texts = []
        self.test_parts_labels = []

        self.spam = []
        self.ham = []

        self.vocabulary = set()

        self.parameters_spam = {}
        self.parameters_ham = {}

        self.p_spam = 0
        self.p_ham = 0

    @staticmethod
    def load_part_data(category, part_count):
        folder = f'./lingspam_public/{category}/part{part_count}'
        texts, labels, spam, ham = Util.load_data(folder)

        return texts, labels, spam, ham

    def load_train_data(self, category, exclude_part=None):
        # Load data
        for part_count in range(1, 10):
            if exclude_part is not None and part_count == exclude_part:
                # Skip the part that is used for cross validation
                continue

            texts, labels, spam, ham = NaiveBayesClassifier.load_part_data(category, part_count)

            # Extend lists
            self.spam.extend(spam)
            self.ham.extend(ham)

            self.train_parts_texts.append(texts)
            self.train_parts_labels.append(labels)

    def prepare_data_for_train(self):
        # Construct vocabulary
        self.vocabulary = Util.construct_vocabulary(self.train_parts_texts)

        # Split spam and ham mails into words
        self.spam = Util.split_mails_into_words(self.spam)
        self.ham = Util.split_mails_into_words(self.ham)

    def calculate_parameters(self):
        # Calculate the number of times each word appears in a spam and ham mail
        spam_word_count = {unique_word: 0 for unique_word in self.vocabulary}
        ham_word_count = {unique_word: 0 for unique_word in self.vocabulary}

        for mail in self.spam:
            for word in mail:
                spam_word_count[word] += 1

        for mail in self.ham:
            for word in mail:
                ham_word_count[word] += 1

        # Calculate parameters
        n_spam = 0
        n_ham = 0

        for mail in self.spam:
            n_spam += len(mail)
        for mail in self.ham:
            n_ham += len(mail)
        n_vocabulary = len(self.vocabulary)

        self.p_spam = n_spam / (n_spam + n_ham)
        self.p_ham = n_ham / (n_spam + n_ham)
        alpha = 1

        self.parameters_spam = {unique_word: 0.0 for unique_word in self.vocabulary}
        self.parameters_ham = {unique_word: 0.0 for unique_word in self.vocabulary}

        for word in self.vocabulary:
            p_word_given_spam = (spam_word_count[word] + alpha) / (n_spam + alpha * n_vocabulary)
            p_word_given_ham = (ham_word_count[word] + alpha) / (n_ham + alpha * n_vocabulary)

            self.parameters_spam[word] = p_word_given_spam
            self.parameters_ham[word] = p_word_given_ham

    def train(self, category, exclude_part=None):
        # Load data
        self.load_train_data(category, exclude_part)

        # Prepare data for training
        self.prepare_data_for_train()

        # Calculate parameters
        self.calculate_parameters()

    def classify(self, text):
        text = Util.clean_text(text)
        text = text.split()

        p_spam_given_text = np.log(self.p_spam)
        p_ham_given_text = np.log(self.p_ham)

        for word in text:
            if word in self.parameters_spam:
                p_spam_given_text += np.log(self.parameters_spam[word])

            if word in self.parameters_ham:
                p_ham_given_text += np.log(self.parameters_ham[word])

        if p_ham_given_text >= p_spam_given_text:
            return 0
        else:
            return 1

    def calculate_accuracy(self, parts_texts, parts_labels):
        # Calculate accuracy
        correct = 0
        total = 0

        for text_part, label_part in zip(parts_texts, parts_labels):
            for text, label in zip(text_part, label_part):
                result = self.classify(text)
                if result == label:
                    correct += 1
                total += 1

        accuracy = correct / total
        return accuracy

    @staticmethod
    def cross_validation():
        # Cross validation
        accuracies = []

        print('\nCross validation:')

        for category in ['bare', 'lemm', 'lemm_stop', 'stop']:
            for i in range(1, 10):
                naive_bayes_classifier = NaiveBayesClassifier()

                # Train with excluded part
                naive_bayes_classifier.train(category, i)

                # Load excluded part separately for testing
                excluded_texts, excluded_labels, _, _ = NaiveBayesClassifier.load_part_data(category, i)

                # Calculate accuracy
                accuracy = naive_bayes_classifier.calculate_accuracy([excluded_texts], [excluded_labels])
                accuracies.append(accuracy)

            # Calculate mean accuracy
            mean_accuracy = np.mean(accuracies)
            print(f'\t{category} mean accuracy: {mean_accuracy}')

            accuracies = []

    def test_accuracy(self, category):
        # Load test data
        test_texts, test_labels, _, _ = NaiveBayesClassifier.load_part_data(category, 10)

        # Calculate accuracy
        accuracy = self.calculate_accuracy([test_texts], [test_labels])
        print(f'{category} accuracy on test data: {accuracy}')


def main():
    # Train with all data
    for category in ['bare', 'lemm', 'lemm_stop', 'stop']:
        naive_bayes_classifier = NaiveBayesClassifier()
        naive_bayes_classifier.train(category)

        naive_bayes_classifier.test_accuracy(category)

    # Cross validation
    NaiveBayesClassifier.cross_validation()


if __name__ == '__main__':
    main()
