import os
import string
import matplotlib.pyplot as plt
import seaborn as sns


class Util:
    @staticmethod
    def clean_text(text):
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove words containing numbers
        text = ' '.join(word for word in text.split() if not any(c.isdigit() for c in word))

        # Lowercase all text
        text = text.lower()

        return text

    @staticmethod
    def load_data(folder):
        texts, labels = [], []
        spam = []
        ham = []

        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.txt'):
                    path = os.path.join(root, file)
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                        # Clean text
                        text = Util.clean_text(text)
                        texts.append(text)

                        # Append text to spam or ham list
                        if 'spmsg' in file:
                            spam.append(text)
                            labels.append(1)
                        else:
                            ham.append(text)
                            labels.append(0)

        return texts, labels, spam, ham

    @staticmethod
    def construct_vocabulary(text_parts):
        vocabulary = set()

        for part in text_parts:
            for text in part:
                for word in text.split():
                    vocabulary.add(word)
        return vocabulary

    @staticmethod
    def split_mails_into_words(mails):
        mails = [mail.split() for mail in mails]
        return mails

    @staticmethod
    def plot_accuracy_per_category(categories, accuracies):
        plt.figure(figsize=(10, 6))
        plt.scatter(categories, accuracies, color='blue', s=100, alpha=0.7)
        plt.xlabel('Categories')
        plt.ylabel('Accuracy')
        plt.title('Test Accuracies for each category')
        plt.ylim(0.95, 1)  # Set y-axis limit between 0.95 and 1 for accuracy
        plt.show()
