'''
KNN spam filter
In this machine learing technique, the given item is compared to the training data,
and the items that are most similar to it are looked at. Say we find 5 items that are
similar to the given item, and 3 of them are classifed as spam. Because the majority of 
the item's neighbours are classified as spam, we also classify the given item as spam.

There is no independent training stage for the k nearest neighbour like there is for 
Guassian Naive Bayes Classification and Multinomial Niave Bayes Classfication.
'''
import os
import string
import nltk
import codecs
import sklearn
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
import numpy as np
import matplotlib.pyplot as plt

def load_training_data():
    '''
    Load data from dataset and files
    '''
    data = []

    # load ham
    for file in os.listdir("emails/training/ham"):
        with codecs.open(os.path.join("emails/training/ham/" + file), 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            data.append([text, "ham"])

    for file in os.listdir("emails/training/spam"):
        with codecs.open(os.path.join("emails/training/spam/" + file), 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            data.append([text, "spam"])

    data = np.array(data)
    print("training data loaded\n")
    return data
def split_data(data):
    data_text = data[:,0]
    data_labels = data[:,1]
    return data_text, data_labels
def clean_data(data):
    '''
    clean up data. Remove punctuation, change all words to lowercase, remove stopwords.
    '''
    # remove punctuation in data
    for i in data:
        # Remove common punctuation and symbols
        for item in string.punctuation:
            i[0] = i[0].replace(item, "")
             
        # Lowercase all letters and remove stopwords 
        splittedWords = i[0].split()
        newText = ""
        for word in splittedWords:
            if word not in stopwords.words("english"):
                word = word.lower()
                newText = newText + " " + word  # Takes back all non-stopwords
        i[0] = newText
    
    return data
def split_data(data):
    '''
    Split data into training and testing
    '''
    features = data[:,0]
    labels = data[:,1]
    training_data, testing_data, training_labels, test_labels = train_test_split(features, labels, test_size = 0.30, random_state = 42)
    return training_data, testing_data, training_labels, test_labels
def get_word_count(text):
    '''
    get text and return a dictionaryu of words and their counts
    '''
    # make a dictionary of {"word": freq}
    wcset = set(text.split())
    wc = {}
    for word in wcset:
        wc[word] = text.split().count(word)
    return wc
def e_dif(training_wcount, testing_wcount):
    '''
    Applying the euclidean difference forumla for the training data and test email
    '''
    total = 0
    for word in testing_wcount:
        if word in testing_wcount and word in training_wcount:
            total += (testing_wcount[word] - training_wcount[word]) ** 2
            del training_wcount[word]
        else:
            total += testing_wcount[word] ** 2
        
    for word in training_wcount:
        total += training_wcount[word]**2
    
    return total**0.5
def get_class(k_vals):
    '''
    classify given test email, given a list of similar emails.
    If there are the same number of ham and spam emails, the tested email
    will be returned as ham (not spam)
    '''
    nham = 0
    nspam = 0
    for val in k_vals:
        if val[1] == 'spam':
            nspam +=1
        else:
            nham += 1

    if nspam > nham:
        return "spam"
    else:
        return "ham"
def knn(training_data, test_data, training_labels, k):
    '''
    Classifying emails using k-nearest neighbour

    Arguments:

    training_data   (list)  processed training data
    test_data       (list)  text of email/s that is/are to be processed
    k               (int)   k value
    t_size          (int)   number of test emails to be tested.
    '''
    res = []
    training_freqs = []
    for training_text in training_data:
        training_freqs.append(get_word_count(training_text))
    for test_text in test_data:
        similarity = []
        test_freqs = get_word_count(test_text)
        for i in range(training_data):
            similarity.append([training_labels[i], e_dif(test_freqs,training_freqs[i])])

        similarity = sorted(similarity, key = lambda i:i[1])
        selected_vals = []
        for i in range(k):
            selected_vals.append(similarity)
        res.append(get_class(selected_vals))

    return res
def main(K):
    data = load_training_data()
    data = clean_data(data)
    training_data, test_data, training_labels, test_labels = split_data(data)
    
    # sample size of test emails to be tested. Use len(test_data) to test all test_data
    tsize = len(test_data)
    
    result = knn(training_data, training_labels, test_data[:tsize], K) 
    accuracy = accuracy_score(test_labels[:tsize], result)
    
    # print("training data size\t: " + str(len(training_data)))
    # print("test data size\t\t: " + str(len(test_data)))
    # print("K value\t\t\t\t: " + str(K))
    # print("Samples tested\t\t: " + str(tsize))
    # print("% accuracy\t\t\t: " + str(accuracy * 100))
    # print("Number correct\t\t: " + str(int(accuracy * tsize)))
    # print("Number wrong\t\t: " + str(int((1 - accuracy) * tsize)))

main(11)


# Plotting accuracy for different test sizes

tsize = [150, 300, 600, 900, 1200, 1582]
accuracy = [82.7, 84.0, 80.7, 79.0, 76.6, 76.7]

plt.figure()
plt.ylim(0, 100)
plt.plot(tsize, accuracy)
plt.xlabel("Number of Test Samples")
plt.ylabel("% Accuracy")
plt.title("KNN Algorithm Accuracy")
plt.grid()
plt.show()


# to determine a suitable value for K

def get_k():
    data = load_training_data()
    data = clean_data(data)
    training_data, test_data, training_labels, test_labels = split_data(data)
    
    # sample size of test emails to be tested. Use len(test_data) to test all test_data
    tsize = 150
    
    K_accuracy = []
    for K in range(1,50, 2):
        result = knn(training_data, training_labels, test_data[:tsize], K, tsize) 
        accuracy = accuracy_score(test_labels[:tsize], result)
        K_accuracy.append([K, accuracy*100])
        print("training data size\t: " + str(len(training_data)))
        print("test data size\t\t: " + str(len(test_data)))
        print("K value\t\t\t\t: " + str(K))
        print("Samples tested\t\t: " + str(tsize))
        print("% accuracy\t\t\t: " + str(accuracy * 100))
        print("Number correct\t\t: " + str(int(accuracy * tsize)))
        print("Number wrong\t\t: " + str(int((1 - accuracy) * tsize)))
    K_accuracy_sorted = sorted(K_accuracy, key = lambda i:i[1])
    print(K_accuracy_sorted)
    print("MAX: " + str(max(K_accuracy_sorted, key = lambda i:i[1])))
    
    # plot
    
    K_accuracy = np.array(K_accuracy)
    K_values = K_accuracy[:, 0]
    accuracies = K_accuracy[:, 1]
    
    plt.figure()
    plt.ylim(0, 101)
    plt.plot(K_values, accuracies)
    plt.xlabel("K Value")
    plt.ylabel("% Accuracy")
    plt.title("KNN Algorithm Accuracy With Different K")
    plt.grid()
    plt.show()
    