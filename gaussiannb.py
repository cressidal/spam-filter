'''
Gaussian Naive Bayes
'''
import string
import nltk
import numpy as np
import math
from nltk.corpus import stopwords
nltk.download('stopwords')
import os
import codecs

'''
Description:
    clean text, remove capitalisation, stopwords and punctuation and return a list of words
Parameters:
    text    (string) - text to be cleaned up
Exceptions
    None
Return:
    clean_text  (string) - clean text
'''
def clean_text(text):
    # remove punctuation in data
    for i in text:
        for j in string.punctuation:
            i = i.replace(j, "")
    # remove stopwords, change all words to lowercase
    words = text.split()
    clean = []
    for word in words:
        if word not in stopwords.words("english"):
            clean.append(word.lower())
    return clean

'''
Description:
    clean up data from testing and training folders and return a workable data set
Parameters:
    None
Exceptions:
    None
Return:
    training_data   (list of tuples) - List of tuples containing (<email_text>, <ham or spam>) 1 for spam and 0 for ham
    test_data       (list of tuples) - List of tuples containing (<email_text>, <ham or spam>) 1 for spam and 0 for ham
'''
def get_data():
    print("getting data")
    testing_data = []
    training_data = []
    # open files
    # c = 0

    for file in os.listdir("emails/training/ham"):
        with codecs.open(os.path.join("emails/training/ham/" + file), 'r', encoding='utf-8', errors='ignore') as f:
            # clean up and append
            training_data.append((clean_text(f.read()), 0))
        # c += 1
        # if c == 200:
        #     break

    for file in os.listdir("emails/training/spam"):
        with codecs.open(os.path.join("emails/training/spam/", file), 'r', encoding='utf-8', errors='ignore') as f:
            training_data.append((clean_text(f.read()), 1))
        # c += 1
        # if c == 400:
        #     break
    
    for file in os.listdir("emails/testing/ham"):
        with codecs.open(os.path.join("emails/testing/ham/", file), 'r', encoding='utf-8', errors='ignore') as f:
            testing_data.append((clean_text(f.read()), 0))
        # c += 1
        # if c == 600:
        #     break

    for file in os.listdir("emails/testing/spam"):
        with codecs.open(os.path.join("emails/testing/spam/", file), 'r', encoding='utf-8', errors='ignore') as f:
            testing_data.append((clean_text(f.read()), 1))
        # c += 1
        # if c == 800:
        #     break
    
    print("finished getting data")
    return training_data, testing_data

'''
Description:
    create histogram of training set's words and their standard deviation and mean.
Parameters:
    training_data   (list of tuples) - List of tuples containing (<email_text>, <ham or spam>) 1 for spam and 0 for ham.
                    you only need to do this for the training data
Exceptions:
    None
Return:
    spam_freq   (dict) - key is the word, and the entry is the frequency.
    ham_freq    (dict) - key is the word, and the entry is the frequency.
'''
def get_training_freqs(training_data, n, select=True):
    # get dictionary with list of all words from both datasets
    # mwlist is master word dictionary !!
    mwl = []

    if select is False:
        n = len(training_data)
    for val in training_data[:n]:
        mwl + val[0]

    mwl = list(set(mwl))

    # master word dictionary
    mwd = {}

    for word in mwl:
        mwd.update([word, []])

    sfreqs = dict(mwd)
    hfreqs = dict(mwd)
    
    for val in training_data[:n]:
        if val[1] == 1:
            for word in val[0]:
                if word in sfreqs:
                    sfreqs[word].append(val[0].count(word)/len(val[0]))
        else:
            for word in val[0]:
                if word in hfreqs:
                    hfreqs[word].append(val[0].count(word)/len(val[0]))

    for word in sfreqs:
        # get rid of the word if it only appears in like 9 emails
        if len(sfreqs[word]) < 3:
            sfreqs.pop(word)
        else:
            # calculate the standard deviation and mean
            mean = sum(sfreqs[word])/len(sfreqs[word])
            sd = np.std(sfreqs[word])

            # if every email has it, then just get rid of it
            if sd == 0:
                sfreqs.pop(word)
            else:
                sfreqs[word] = [mean,sd]

    for word in hfreqs:
        if len(hfreqs[word]) < 3:
            hfreqs.pop(word)
        else:
            mean = sum(hfreqs[word])/len(hfreqs[word])
            sd = np.std(hfreqs[word])
            if sd == 0:
                hfreqs.pop(word)
            else:
                hfreqs[word] = [mean,sd]


    return sfreqs, hfreqs, mwd

'''
Description:
    guassian niave bayes classification
    calulate similarity score against spam and ham data for ONE email
    and return what it thinks the email is.
    We also assume that the email is ham, because that's what our dataset looks like.
Parameters:
    sfreq   (dict)
    hfreq   (dict)
    val     (tuple)
Exceptions:
    None
Return:
    res         (int)   -   either a 0 for ham or a 1 for spam, result from classification
    actual      (int)   -   either a 0 for ham or a 1 for spam, actual classification of email
    accuracy    (int)   -   either a 0 for correct or 1 for incorrect evaluation
'''
def gnbclassify(sfreq, hfreq, val):
    res = 0
    accuracy = 1
    # turn val into a dictionary with values we can process
    # wl = word list
    # wcl = word count list
    wl = set(val[0])
    # print(wl)
    wcl = {}
    for word in wl:
        wcl[word] = val[0].count(word)/ len(val[0])
    
    # print(wcl)

    # pp = prior probability, which is constant we get from the dataset
    # ham / total number of emails
    pp = 3672/5975

    # slh = spam likelikhood
    slh = pp
    for word in wcl:
        if word in sfreq:
            mean, sd = sfreq[word]
            # maximum likelihood formula for guassian distribution, with log loading to avoid underflow
            # print(slh)
            slh += math.log((1 / (math.sqrt(2 * np.pi * (sd ** 2)))) * np.exp(-((wcl[word] - mean) ** 2) / (2 * (sd ** 2))))
        
    # hlh = ham likelikhood
    hlh = pp
    for word in wcl:
        if word in hfreq:
            mean, sd = hfreq[word][0], hfreq[word][1]
            # print(f"{mean}{sd}")
            hlh += math.log((1 / (math.sqrt(2 * np.pi * (sd ** 2)))) * np.exp(-((wcl[word] - mean) ** 2) / (2 * (sd ** 2))))

    if slh > hlh:
        res = 1

    if res == val[1]:
        accuracy = 0
    # print (f"h:{hlh}, s:{slh}")
    return res, val[1], accuracy


if __name__ == "__main__":
    training_data, testing_data = get_data()
    training_sprob, training_hprob, mwdict = get_training_freqs(training_data, 500, False)

    res = []
    # failure rate
    fr = 0
    for val in testing_data[:2]:
        # classify return
        cret = gnbclassify(training_sprob,training_hprob,val)
        res.append(cret)
        fr += cret[2]
        
        # print(f"{cret[0]}:{cret[1]}:{cret[2]}")
        
    print(f"Accuracy rate of {100*(len(res) - fr)/len(res)}%")