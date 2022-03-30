'''
Multinomial Naive Bayes Classifier
'''
import string
import nltk
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
    create histogram of training set's words and their frequencies.
Parameters:
    training_data   (list of tuples) - List of tuples containing (<email_text>, <ham or spam>) 1 for spam and 0 for ham.
                    you only need to do this for the training data
Exceptions:
    None
Return:
    spam_freq   (dict) - key is the word, and the entry is the frequency.
    ham_freq    (dict) - key is the word, and the entry is the frequency.
'''
def get_training_prob(training_data, n, select=True):
    # get dictionary with list of all words from both datasets
    # mwlist is master word dictionary !!
    if select is False:
        n = len(training_data)
    mwdict = {}
    for val in training_data[:n]:
        for word in val[0]:
            if word not in mwdict:
                mwdict.update([(word, 1)])
    # sprob = spam wordprobabilities, hprob = ham word probabilities
    sprob, hprob = mwdict.copy(), mwdict.copy()
    #swd is spam word count
    swc, hwc = 0, 0
    # count spam and ham freqs
    for val in training_data[:n]:
        # for spam
        if val[1] == 1:
            for word in val[0]:
                sprob[word] += 1
                swc += 1
        # for ham
        if val[1] == 0:
            for word in val[0]:
                hprob[word] += 1
                hwc += 1

    # calulate spam word appearance probability
    for key in sprob:
        sprob[key] = sprob[key] / swc
    # calulate ham word appearance probability
    for key in hprob:
        hprob[key] = hprob[key] / hwc
    with open('svocab.txt','w') as f:
        f.write(str(sprob))
    f.close()
    with open('hvocab.txt','w') as f:
        f.write(str(hprob))
    f.close()
    return sprob, hprob, mwdict

'''
Description:
    multinomial niave bayes classification
    calulate similarity score against spam and ham data for ONE email
    and return what it thinks the email is
Parameters:
    sprob   (dict)
    hprob   (dict)
    val         (tuple)
Exceptions:
    None
Return:
    res         (int)   -   either a 0 for ham or a 1 for spam, result from classification
    actual      (int)   -   either a 0 for ham or a 1 for spam, actual classification of email
    accuracy    (int)   -   either a 0 for correct or 1 for incorrect evaluation
'''
def mnbclassify(spam_freqs, ham_freqs, val):
    # calculate similarity, ssim = spam similarity score
    res = 0
    accuracy = 1
    ssim = 1
    for word in val[0]:
        if word in spam_freqs:
            ssim *= spam_freqs[word]
    hsim = 1
    for word in val[0]:
        if word in ham_freqs:
            hsim *= ham_freqs[word]
    
    # if spam likelihood is greater
    if ssim > hsim:
        res = 1
    
    # if evaluation is correct
    if res == val[1]:
        accuracy = 0

    return res, val[1], accuracy


if __name__ == "__main__":
    training_data, testing_data = get_data()
    training_sprob, training_hprob, mwdict = get_training_prob(training_data, 500, False)

    res = []
    # failure rate
    fr = 0
    for val in testing_data[:600]:
        # classify return
        cret = mnbclassify(training_sprob,training_hprob,val)
        res.append(cret)
        fr += cret[2]
        
        # print(f"{cret[0]}:{cret[1]}:{cret[2]}")
        
    print(f"Accuracy rate of {100*(len(res) - fr)/len(res)}%")
