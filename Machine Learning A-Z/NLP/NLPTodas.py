import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def NB (x_train, x_test, y_train, y_test):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    AS = accuracy_score(y_test, y_pred)
    return AS

def LR (x_train, x_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(x_train, y_train)
    
    y_pred = classifier.predict(x_test)
    
    cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    AS = accuracy_score(y_test, y_pred)
    return AS

def KNN (x_train, x_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(x_train, y_train)
    
    y_pred = classifier.predict(x_test)
    
    cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    AS = accuracy_score(y_test, y_pred)
    return AS

def KernelSVM (x_train, x_test, y_train, y_test):
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(x_train, y_train) 
    
    y_pred = classifier.predict(x_test)
    
    cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    AS = accuracy_score(y_test, y_pred)
    return AS

def SVM (x_train, x_test, y_train, y_test):
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(x_train, y_train)
    
    y_pred = classifier.predict(x_test)
    
    cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    AS = accuracy_score(y_test, y_pred)
    return AS

def DecisionTree (x_train, x_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(x_train, y_train)
    
    y_pred = classifier.predict(x_test)
    
    cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    AS = accuracy_score(y_test, y_pred)
    return AS

def RandomForest (x_train, x_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
    classifier.fit(x_train, y_train)
    
    y_pred = classifier.predict(x_test)
    
    cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    AS = accuracy_score(y_test, y_pred)
    return AS

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.metrics import confusion_matrix, accuracy_score

AS_NB = NB(x_train, x_test, y_train, y_test)
AS_LR = LR(x_train, x_test, y_train, y_test)
AS_KNN = KNN(x_train, x_test, y_train, y_test)
AS_KSVM = KernelSVM(x_train, x_test, y_train, y_test)
AS_SVM = SVM(x_train, x_test, y_train, y_test)
AS_DT = DecisionTree(x_train, x_test, y_train, y_test)
AS_RF = RandomForest(x_train, x_test, y_train, y_test)

print ("AS Naive Bayes = ", AS_NB)
print ("AS Logistic Regression = ", AS_LR)
print ("AS KNN = ", AS_KNN)
print ("AS Kernel SVM = ", AS_KSVM)
print ("AS SVM = ", AS_SVM)
print ("AS Decision Tree = ", AS_DT)
print ("AS Random Forest = ", AS_RF)
