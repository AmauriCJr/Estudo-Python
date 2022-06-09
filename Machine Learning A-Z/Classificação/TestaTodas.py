import pandas as pd



def Logistic_Regression (x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    ac_sc = accuracy_score(y_test, y_pred)
    
    return cm, ac_sc

def KNN(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    
    cm = confusion_matrix(y_test, y_pred)
    as_sc = accuracy_score(y_test, y_pred)

    return cm, as_sc

def SVM(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    as_sc = accuracy_score(y_test, y_pred)
    
    return cm, as_sc

def Kernel_SVM (x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    as_sc = accuracy_score(y_test, y_pred)
    
    return cm, as_sc

def Naive_Bayes (x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    
    cm = confusion_matrix(y_test, y_pred)
    as_sc = accuracy_score(y_test, y_pred)
    
    return cm, as_sc

def Decision_Tree(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(x_train, y_train)
    
    y_pred = classifier.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    as_sc = accuracy_score(y_test, y_pred)
    
    return cm, as_sc

def Random_Forest(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    as_sc = accuracy_score(y_test, y_pred)
    
    return cm, as_sc

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score



cm_log, as_log = Logistic_Regression (x, y)
cm_KNN, as_KNN = KNN(x, y)
cm_SVM, as_SVM = SVM(x, y)
cm_ker, as_ker = Kernel_SVM(x, y)
cm_NB, as_NB = Naive_Bayes(x, y)
cm_DT, as_DT = Decision_Tree(x, y)
cm_RF, as_RF = Random_Forest(x, y)

print("Acurácia LogReg = ", as_log)
print("Acurácia KNN = ", as_KNN)
print("Acurácia SVM = ", as_SVM)
print("Acurácia Kernel SVM = ", as_ker)
print("Acurácia Naive Bayes = ", as_NB)
print("Acurácia Decision Tree = ", as_DT)
print("Acurácia Random Forest = ", as_RF)
