import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def multilinear_reg(x,y):
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

    from sklearn.linear_model import LinearRegression
    reg_mult = LinearRegression()
    reg_mult.fit(x_train, y_train)

    y_pred = reg_mult.predict(x_test)
    np.set_printoptions(precision=2)
    #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

    from sklearn.metrics import r2_score
    mult_r2 = r2_score(y_test, y_pred)
    return mult_r2

def polynomial_reg(x,y):
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree = 4)
    x_poly = poly_reg.fit_transform(x_train)
    reg_poli = LinearRegression()
    reg_poli.fit(x_poly,y_train)

    y_pred = reg_poli.predict(poly_reg.transform(x_test))
    np.set_printoptions(precision=2)
    #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

    from sklearn.metrics import r2_score
    poli_r2 = r2_score(y_test, y_pred)
    return poli_r2

def SVR(x,y):
    
    y_SVR = y.reshape((len(y), 1))

    from sklearn.model_selection import train_test_split
    x_train_SVR, x_test_SVR, y_train_SVR, y_test_SVR = train_test_split(x, y_SVR, test_size = 0.2, random_state = 0)


    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x_train_SVR = sc_x.fit_transform(x_train_SVR)
    y_train_SVR = sc_y.fit_transform(y_train_SVR)

    from sklearn.svm import SVR
    reg_SVR = SVR(kernel = 'rbf')
    reg_SVR.fit(x_train_SVR, y_train_SVR)

    y_pred = sc_y.inverse_transform(reg_SVR.predict(sc_x.transform(x_test_SVR)).reshape(-1, 1))
    np.set_printoptions(precision=2)
    #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

    from sklearn.metrics import r2_score
    SVR_r2 = r2_score(y_test_SVR, y_pred)
    return SVR_r2

def decision_tree_reg(x,y):
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

    
    from sklearn.tree import DecisionTreeRegressor
    reg_decision = DecisionTreeRegressor(random_state = 0)
    reg_decision.fit(x_train, y_train)

    y_pred = reg_decision.predict(x_test)
    np.set_printoptions(precision=2)
    #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

    from sklearn.metrics import r2_score
    decision_r2 = r2_score(y_test, y_pred)
    return decision_r2

def random_forest_reg(x,y):
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

    
    from sklearn.ensemble import RandomForestRegressor
    reg_forest = RandomForestRegressor(n_estimators = 10, random_state = 0)
    reg_forest.fit(x_train, y_train)

    y_pred = reg_forest.predict(x_test)
    np.set_printoptions(precision=2)
    #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

    from sklearn.metrics import r2_score
    forest_r2 = r2_score(y_test, y_pred)
    return forest_r2



#Importar os dados
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

mult_r2 = multilinear_reg(x,y)
poli_r2 = polynomial_reg(x,y)
SVR_r2 = SVR(x,y)
decision_r2 = decision_tree_reg(x,y)
forest_r2 = random_forest_reg(x,y)

#Resposta
print("Multi R2 = ", mult_r2)
print("Poli R2 = ", poli_r2)
print("SVR R2 = ", SVR_r2)
print("Decision Tree R2 = ", decision_r2)
print("Random Forest R2 = ", forest_r2)



