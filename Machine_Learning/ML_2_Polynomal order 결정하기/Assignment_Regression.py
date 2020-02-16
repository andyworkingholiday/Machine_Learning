#2015104124 진우빈
#2015104045 서재하
#2015104032 박지훈

import sklearn
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.linear_model import LinearRegression
#import pandas as pd
from sklearn.metrics import mean_squared_error
import random

train_percent = 0.36
test_percent= 0.6
valid_percent = 0.04

data = np.loadtxt('data.csv', unpack=True, delimiter=',',skiprows=0 )
data = np.transpose(data)

#Write a code for acquiring unbiased data
random.shuffle(data)


#Obtaining Training data set
train_set = data[0:int(len(data)*train_percent)]
train_set = sorted(train_set, key=lambda train_set: train_set[0]) # Sorting again for data in order
print(len(train_set))
train_set = np.transpose(train_set)

#Reallocate for efficient programming
train_x = train_set[0]  #train_set[0] : feature data set (i.e, x)
train_y = train_set[1] #train_set[1] : lable data set (i.e, y)


#Write code for obtaining valid data set : valid_set
train_set = np.transpose(train_set)
valid_set = data[int(len(train_set)):int(len(data)*(train_percent + valid_percent))+1]
valid_set = sorted(valid_set, key=lambda valid_set: valid_set[0]) # Sorting again for data in order
print(len(valid_set))
train_set = np.transpose(train_set)
valid_set = np.transpose(valid_set)

valid_x = valid_set[0]
valid_y = valid_set[1]

#Write code for obtaining test data set :test_set
test_set = data[int(len(data)*(train_percent + valid_percent))+1:]
test_set = sorted(test_set, key=lambda test_set: test_set[0]) # Sorting again for data in order
print(len(test_set))
test_set = np.transpose(test_set)
test_x = test_set[0]
test_y = test_set[1]


##################### Regression Libraries #############
def fit_polynomial(x,y , degree):
    '''
    Fits a polynomial to the input sample.
    (x,y): input sample
    degree: polynomial degree
    '''
    model = LinearRegression()
    model.fit(np.vander(x, degree + 1), y)
    return model

def apply_polynomial(model, x):
    '''
    Evaluates a linear regression model in an input sample
    model: linear regression model
    x: input sample
    '''
    degree = model.coef_.size - 1
    y = model.predict(np.vander(x, degree + 1))
    return y
##################### End of Regression Libraries #############

# Starting values
Optimal_Order = 0
Minimum_MSE = 9999
Optimal_Model=None

# Determine minimum MSE for valid set as increasing polynomial order from 1 to 10.

   # Write codes measuring MSE for valid set
for polynomial_order in range(1, 10) :

    model = fit_polynomial(train_x, train_y, polynomial_order)
    Estimated_valid_y = apply_polynomial(model, valid_x)
   
   # For calculating MSE use the library "mean_squared_error" in "sklearn.metrics"
    MSE=mean_squared_error(valid_y, Estimated_valid_y)

    if Minimum_MSE> MSE :
        Optimal_Order = polynomial_order
        Minimum_MSE = MSE
        Optimal_Model = model


print("----------------------","\n")
print("We can choose best polynomial order with MSE of validation set.")
print("Optimal order is ", Optimal_Order)
print("Minimum MSE is ", Minimum_MSE)
print("\n")
print("-----TEST RESULT-----")

#Overlay Regression polynomial along training feature data and test data set (test_x, test,y)
plt.plot(train_x, apply_polynomial(Optimal_Model, train_x),'g')   # Display with lines colored with green (g).
plt.plot(test_x, test_y, 'b.') # Display with dots colored with blue (b).

#Write code for calculating MSE performance of the Optimal regression polynomial.
Estimated_test_y =  apply_polynomial(Optimal_Model, test_x)
MSE_Performance = mean_squared_error(test_y, Estimated_test_y)

print("MSE : ", MSE_Performance)

print("The coefficient of model is ",Optimal_Model.coef_,",  ",Optimal_Model.intercept_)

plt.plot(train_x, apply_polynomial(Optimal_Model, train_x), 'b', markersize=1)
plt.xlabel('Feature values : x')
plt.ylabel('Lable values : y')
plt.grid()
plt.suptitle('Polynomial Regression',fontsize=16)


plt.show()
