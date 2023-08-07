# Importing libraries
import pandas as pd    
import numpy as np
# Reading data to Dataframes
    
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data_check = pd.read_csv('test_result.csv')

#changing column name
data_train = data_train.rename(columns = {'Pclass' : 'TicketClass'})
data_test = data_test.rename(columns = {'Pclass' : 'TicketClass'})

#Removing unused columns
data_train = data_train.drop(['Name','Ticket','Fare','Cabin','Embarked','Age'],axis =1)
data_test = data_test.drop(['Name','Age','Ticket','Fare','Cabin','Embarked'], axis =1)

#Importing LabelEncoder from Sklearn
from sklearn.preprocessing import LabelEncoder
label_encoder_sex = LabelEncoder()

# Transforming sex column values using label Encoder
data_train.Sex  = label_encoder_sex.fit_transform(data_train.Sex)
data_test.Sex = label_encoder_sex.fit_transform(data_test.Sex)

data_train = data_train[['PassengerId','Sex','SibSp','Parch','TicketClass','Survived']]
data_test = data_test[['PassengerId','Sex','SibSp','Parch','TicketClass']]

# Split data into inputs and outputs
X_train = data_train.iloc[:,0:5]   # Inputs
y_train = data_train.iloc[:,5]     # Output (Survived)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(data_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#Input layer with 5 inputs neurons
classifier.add(Dense(units = 3, activation = 'relu', input_dim = 5))
#Hidden layer
classifier.add(Dense(units = 2, activation = 'relu'))
#output layer with 1 output neuron which will predict 1 or 0
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

#getting predictions of test data
prediction = classifier.predict(X_test).tolist()
# list to series
se = pd.Series(prediction)
# creating new column of predictions in data_check dataframe
data_check['check'] = se
data_check['check'] = data_check['check'].str.get(0)


series = []
for val in data_check.check:
    if val >= 0.5:
        series.append(1)
    else:
        series.append(0)
data_check['final'] = series

match = 0
nomatch = 0
for val in data_check.values:
    if val[1] == val[3]:
        match = match +1
    else:
        nomatch = nomatch +1

print(match)
print(nomatch)


