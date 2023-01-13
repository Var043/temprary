# imports the necessary libraries for data manipulation 
# and visualization (numpy, pandas, and matplotlib), and 
# machine learning (sklearn). 

import numpy as np 
import pandas as pd
import pickle
from matplotlib import pyplot as plt


# reads a csv file called 'LR-1' and 
# saves it to a pandas dataframe 'df' 
df=pd.read_csv(r"LR-1.csv")

# creates an instance of the Logistic Regression model 
# and assigns it to the variable 'model'
from sklearn.linear_model import LogisticRegression
model =LogisticRegression()


# separates the dataframe columns 'gre', 'gpa', 'rank' and 
# saves them as a numpy array 'x' and 
# the column 'admit' is saved as 'y' which is dependent variable
x=df[['gre', 'gpa', 'rank']].to_numpy()
y=df.admit


# 'model' is fit to the data 'x' and 'y' using the fit() method
model.fit(x,y)

# the score of the model is printed using the score() method which 
# returns the mean accuracy on the given test data and labels.
print(model.score(x,y))

# predict_proba() returns the probability of the sample 
# for each class in the model.
print(model.predict_proba(x))

# the predict() method to predict the probability of the given sample [789,3.43,1].
print(model.predict([[789,3.43,1]]))


# then uses the pickle library to save the model to a file named "model.pkl" and 
# later on it loads the model from the same file using the load() method of pickle
pickle.dump(model, open("model.pkl", "wb"))
model=pickle.load(open('model.pkl','rb'))