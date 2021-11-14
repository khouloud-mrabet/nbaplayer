import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pickle

df = pd.read_csv("clean_data.csv")
df.drop( axis=1,columns='Unnamed: 0' ,inplace=True)


#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)
x = df.iloc[:,:-1]
y = df.iloc[:, -1]

Classifier = Pipeline([("Scaler", StandardScaler()), ("LR", LogisticRegression(C= 10, penalty = 'l2'))])

#Fitting model with trainig data
Classifier.fit(x, y)

# Saving model
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(Classifier, open('C:/Users/user/Desktop/model.pkl','wb'))