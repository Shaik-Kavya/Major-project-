import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import warnings 
warnings.filterwarnings('ignore') 
forest_fires=pd.read_csv(r"C:\Users\Kavya shaik\Downloads\myproject\forestfires.csv")
print(forest_fires.columns)
#mapping
forest_fires['month'] = forest_fires['month'].map({'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct': 10,'nov':11,'dec':12})

forest_fires['day'] = forest_fires['day'].map({'sun':1,'mon':2,'tue':3,'wed':4,'thu':5,'fri':6,'sat':7})
attribute_list=forest_fires[list(forest_fires.columns)[:-1]]
area_values=forest_fires['area']
area_values=area_values.astype('int')
x_train,x_test,y_train,y_test=train_test_split(attribute_list,area_values,test_size=0.11,random_state=69)

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression 
logistic_regression=LogisticRegression() 
logistic_regression.fit(x_train,y_train)
pickle.dump(logistic_regression,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
