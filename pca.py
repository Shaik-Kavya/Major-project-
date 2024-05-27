import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import linear_model, preprocessing, svm ,tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import warnings
# ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
forest_fires = pd.read_csv(r"C:\Users\Kavya shaik\Downloads\myproject\forestfires.csv")
# cheking null/missing values
forest_fires.isnull()
# replacing months to numeric values

forest_fires.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)

#replacing days to numbers
forest_fires.day.replace(('sun','mon','tue','wed','thu','fri','sat'),(1,2,3,4,5,6,7), inplace=True)

#extracting independent variables
attribute_list=forest_fires.iloc[:,:-1].values
#extracting dependent variables
area_values=forest_fires.iloc[:,12]
area_values=area_values.astype('int')
#Statistical analysis of dataset
forest_fires.describe()
x_train,x_test,y_train,y_test = train_test_split(attribute_list,area_values, test_size=0.11,random_state=69)
accuracy_values=[] #PCA
pca = PCA()

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_
#Decision tree
decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
predicted_y = decision_tree.predict(x_test)
print('\nThe accuracy score : %.2f ' % (accuracy_score(y_test, predicted_y) * 100))
accuracy_values.append(accuracy_score(y_test,predicted_y)*100)

#SVM
svm_model = svm.SVC()
svm_model.fit(x_train, y_train)
predicted_y = svm_model.predict(x_test)

print('\nThe accuracy score : %.2f ' % (accuracy_score(y_test, predicted_y) * 100))
accuracy_values.append(accuracy_score(y_test,predicted_y)*100)
#Random forest

random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)
predicted_y = random_forest.predict(x_test)
print('\nThe accuracy score : %.2f ' % (accuracy_score(y_test, predicted_y) * 100))
accuracy_values.append(accuracy_score(y_test,predicted_y)*100)
#KNC

KNC = KNeighborsClassifier()
KNC.fit(x_train, y_train)
predicted_y = KNC.predict(x_test)
print('\nThe accuracy score : %.2f ' % (accuracy_score(y_test, predicted_y) * 100))
accuracy_values.append(accuracy_score(y_test,predicted_y)*100)


#Logistic Regression

from sklearn.preprocessing import StandardScaler

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Increase max_iter and fit the logistic regression model
logistic_regression = LogisticRegression(max_iter=1000)  # Increase max_iter
logistic_regression.fit(x_train_scaled, y_train)

# Predict and calculate accuracy
predicted_y = logistic_regression.predict(x_test_scaled)
print('\nThe accuracy score : %.2f ' % (accuracy_score(y_test, predicted_y) * 100))
accuracy_values.append(accuracy_score(y_test, predicted_y) * 100)

def generate_plot(title, ticks, dataset, color_number):
    colors = ["slateblue", "mediumseagreen", "tomato"]
    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xticks(np.arange(len(ticks)), ticks, fontsize=10, rotation=30)
    plt.title(title, fontsize = 22)
    plt.bar(ticks, dataset, linewidth=1.2, color=colors[color_number])

ticks = [ "Decision tree","SVM", "Random Forest","KNC","Logistic Regression"]
generate_plot("Accuracy scores after PCA", ticks, accuracy_values, 2) 
