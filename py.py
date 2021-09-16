# Load libraries

#!pip install pandas
#!pip install scikit-learn>=0.24.1, <=0.24.2
#!pip install matplotlib>=3.3.4, <=3.4.1             

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

titanic = pd.read_csv ('minichallenge_titanic.csv')

titanic_mean_sex = titanic.groupby(['Sex']).mean().sort_values(
     ['Survived'], ascending=False)
titanic_mean_Pclass = titanic.groupby(['Pclass']).mean(
 ).sort_values(['Survived'], ascending=False)

values = []
for  key in titanic['Age'].index:
    Age = titanic['Age'].loc[key]
    if Age in range(0,18):
         values.append(1)
    elif Age in range(18,65):
         values.append(2)
    elif Age in range(65,999):
         values.append(3)
    else:
         values.append(4)
titanic['Age_cat'] = values

feature_cols = ["PassengerId","Pclass","Sex","SibSp","Parch","Ticket","Fare","Cabin", "Age_cat"]
X = titanic.notna()[feature_cols] # Features
y = titanic.Survived # Target variable

#titanic['Family Members'] = titanic['SibSp'] + titanic['Parch'] + 1


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test|



# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


print(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5), dpi=250)
tree.plot_tree(clf,
               feature_names=feature_cols,
               class_names=["0", "1"],
               filled=True,
               rounded=True);


