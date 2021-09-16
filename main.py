import pandas as pd

titanic = pd.read_csv('minichallenge_titanic.csv')

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
print(titanic)


