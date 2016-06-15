import pandas as pd #data analytics
import numpy as np #scientific computing
from pandas import Series,DataFrame

data_train = pd.read_csv("Train.csv")

print
print "Attributes"
print
print data_train.columns

print
print "Data Type"
print
print data_train.info()

print
print "Data Description"
print
print data_train.describe()

###########################################
# Data Visualization
###########################################

import matplotlib.pyplot as plt
fig = plt.figure()
fig.set(alpha=0.2)

#data_train.Survived.value_counts().plot(kind='bar')# plots a bar graph of those who surived vs those who did not.
#plt.title(u"Survival (1 is survived)") # puts a title on our graph
#plt.ylabel(u"Passenger Amount")
#plt.show()
#
#data_train.Pclass.value_counts().plot(kind="bar")
#plt.ylabel(u"Passenger Amount")
#plt.title(u"Passenger Class")
#plt.show()

#plt.scatter(data_train.Survived, data_train.Age)
#plt.ylabel(u"Age")                         # sets the y axis lable
#plt.grid(b=True, which='major', axis='y') # formats the grid line style of our graphs
#plt.title(u"Age vs Survival (1 is survived)")
#plt.show()

#data_train.Age[data_train.Pclass == 1].plot(kind='kde')   # plots a kernel desnsity estimate of the subset of the 1st class passanges's age
#data_train.Age[data_train.Pclass == 2].plot(kind='kde')
#data_train.Age[data_train.Pclass == 3].plot(kind='kde')
#plt.xlabel(u"Age")# plots an axis lable
#plt.ylabel(u"Density")
#plt.title(u"Age Distribution of Different Classes")
#plt.legend((u'1st Class', u'2nd Class',u'3rd Class'),loc='best') # sets our legend for our graph.
#plt.show()

#data_train.Embarked.value_counts().plot(kind='bar')
#plt.title(u"How many passengers emarked from different port?")
#plt.xlabel(u"Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)")
#plt.ylabel(u"Passenger Amount")
#plt.show()

# Survival of different classes

#Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
#Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
#df=pd.DataFrame({u'Survived':Survived_1, u'Died':Survived_0})
#df.plot(kind='bar', stacked=True)
#plt.title(u"Survivial of Different Classes")
#plt.xlabel(u"Class")
#plt.ylabel(u"Passenger Amount")
#
#plt.show()

#Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
#Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
#df=pd.DataFrame({u'Survived':Survived_1, u'Died':Survived_0})
#df.plot(kind='bar', stacked=True)
#plt.title(u"Survival vs Port")
#plt.xlabel(u"Port of Embarkation")
#plt.ylabel(u"Passenger Amount")
#
#plt.show()

#Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
#Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
#df=pd.DataFrame({u'Male':Survived_m, u'Femal':Survived_f})
#df.plot(kind='bar', stacked=True)
#plt.title(u"Survival vs Gender")
#plt.xlabel(u"Survival")
#plt.ylabel(u"Passenger Amount")
#plt.show()

print
print "About Siblings and Spouses"
g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print df

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({u'Has':Survived_cabin, u'Doesn\'t have':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u"Cabin or not vs Survival")
plt.xlabel(u"Has Cabin or not?")
plt.ylabel(u"Passenger Amount")
plt.show()
