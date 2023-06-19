import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


dataset = {
    'Id':[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    'Age':['<21','<21','21-35','>35','>35','>35','21-35','<21','<21','>35','<21','21-35','21-35','>35'],
    'Income':['High','High','High','Medium','Low','Low','Low','Medium','Low','Medium','Medium','Medium','High','Medium'],
    'Gender':['Male','Male','Male','Male','Female','Female','Female','Male','Female','Female','Female','Male','Female','Male'],
    'MaritalStatus':['Single','Married','Single','Single','Single','Married','Married','Single','Married','Single','Married','Married','Single','Married'],
    'Buys':['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(dataset,columns=['Id','Age','Income','Gender','MaritalStatus','Buys'])
from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
df['Age'] = l.fit_transform(df['Age'])
df['Income'] = l.fit_transform(df['Income'])
df['Gender'] = l.fit_transform(df['Gender'])
df['MaritalStatus'] = l.fit_transform(df['MaritalStatus'])
df['Buys'] = l.fit_transform(df['Buys'])
df['Age']
x = df.drop(['Buys'],axis = 1)
x
y = df['Buys']
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 10)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(y_test,y_pred))
test_data = pd.DataFrame({'Age': ['<21'], 'Income': ['Low'], 'Gender': ['Female'], 'MaritalStatus': ['Married']})
test_data['Age'] = l.fit_transform(test_data['Age'])
test_data['Income'] = l.fit_transform(test_data['Income'])
test_data['Gender'] = l.fit_transform(test_data['Gender'])
test_data['MaritalStatus'] = l.fit_transform(test_data['MaritalStatus'])
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, rounded=True, feature_names=x.columns, class_names=['No', 'Yes'])
plt.show()
