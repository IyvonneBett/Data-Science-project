import pandas
import matplotlib

df = pandas.read_csv('heart.csv')
print(df)
print(df.isnull().sum())
print(df.dtypes)

array = df.values

features = array[:, 0:13]
target = array[:, 13]

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features, target, test_size=0.3, random_state=42)


from sklearn.neighbors import KNeighborsClassifier # model...algorithm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
print('Model finished...')

# ask the model to predict x_test features ...hide the y_test (Loan Status)
predictions = model.predict(X_test)
print(predictions)

#compare predictions and y_test
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))

from sklearn.metrics import classification_report
print(classification_report(Y_test, predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, predictions))

import matplotlib.pyplot as plt
#bar chart
df.groupby('age')['trestbps'].mean().plot(kind='bar', color='green')
plt.title('Age vs blood pressure')
plt.xlabel('age')
plt.ylabel('blood pressure')
plt.show()

df.groupby('age')['fbs'].mean().plot(kind='bar', color='green')
plt.title('Age vs blood sugar')
plt.xlabel('age')
plt.ylabel('blood pressure')
plt.show()

df.groupby('age')['chol'].mean().plot(kind='bar', color='green')
plt.title('Age vs cholestral')
plt.xlabel('age')
plt.ylabel('cholestral')
plt.show()

df.groupby('age')['thalach'].mean().plot(kind='bar', color='green')
plt.title('Age vs heart rate')
plt.xlabel('age')
plt.ylabel('blood pressure')
plt.show()



print(plt.style.available)
plt.style.use('seaborn-dark-palette')

figure, ax = plt.subplots()
ax.hist(df['age'], color = 'blue')
ax.set_xlabel('age')
ax.set_ylabel('freq.')
ax.set_title('Age')
plt.show()





