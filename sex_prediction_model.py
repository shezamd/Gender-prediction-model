import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

###1. Formulating a Question
#Can we predict sex with education level and income??


###2. Finding and Understanding the Data
df = pd.read_csv("/Users/sheza/desktop/capstone_starter/profiles.csv")
#print(df.columns)
#print(df.job.head())
#print(df['religion'].value_counts())


###3. Cleaning the Data and Feature Engineering
#Augment data

#print(df.sex.value_counts())
sex_mapping = {"m": 0, "f": 1}
df["sex_code"] = df.sex.map(sex_mapping)
#print(df.sex_code.value_counts())

#print(df.education.value_counts())
education_mapping = {'graduated from ph.d program' : 0, 'graduated from med school' : 1, 'graduated from law school' : 2, 'graduated from masters program' : 3, 'graduated from college/university' : 4, 'graduated from two-year college' : 5, 'graduated from space camp' : 6, 'ph.d program' : 7, 'med school' : 8, 'law school' : 9, 'masters program' : 10, 'college/university' : 11, 'two-year college' : 12, 'space camp' : 13, 'working on ph.d program' : 14, 'working on med school' : 15, 'working on law school' : 16, 'working on masters program' : 17, 'working on college/university' : 18, 'working on two-year college' : 19, 'working on space camp' : 20, 'dropped out of ph.d program' : 21, 'dropped out of med school' : 22, 'dropped out of law school' : 23, 'dropped out of masters program' : 24, 'dropped out of college/university' : 25, 'dropped out of two-year college' : 26, 'dropped out of space camp' : 27, 'graduated from high school' : 28, 'high school' : 29, 'working on high school' : 30, 'dropped out of high school' : 31}
df["education_code"] = df.education.map(education_mapping)
#print(df.education_code.value_counts())

#print(df.income.value_counts())
income_mapping = { 1000000 : 0, 500000 : 1, 250000 : 2, 150000 : 3, 100000 : 4, 80000 : 5, 70000 : 6, 60000 : 7, 50000 : 8, 40000 : 9, 30000 : 10, 20000 : 11, -1 : 12}
df["income_code"] = df.income.map(income_mapping)
#print(df.income_code.value_counts())

#Removing unnecessary columns
df = df.dropna(subset = ['education_code', 'income_code', 'sex_code'])

#Normalize your Data!
from sklearn import preprocessing

feature_data = df[['education_code', 'income_code']]
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

#labels are already between 1 and 0, hence normalized
labels = df['sex_code']

###4. Choosing a Model & 5. Tuning and Evaluating
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, labels, train_size = 0.8, random_state =  90)

# code to find best k value

# accuracies = []
# for k in range(41, 101):
#     sex_prediction_model = KNeighborsClassifier(n_neighbors = k)
#     sex_prediction_model.fit(x_train, y_train)
#     score = sex_prediction_model.score(x_test, y_test)
#     accuracies.append(score) 

# print(max(accuracies))
# k_list = range(41, 101)
# plt.plot(k_list, accuracies)
# plt.xlabel("k")
# plt.ylabel("Score ")
# plt.title("Sex Classifier Accuracy")
# plt.show()

# print(max(accuracies))

sex_prediction_model = KNeighborsClassifier(n_neighbors = 2)
sex_prediction_model.fit(x_train, y_train)
score = sex_prediction_model.score(x_test, y_test)
prediction = sex_prediction_model.predict([[0, 0], [4, 4]])
#print(prediction)
print('The sex prediction model has an accuracy of: %3f' % (score * 100))