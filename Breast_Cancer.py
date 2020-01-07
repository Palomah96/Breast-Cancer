#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.datasets import load_breast_cancer #This lib for importing the breast cancer data or we can import it from UCI
import seaborn as sns 
from sklearn.model_selection import train_test_split #For Splitting the data 
from sklearn import svm #Import the SVM model
from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix
%matplotlib inline

# 1/  Load data
cancer = load_breast_cancer()

# 2/ Getting some Information about the data 
print("Features: ", cancer.feature_names) # print the names of the 30 features
print("Labels: ", cancer.target_names)   # print the label type of cancer('malignant' 'benign')
print(cancer.data[0:5]) # print the cancer data features (top 5 records)
print(cancer.target)  # print the cancer labels (0:malignant, 1:benign) Target is the type of cancer 

#Put our data into a data frame 
df_cancer=pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns=np.append(cancer['feature_names'],['target']))
df_cancer.head() #Print the dataframe 
#Visualize the relationship between our features : we pick only 5 features 
sns.pairplot(df_cancer,hue='target',vars=['mean radius','mean texture','mean perimeter','mean area','mean smoothness'])

#the correlation between our features
plt.figure(figsize=(20,12))
sns.heatmap(df_cancer.corr(),annot=True)
#The predictor Matrix 
X=df_cancer.drop(['target'],axis=1) #The predictors which are the remaining columns
X.head()
Y =df_cancer['target'] #Is the feature we are trying to predict (Output)
Y.head()

# 3/ Splitting the Data 70% training and 30% test 
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109)

# 4 / Creating a model
clf = svm.SVC(kernel='linear') # Linear Kernel

clf.fit(X_train, y_train) # Fit is for Training the model using the training sets

# 5 /Predict the response for test dataset
y_pred = clf.predict(X_test)

#For the confusion Matrix 
cm = np.array(confusion_matrix(y_test,y_pred, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['Is Cancer', 'Is healthy'], columns=['predicted_cancer','predicted_healthy'])
confusion
#Visualize the Confusion Matrix
sns.heatmap(confusion , annot=True)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) #Accuracy Of the model
print("Precision:",metrics.precision_score(y_test, y_pred)) #The Precision 
print("Recall:",metrics.recall_score(y_test, y_pred)) #The recall

#We can also use this function to calculate the accuracy score and the precision and the recall
print(classification_report(y_test,y_pred))


#Using logistic Regression 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#Using Naive_Bayes 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train ,y_train)
