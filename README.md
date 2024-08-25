## NAME : Pradeep RaJ
## REG No : 212222240073
## EX NO - 1
## Date : 
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv('Churn_Modelling.csv')
df.head()

X=df.iloc[:,:-1].values
print(X)

y=df.iloc[:,-1].values
print(y)

print(df.isnull().sum())

print(df.columns)

df=df.drop(['Surname', 'Geography','Gender'], axis=1)
df.fillna(df.mean(),inplace=True)
df.duplicated()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train

X_test

print("Lenght of X_test ",len(X_test))
```


## OUTPUT:
### Dataset
![image](https://github.com/user-attachments/assets/3518aa2a-0fde-403b-8543-168ef3c225d7)

### X values
![image](https://github.com/user-attachments/assets/7447170b-4cc7-4a48-8bc3-aed66a35d236)

### Y values
![image](https://github.com/user-attachments/assets/ed0e0e2e-8dd8-488d-8969-a0a1cb10ecc3)

### Null Values
![image](https://github.com/user-attachments/assets/5fe72b97-09f0-4b96-b4c0-ee21605ea43f)

### Columns
![image](https://github.com/user-attachments/assets/c52e39f0-a530-4235-bfef-8651d4f10671)

### Duplicated
![image](https://github.com/user-attachments/assets/7e3703a4-46f7-4303-842f-4bc0ed7a381e)

### Normalized data
![image](https://github.com/user-attachments/assets/aba16afb-c2bb-4e55-a712-509fde1217c4)

### X_train 
![image](https://github.com/user-attachments/assets/11b7ef94-b1fa-44aa-aeee-b28725889662)

### X_test
![image](https://github.com/user-attachments/assets/80287732-2324-4802-a3f8-a00449ddf750)

### Length of test
![image](https://github.com/user-attachments/assets/44c9b2cc-ae5a-40ed-9bb0-b0c523ea4b24)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


