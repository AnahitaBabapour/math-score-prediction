# math-score-prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import warnings

data = pd.read_excel('mathscore.xlsx')
data

data.describe()

data.info()

data.isna().sum()

# Assuming 'EthnicGroup' is the column containing group labels
ethnic_mapping = {
    'group A': 2,
    'group B': 3,
    'group C': 4,
    'group D': 5,
    'group E': 6
}

# Replace the generic group labels with ethnic descriptions
data['EthnicGroup'].replace(ethnic_mapping, inplace=True)
data

# fill the missing values for 'EthicGroup' column by the most frequent value
mst_frq = data['EthnicGroup'].mode()[0]
data['EthnicGroup'].fillna(mst_frq ,inplace= True)


# fill the missing values for 'ParentEduc' column by the common degree they have
mst_frq = data['ParentEduc'].mode()[0]
data['ParentEduc'].fillna(mst_frq , inplace =True)


# fill the missing values for 'ParentMaritalStatus' column by the single status
data['ParentMaritalStatus'].fillna('single' , inplace = True)


# fill the missing values for 'PracticeSport' column by the most common status
mode = data['PracticeSport'].mode()[0]
data['PracticeSport'].fillna(mode , inplace = True)


# fill the missing calues for 'NrSiblings' column by average number of siblings the student have
avg_siblings = data['NrSiblings'].mean()
data['NrSiblings'].fillna(avg_siblings , inplace = True)


# fill the missing calues for 'WklyStudyHours' column by average number of siblings the student have
avg_hours = data['WklyStudyHours'].mode()[0]
data['WklyStudyHours'].fillna(avg_hours , inplace = True)


avg_IsFirstChild = data['IsFirstChild'].mode()[0]
data['IsFirstChild'].fillna(avg_IsFirstChild , inplace = True)

data

data["WklyStudyHours"] = data["WklyStudyHours"].str.replace("2024-05-10","5-10")
data["WklyStudyHours"] = data["WklyStudyHours"].fillna("5-10")
#data

data = pd.DataFrame(data)
data['MathScore'] = data['MathScore']/5
data['ReadingScore'] = data['ReadingScore']/5
data['WritingScore'] = data['WritingScore']/5

categorical = ['Gender','ParentEduc','ParentMaritalStatus','PracticeSport', 'IsFirstChild', 'WklyStudyHours']
i=0
while i<6:
    fig = plt.figure(figsize= (10,5),dpi= 70)
    fig.tight_layout()
    
    plt.subplot(1,3,1)
    sns.countplot(x= categorical[i], data= data)
    i+=1
   
    plt.subplot(1,3,2)
    sns.countplot(x= categorical[i], data= data)
    i+=1
    
    plt.show

data['Gender']=data['Gender'].map({'female':2,'male':3})

data['IsFirstChild']=data['IsFirstChild'].map({'no':2,'yes':3})

#data

data['ParentMaritalStatus']=data['ParentMaritalStatus'].map({'married':2, 'single':3, 'widowed':4, 'divorced':5})

data['ParentEduc']=data['ParentEduc'].map({"bachelor's degree":2, "master's degree":3, "associate's degree":4, "some college":5, "high school":6, "some high school":7})

#data

data['PracticeSport']=data['PracticeSport'].map({'regularly':2,'sometimes':3,'never':4})
data['WklyStudyHours']=data['WklyStudyHours'].map({'< 5':2,'5-10':3,'> 10':4})
data

x = data[['Gender','ParentEduc','ParentMaritalStatus','PracticeSport','IsFirstChild',
          'NrSiblings','WklyStudyHours','EthnicGroup','ReadingScore','WritingScore']]
y = data[['MathScore']].values.reshape(-1,1)
#x.isna().sum()

model = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print('x_train shape: ',x_train.shape)
print('y_train shape: ',y_train.shape)
print('x_test shape:',x_test.shape)
print('y_test shape: ',y_test.shape)

i=2
best_result=1
while i<11:
    k_fold = KFold(i)
    results = cross_val_score(model, x, y, cv = k_fold)
    z = np.mean(results)
    if z<best_result :
        best_result=z
        i+=1
    else:
        i+=1
print(results)
print(best_result)

model.fit(x_train,y_train)
print('Intercept: ',model.intercept_)
print('Weights: ',model.coef_)
y_pred = model.predict(x_test)

print('MSE: ',metrics.mean_squared_error(y_test,y_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print('MAE: ',metrics.mean_absolute_error(y_test,y_pred))
print('R2 score: ',metrics.r2_score(y_test,y_pred))
#print(x)

w0 = 0
weights=[]
def check(Dimension, testsize):
    r2= 0.8423310386334251
    for column in x:
        new_col_name = column + str(Dimension)
        new_col_value = x[column]**Dimension
        x.insert(0,new_col_name,new_col_value)
        X_train,X_test,Y_train,Y_test = train_test_split(x, y, test_size=testsize, random_state=0)
        new_model= LinearRegression()
        new_model.fit(X_train,Y_train)
        Y_pred= new_model.predict(X_test)
        r2_new= metrics.r2_score(Y_test,Y_pred)
        if r2_new<r2 :
            x.drop([new_col_name], axis=1, inplace=True)
        else:
            r2= r2_new
            w0 = new_model.intercept_
            weights = new_model.coef_
    print('r2 score: ',r2)
    print('intercept: ',w0)
    print('coef: ',weights)
check(2,0.2)

i=2
best_result=1
while i<11:
    k_fold = KFold(i)
    results = cross_val_score(model, x, y, cv = k_fold)
    z = np.mean(results)
    if z<best_result :
        best_result=z
        best_k = i
        i+=1
    else:
        i+=1
print(i)
print(results)
print(best_result)

def mathscore():
    EthnicGroup = int(input('''please enter your EthnicGroup ----> A: European students,B: Asian students,
                    C: African students, D: Hispanic/Latino students, E: Middle Eastern students,
    group A: 2,
    group B: 3,
    group C: 4,
    group D: 5,
    group E: 6
    '''))
    IsFirstChild = int(input('are you first child? ----> Yes:3, No:2 '))
    PracticeSport = int(input('are you practice sport? ----> regularly:2, sometimes:3, never:4'))
    ParentEduc = int(input('what is your parent education? ----> bachelors degree:2, masters degree:3, associates degree:4, some college:5, high school:6, some high school:7 '))
    Gender = int(input('what is your gender? ----> female:2, male:3'))
    ParentMaritalStatus = int(input('how is your parent martial status? ----> married:2, single:3, widowed:4, divorced:5'))
    NrSiblings = int(input('how many brothers and sisters do you have? ----> 1 to 7'))
    WklyStudyHours = int(input('how many hours do you study? ----> < 5:2, 5-10:3, > 10:4'))
    ReadingScore = int(input('what is your reading score? ----> 0 t0 20'))
    WritingScore = int(input('what is your writing score? ----> 0 to 20'))
    math = -5.28171398 + (EthnicGroup*EthnicGroup*0.145698259)+(IsFirstChild*IsFirstChild*0.00157803307) + (PracticeSport*PracticeSport*(-0.0415686388))+(ParentEduc*ParentEduc*(-0.03199921330)) + (Gender*2.53717785) + (ParentEduc*0.0641129304)+(ParentMaritalStatus*0.00100250871)+ (PracticeSport*(-0.00319992133))+(IsFirstChild*(0.000315606613))+(NrSiblings*0.00139269024) + (WklyStudyHours*0.171745904) + (EthnicGroup*(-0.993689152)) + (ReadingScore*0.365143692) + (WritingScore*0.575012093)
    result= print('probably your math score will be: ' +str(math))
mathscore()

