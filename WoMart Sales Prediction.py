
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression ,Ridge , Lasso
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score ,KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler ,LabelEncoder
from catboost import CatBoostRegressor
import math
from sklearn.metrics import mean_squared_log_error ,r _score


#Trainning file 
df_train = pd.read_csv('TRAIN ( ).csv')

# Testing file
df_test = pd.read_csv('TEST_FINAL.csv')

df_train .head(2)
df_test .head(2)

# statistics ...
df_train.describe()

# checking null values
df_train.apply(lambda x : len(x.unique()))


#Checking information about the dataset
df_train.info()


#Checking NULL values
df_train.isnull().sum()

# Checking categorical features and confirming whether it has duplicate categories
a , b , c ,d = set(df_train['Store_Type']),set(df_train['Location_Type']) ,set(df_train['Region_Code']),set(df_train['Discount'])
print(a,b,c,d)


# ## Exploratory Data Analysis

sns.catplot(x=df_train['Discount'] ,y= df_train['Sales'], data = df_train ,kind = 'bar')


#Checking sales outcome w.r.t region
sns.catplot(x=df_train['Region_Code'],y=df_train['Sales'],data = df_train , kind = 'bar')


# sales outcome by location type w.r.t region
sns.catplot(x=df_train['Store_Type'],y=df_train['Sales'],kind = 'bar',data = df_train)


sns.catplot(x=df_train['Location_Type'],y=df_train['Sales'],kind = 'bar',data = df_train)

#Checking outliers w.r.t store type 
sns.catplot(x=df_train['Store_Type'],y=df_train['Sales'],kind ='box',data = df_train)


# ## Feature Engineering

#copying training...
train = df_train.copy()


sns.displot(train['Sales'])

## Removing outliers
# IQR
Q1  = np.percentile(train['Sales'],  25 ,
                   interpolation = 'midpoint')
 
Q3  = np.percentile(train['Sales'], 75  ,
                   interpolation = 'midpoint')
IQR = Q3  - Q1 
 
print("Old Shape: ", train.shape)
 
# Upper bound
upper = np.where(train['Sales'] >= (Q1 + . *IQR))
# Lower bound
lower = np.where(train['Sales'] <= (Q3 - . *IQR))
 
''' Removing the Outliers '''
train.drop(upper[ ] ,inplace = True)
train.drop(lower[ ],inplace = True )
 
print("New Shape: ", train.shape)


#checking rows which have no sales 

zero_sales = train[train['Sales']<= ].index
len(zero_sales)

# Dropping zero value rows in sale
train.drop(zero_sales ,inplace = True)


# #  Encoding and Transformation

# Applying one hot encoding with store_type,'Discount' ,location_type

train = pd.get_dummies(train ,columns=['Store_Type','Location_Type','Discount'])

#Encoding Region codes
reg_encode= LabelEncoder()
train['Region_Code'] = reg_encode.fit_transform(train['Region_Code'])

#Converting date column to date time object
train['Date'] = pd.to_datetime(train['Date'])

# Extracting month from date 
train['Month'] = train['Date'].dt.month

# #Extracting days from date 
train['sale_weekdays'] = train['Date'].dt.dayofweek

#sale on weekend and week day apart
train['sale_week'] = train['sale_weekdays'].apply(lambda x :   if x >=  else   )


sns.catplot(x=train['sale_weekdays'],y=train['Sales'],kind = 'bar',data = train)

#Sale on weekdays vs weekend
sns.catplot(x=train['sale_week'],y=train['Sales'],kind = 'bar',data = train)


# Months vs sale visualisation
sns.catplot(x=train['Month'],y=train['Sales'],kind = 'bar',data = train)

#sart date
train['Date'].min()

#end date
train['Date'].max()

# Dropping unnecessary columns
train.drop(['ID','Date'	,'sale_weekdays'] ,axis =   ,inplace = True)

#moving sales column to the last position

move_train = train.pop('Sales')
train.insert(len(train.columns),'Sales' ,move_train)

# Finding correlation
train.corr()

# # Test Data Preprocessing

#copying data
test = df_test.copy()

test .head(2)

# Checking null values
test.isnull().sum()

# Info
df_test.info()

# Applying one hot encoding with store_type,'Discount' ,location_type
test = pd.get_dummies(test ,columns=['Store_Type','Location_Type','Discount'])

#Encoding Region codes
reg_encode= LabelEncoder()
test['Region_Code'] = reg_encode.fit_transform(test['Region_Code'])

#Converting date column to date time object
test['Date'] = pd.to_datetime(test['Date'])

# Extracting month from date 
test['Month'] = test['Date'].dt.month

# #Extracting days from date 
test['sale_weekdays'] = test['Date'].dt.dayofweek

#sale on weekend and week day apart
test['sale_week'] = test['sale_weekdays'].apply(lambda x :   if x >=  else   )

#start date
test['Date'].min()

#end date
test['Date'].max()

# Dropping unnecesaary columns
test.drop(['ID','Date'	,'sale_weekdays'] ,axis =   ,inplace = True)


test .head(2)


# # Finding order columns in test data by training the models 

train.columns

#Selecting only those column which are present in training data
columns = ['Store_id', 'Region_Code', 'Holiday', 
       'Store_Type_S ', 'Store_Type_S ', 'Store_Type_S ', 'Store_Type_S ',
       'Location_Type_L ', 'Location_Type_L ', 'Location_Type_L ',
       'Location_Type_L ', 'Location_Type_L ', 'Discount_No', 'Discount_Yes',
       'Month', 'sale_week']

        
#Features columns for predicting order
ord_train = train[columns]

#Target columns for order
ord_target = train['#Order']


#Initialising....
reg = LinearRegression()

#Fitting the model
reg.fit(ord_train,ord_target)


#Predicting order 
y_pred = reg.predict(test)

#Converting it into integer as order cannot be float value 
y_pred = [int(i) for i in y_pred]

#Merging into the test data ...
test.insert( ,'#Order' ,y_pred)

test .head(2)


# # Training models and evaluating it's performance

# Feature column selection
X = train.iloc[:,:- ]

#Target column
y = train.loc[:,'Sales']


# This 'test' dataframe is given your problem satement to get prediction 
    # train the model
def salesPrediction(model,model_name , X, y ,test): 
    model.fit(X, y)
    
    # predict the training set
    pred = model.predict(X)
    
    # perform cross-validation
    folds = KFold(n_splits =  , shuffle = True, random_state =    )
    cv_score = cross_val_score(model, X, y , scoring='r ', cv=folds)
    cv_score = np.mean(cv_score)
    
    print("Model Performance Synapsis")
    print("CV Score:", cv_score)
    
    
    #Submission prediction ..
    predict_submit = model.predict(test) 
    predict_submit= [int(i) for i in predict_submit]
    submission = pd.read_csv('SAMPLE.csv')
    submission["Sales"] = predict_submit
    submission.to_csv(f'sub_{model_name}.csv', index=False)


# Linear Regression
lm = LinearRegression(normalize=True)
salesPrediction(lm,'Linear' ,X , y ,test)


#Lasso regression
laso_reg = Lasso(normalize=True)
salesPrediction(laso_reg,'Laso' ,X , y ,test)


# Ridge Regression
rid_reg = Ridge(normalize=True)
salesPrediction(rid_reg,'Ridge' ,X , y ,test)

# Random forest Regressor
rfg = RandomForestRegressor()
salesPrediction(rfg,'randForest' ,X , y ,test)


# Cat Boost Regressor
cat_model = CatBoostRegressor(loss_function="RMSE")
salesPrediction(cat_model,'catBoost' ,X , y ,test)

# Create a dataframe of feature importance 
df_feature_importance = pd.DataFrame(cat_model.get_feature_importance(prettified=True))

#plotting feature importance
plt.figure(figsize=( 12 , 6  ))

feature_plot= sns.barplot(x="Importances", y="Feature Id", data=df_feature_importance,palette="cool")

plt.title('features importance')


   




