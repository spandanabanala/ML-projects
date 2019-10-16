import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_absolute_error
X=pd.read_csv(r'C:\Users\spanda\Desktop\MLDL\day2\tcd ml 2019-20 income prediction training (with labels).csv')

df=pd.DataFrame(X)

df.head()
df=df.drop(['Wears Glasses','Hair Color'], axis=1)
df.head()
df=df.replace('invalid',np.nan)

df['Gender']=df['Gender'].replace('0','unknown')

col=['Gender','University Degree','Profession','Country']
df.loc[:,col] = df.loc[:,col].replace('0',np.nan)

df.loc[:,:] = df.loc[:,:].ffill()


labelencoder_X=LabelEncoder()
df['Profession']=labelencoder_X.fit_transform(df['Profession'])

onehotencoder = ce.OneHotEncoder(cols=['Gender','University Degree','Country'])
df= onehotencoder.fit_transform(df)

reg = RandomForestRegressor()
df=df.drop(['Instance'], axis=1)
x=df.loc[:, df.columns !='Income in EUR']
y=df['Income in EUR' ] 

reg.fit(x, y)
X1=pd.read_csv(r'C:\Users\spanda\Desktop\MLDL\day2\tcd ml 2019-20 income prediction test (without labels).csv')
dtf=pd.DataFrame(X1)
print(dtf)
dtf=dtf.drop(['Wears Glasses','Hair Color'], axis=1)
dtf=dtf.replace('invalid',np.nan)
print(dtf)
dtf['Gender']=dtf['Gender'].replace('0','unknown')
dtf.loc[:,col] = dtf.loc[:,col].replace('0',np.nan)
dtf.loc[:,:] = dtf.loc[:,:].ffill()

print(dtf)
dtf['Profession']=labelencoder_X.fit_transform(dtf['Profession'])

dtf.isnull().values.any()
print(dtf['Profession'])
dtf= onehotencoder.transform(dtf)



#linreg.score(x_test,y_test)
#print(linreg.intercept_)
#print(linreg.coef_)
#y=df['Income']
x_test =dtf.drop(columns=['Instance','Income'])
x_test.isnull().values.any()


dtf['Income']= reg.predict(x_test)

dtf.to_csv(r'C:\Users\spanda\Desktop\MLDL\day2\final6.csv')


