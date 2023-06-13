# Mini-Project
# Car Prediction Model Using Data Science
```py
import pandas as pd
import seaborn as sns
df = pd.read_csv("/content/data.csv")
df
df.info()
df.isnull().sum()
df['Engine Fuel Type']=df['Engine Fuel Type'].fillna(df['Engine Fuel Type']).mode()[0]
df['Engine Cylinders']=df['Engine Cylinders'].fillna(df['Engine Cylinders']).mean()
df['Engine HP']=df['Engine HP'].fillna(df['Engine HP']).mean()
df['Market Category']=df['Market Category'].fillna(df['Market Category']).mode()[0]
df.isnull().sum()
df.describe()
import matplotlib.pyplot as plt
plt.figure(figsize=(20,7))
sns.boxplot(data = df)
plt.show()
plt.figure(figsize=(20,7))
cols = ['Year','highway MPG','city mpg','Popularity','Number of Doors']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("After removing outliers")
sns.boxplot(data=df)
plt.show()
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(df.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
plt.figure(figsize=(50,7))
sns.lineplot(x=df['Make'],y=df['Number of Doors'],data=df,marker='o')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(50,22))
sns.barplot(x='Make',y='city mpg',data=df)
plt.show()
plt.figure(figsize=(50,22))
sns.barplot(y='highway MPG',x='Make',data=df)
plt.xticks(rotation=90)
plt.show()
hig_profit=df.loc[:,["Make","Popularity"]]
hig_profit=hig_profit.groupby(by=['Make']).max().sort_values(by=['Popularity'])
plt.figure(figsize=(40,15))
sns.barplot(x=hig_profit.index,y='Popularity',data=hig_profit)
plt.xticks(rotation=90)
plt.title("Make vs Popularity	")
plt.show()
hig=df.loc[:,["Make","MSRP"]]
hig=hig.groupby(by=['Make']).mean().sort_values(by=['MSRP'])
plt.figure(figsize=(40,15))
sns.barplot(x=hig.index,y='MSRP',data=hig)
plt.xticks(rotation=90)
plt.title("Make vs MSRP")
plt.show()
sns.jointplot(data=df, x="Popularity", y="Year", kind="hex")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
features = ['Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg']
X_train, X_test, y_train, y_test = train_test_split(df[features], df['MSRP'], test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('Root Mean Squared Error:', rmse)
Hp = int(input("Enter the Hp:"))
Cylinder = int(input("Enter the Cylinder:"))
MPG = int(input("Enter the MPG:"))
cityMPG = int(input("Enter the cityMPG:"))
new_car = pd.DataFrame([[Hp,Cylinder,MPG,cityMPG]], columns=features)
predicted_price = model.predict(new_car)
print(f"Predicted MSRP of the New Car: {predicted_price[0]}")
```

# OUTPUT:
![Screenshot 2023-06-13 210655](https://github.com/Nagul71/Mini-Project/assets/118661118/e9fff5de-b265-458a-8a64-3abb3349f2b6)

![Screenshot 2023-06-13 210703](https://github.com/Nagul71/Mini-Project/assets/118661118/710c5db1-f0ae-4394-bf3e-a9db710749b0)

![Screenshot 2023-06-13 210708](https://github.com/Nagul71/Mini-Project/assets/118661118/9609d35a-8d58-4130-8e8a-8d62292be1aa)

![Screenshot 2023-06-13 210718](https://github.com/Nagul71/Mini-Project/assets/118661118/38b17bc3-2da9-4d59-91c2-8a5607c85637)

![Screenshot 2023-06-13 210724](https://github.com/Nagul71/Mini-Project/assets/118661118/e27ad3b6-49ae-4252-897f-baddce6e4b19)

![Screenshot 2023-06-13 210736](https://github.com/Nagul71/Mini-Project/assets/118661118/4d43890a-c1a2-46c3-9be7-1d86b23d7fdf)

![Screenshot 2023-06-13 210750](https://github.com/Nagul71/Mini-Project/assets/118661118/c2a5a919-6f3c-4a88-8e48-4e3893897ec7)

![Screenshot 2023-06-13 210804](https://github.com/Nagul71/Mini-Project/assets/118661118/a724a83c-4a8b-4b6d-93dc-e51bcaaaea51)

![Screenshot 2023-06-13 210832](https://github.com/Nagul71/Mini-Project/assets/118661118/cb2afa2f-5918-404b-9f07-8411d596f3ac)

![Screenshot 2023-06-13 210845](https://github.com/Nagul71/Mini-Project/assets/118661118/78e8bb0d-1f75-472b-8f1f-2e7872da5ccf)

![Screenshot 2023-06-13 210857](https://github.com/Nagul71/Mini-Project/assets/118661118/4cd337c4-71e3-4197-ab56-39b65b09e429)

![Screenshot 2023-06-13 212445](https://github.com/Nagul71/Mini-Project/assets/118661118/fb7be0cf-59ec-4892-b706-235fb60f17ee)


