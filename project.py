#import packages dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Reading data from external file
df = pd.read_csv('dataset.csv')
print('Dataframe loaded\n',df.head()) 
print('No of rows as Features:',df.shape[0])
#Checking for any missing values
print("Any missing values",df.isnull().sum())

# Divide data into input and target
target=df['LastUpdated']
cols = df.columns[:3]
print('Columns:',cols)
input_feild = df[cols]
print('Input data shape:',input_feild.shape)
cols1 = df.columns[3:]
print(cols1)
#Details of given data
input_feild.describe()
print(input_feild.isnull().sum())
print(input_feild.nunique())

le = LabelEncoder()
scn_le = le.fit_transform(input_['SystemCodeNumber'].values)
input_ = scn_le[:, np.newaxis]

x_train, x_test, y_train, y_test = train_test_split(input_feild,target,test_size = 0.2, random_state = 0)
df1 = df.iloc[:,[1,2]]


#1.  K-Means cluster
km = KMeans()
y_km = km.fit_predict(df1)
print(y_km)
plt.scatter(df1.iloc[y_km == 0,0], df1.iloc[y_km == 0,1])
plt.scatter(df1.iloc[y_km == 1,0], df1.iloc[y_km == 1,1])
plt.scatter(df1.iloc[y_km == 0,0], df1.iloc[y_km == 0,1])
plt.scatter(df1.iloc[y_km == 1,0], df1.iloc[y_km == 1,1])
plt.scatter(df1.iloc[y_km == 0,0], df1.iloc[y_km == 0,1])
plt.title('KMeans Clustering')
plt.plot()
plt.show()


#2. DBSCAN CLUSTERING
db=DBSCAN(eps=0.2,min_samples=5,metric='euclidean')
y_db=db.fit_predict(df1)
plt.figure()
plt.scatter(df1.iloc[y_db==0,0],df1.iloc[y_db==0,1],c='lightblue',edgecolor='black', marker='o',s=40,label='cluster 1')
plt.scatter(df1.iloc[y_db==1,0],df1.iloc[y_db==1,1],c='red',edgecolor='black', marker='s',s=40,label='cluster 1')
plt.title('DBSCAN Clustering')
plt.legend()
plt.show()

#3. AgglomerativeClustering 
ac=AgglomerativeClustering(n_clusters=3,affinity='euclidean', linkage='complete')
labels=ac.fit_predict(df1)
print('Cluster labels:%s'%labels)
plt.scatter(df1.iloc[labels == 0,0], df1.iloc[labels == 0,1])
plt.scatter(df1.iloc[labels == 1,0], df1.iloc[labels == 1,1])
plt.scatter(df1.iloc[labels == 0,0], df1.iloc[labels == 0,1])
plt.scatter(df1.iloc[labels == 1,0], df1.iloc[labels == 1,1])
plt.scatter(df1.iloc[labels == 0,0], df1.iloc[labels == 0,1])
plt.title('Agglomerative Clustering')
plt.plot()
plt.show()


#1. Linear Regression
X = df[['Capacity']].values
Y = df[['Occupancy']].values
plt.scatter(X,Y)
plt.show()
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.30, random_state = 0)
slr = LinearRegression()
slr.fit(x_train,y_train)
y_predict = slr.predict(x_test)
plt.plot(x_test,y_predict, color = 'r')
plt.scatter(x_test,y_test,color ='g')
plt.title('linear regression')
plt.show()
slr.score(x_test,y_test)

print(mean_squared_error(slr.predict(x_train),y_train))
print(r2_score(slr.predict(x_train),y_train))


#2. Decision Tree Regressor
tree1 = DecisionTreeRegressor(max_depth = 3, random_state = 0)
tree1.fit(X,Y)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], Y[sort_idx], tree1)
plt.scatter(X[sort_idx], Y[sort_idx],c = 'steelblue', edgecolor = 'white', s = 70)
plt.plot(X[sort_idx],tree1.predict(X[sort_idx]), color = 'black', lw = 2)
plt.xlabel('%lower status of population')
plt.ylabel('price in $100s')
plt.title('Decision Tree Regressor')
plt.show()
print(mean_squared_error(tree1.predict(x_train),y_train))
print(r2_score(tree1.predict(x_train),y_train))


#3. Random Forest Regressor
tree2 = RandomForestRegressor(max_depth = 3, random_state = 0)
tree2.fit(X,Y)
sort_idx = X.flatten().argsort()
plt.scatter(X[sort_idx], Y[sort_idx],c = 'steelblue', edgecolor = 'white', s = 70)
plt.plot(X[sort_idx],tree2.predict(X[sort_idx]), color = 'black', lw = 2)
plt.xlabel('%lower status of population')
plt.ylabel('price in $100s')
plt.title('Random Forest Regressor')
plt.show()
print(mean_squared_error(tree2.predict(x_train),y_train))
print(r2_score(tree2.predict(x_train),y_train))


#1.  Decision Tree
tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
tree.fit(x_train, y_train)
    
#2.  Random Forest Classifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(x_train, y_train)
    
#3.  Logistic Regression
log = LogisticRegression(random_state = 0, max_iter = 10000)
log.fit(x_train, y_train)
    
    
# Print the models accuracy on the training data   
print('[1] Decision Tree Classifier Training Accuracy:', tree.score(x_train, y_train))
print('[2] Random Forest Classifier Training Accuracy:', forest.score(x_train, y_train))
print('[3] Logistic Regression Classifier Training Accuracy:', log.score(x_train, y_train))

