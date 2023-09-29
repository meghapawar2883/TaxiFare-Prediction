#!/usr/bin/env python
# coding: utf-8

# <div style="color:black;background-color:brown;padding:3%;border-radius:150px 150px;font-size:2em;text-align:center">TaxiFare amount Prediction-> Regression
# </div>

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')


# In[2]:


get_ipython().system('pip install geopy')


# In[4]:


df=pd.read_csv('TaxiFaredata.csv')
df.head()


# In[5]:


df.shape


# In[6]:


df.isnull().count()


# In[7]:


df.nunique()


# In[8]:


df=df.drop(['unique_id'],axis=1)


# In[9]:


df.corr()


# In[10]:


df.dtypes


# In[11]:


df.date_time_of_pickup=pd.to_datetime(df.date_time_of_pickup,errors='coerce')


# In[12]:


df.dtypes


# In[13]:


pd.DataFrame(df.date_time_of_pickup)


# In[14]:


df = df.assign(hour = df['date_time_of_pickup'].dt.hour,
             day = df['date_time_of_pickup'].dt.day,
             dayofweek = df['date_time_of_pickup'].dt.dayofweek,
             month = df['date_time_of_pickup'].dt.month,
             year = df['date_time_of_pickup'].dt.year)


# In[15]:


df.drop(['date_time_of_pickup'],axis=1,inplace=True)
df.head()


# In[16]:


from math import radians, cos, sin, sqrt, asin


# In[17]:


def distance_transform(longitude1, latitude1, longitude2, latitude2):
    travel_distance = []
    
    for i in range(len(longitude1)):
        long1,lati1,long2,lati2 = map(radians,[longitude1[i],latitude1[i],longitude2[i],latitude2[i]])
        dist_long = long2 - long1
        dist_lati = lati2 - lati1
        a = sin(dist_lati/2)**2 + cos(lati1) * cos(lati2) * sin(dist_long/2)**2
        c = 2*asin(sqrt(a))*6371    # 6371 is the radias of earth while measuring the distance of longitude and latitude 
        travel_distance.append(c)
        
    return travel_distance


# In[18]:


df['travel_distance_klm']=distance_transform(df['longitude_of_pickup'].to_numpy(),
                                             df['latitude_of_pickup'].to_numpy(),
                                             df['longitude_of_dropoff'].to_numpy(),
                                             df['latitude_of_dropoff'].to_numpy())


# In[19]:


df.head()


# In[20]:


df.dayofweek.value_counts().sort_values()


# In[21]:


df.describe()


# In[22]:


df=df.loc[(df.amount >= 2.5)]


# In[23]:


df


# In[24]:


df.shape


# In[25]:


plt.figure(figsize=(12,5))
sns.distplot(df['travel_distance_klm'])
plt.title('distribution plot')


# In[27]:


df=df.loc[(df.travel_distance_klm>=1) | (df.travel_distance_klm<=130)]
df


# In[28]:


incorrect_cordinates = df.loc[(df['latitude_of_pickup'] > 90) | (df['latitude_of_pickup'] < -90) |
           (df['latitude_of_dropoff'] > 90) | (df['latitude_of_dropoff'] < -90)|
           (df['longitude_of_pickup'] >180) | (df['longitude_of_pickup'] <-180)|
           (df['longitude_of_dropoff'] >90) | (df['longitude_of_pickup'] <-90)].index


# In[29]:


incorrect_cordinates


# In[30]:


df.drop(incorrect_cordinates, inplace=True)


# In[31]:


df


# In[32]:


plt.figure(figsize=(20,5))
plt.title ('Peek Hours during week days')
sns.countplot(x='hour',data=df.loc[(df['dayofweek']>=0) & (df['dayofweek']<=4)])


# In[33]:


plt.figure(figsize=(20,5))
plt.title ('Peek Hours during week ends')
sns.countplot(x='hour',data=df.loc[(df['dayofweek']>=5) & (df['dayofweek']<=6)])


# In[34]:


# Set the days in the dataset as week days and week ends
week_days = df.loc[(df.dayofweek >= 0) & (df.dayofweek <= 4)]
week_ends = df.loc[(df.dayofweek >= 5) & (df.dayofweek <= 6)]

week_days_fare = week_days.groupby(['hour']).amount.mean().to_frame().reset_index()
week_ends_fare = week_ends.groupby(['hour']).amount.mean().to_frame().reset_index()


# In[35]:


week_days_fare.head()


# In[37]:


wwc = pd.merge(week_days_fare, week_ends_fare, on='hour',suffixes=['_Weekday','_Weekends'])
wwc.head()


# In[38]:


x=np.array(wwc.hour)
y=np.array(wwc.amount_Weekday)
z=np.array(wwc.amount_Weekends)

# Set the figure size, title, x and y labels
plt.figure(figsize=(20,6))
plt.title('Mean fare amount of each hour- Weekdays Vs Weekends')
plt.xlabel('Hours')
plt.ylabel('Mean Fare')

#pass the x,y,z to make the barplot
ax=plt.subplot(1,1,1)
ax.bar(x-0.3 , y , width=0.3,color='gray', align='center', label = 'Week days')
ax.bar(x , z , width=0.3,color='red', align='center', label = 'Week ends')
plt.xticks(range(0,24))
plt.legend()
plt.show()


# In[39]:


plt.figure(figsize=(20,10))
sns.set_style=("darkgrid")
plt.title("Distribution of the fare amount")
plt.xlabel("Fare Amount")
plt.ylabel("Frequency")
plt.xticks(range(0,200,5))

snsplot = sns.kdeplot(df.amount, shade=True)


# In[40]:


plt.figure(figsize=(15,10))
sns.set_style=('darkgrid')
plt.title('Distribution of average distance travel')
plt.xlabel('Mean distance travel')
plt.ylabel('Frequency')
plt.xlim(-10, 200)
plt.xticks(range(0,200,5))

sns.plot = sns.kdeplot(df[df.travel_distance_klm<130].travel_distance_klm, shade=True)


# In[45]:


x=df.drop(['amount'],axis=1)
y=pd.DataFrame(df['amount'])


# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)


# In[49]:


print('shape of x_train',x_train.shape,'\n'
     'shape of x_test',x_test.shape,'\n'
     'shape of y_train',y_train.shape,'\n'
     'shape of y_test',y_test.shape,'\n')


# In[50]:


x_train


# In[51]:


y_train


# In[52]:


from sklearn.ensemble import RandomForestRegressor


# In[53]:


RFR=RandomForestRegressor(n_estimators=100,random_state=10)


# In[54]:


RFR.fit(x_train,y_train)


# In[55]:


y_pred=RFR.predict(x_test)


# In[56]:


from sklearn import metrics


# In[57]:


# MAE
Mean_absolute_error=metrics.mean_absolute_error(y_test,y_pred)
print('mean absolute error',Mean_absolute_error)

#MSE
Mean_Squared_Error=metrics.mean_squared_error(y_test,y_pred)
print('mean square error',Mean_Squared_Error)

#RMSE
Root_Mean_Squared_Error=np.sqrt(metrics.mean_squared_error(y_test,y_pred))
print('root mean square error',Root_Mean_Squared_Error)


# In[58]:


from sklearn.model_selection import GridSearchCV , RandomizedSearchCV , StratifiedKFold, cross_val_score
from scipy.stats import randint


# In[59]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]


# In[60]:


# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid)


# In[61]:


from sklearn.model_selection import RandomizedSearchCV
rf_RandomGrid = RandomizedSearchCV(estimator = RFR, param_distributions = param_grid, cv = 10, verbose=2, n_jobs = 4)


# In[62]:


rf_RandomGrid.fit(x_train, y_train)


# In[63]:


rf_RandomGrid.best_params_


# In[64]:


print (f'Train Accuracy - : {rf_RandomGrid.score(x_train,y_train):.3f}')
print (f'Test Accuracy - : {rf_RandomGrid.score(x_test,y_test):.3f}')


# In[ ]:




