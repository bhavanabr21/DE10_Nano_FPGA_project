#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
#from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("DATA.csv")
data=data.drop(['Latitude','Longitude'],axis=1)
data = np.array(data)


X = data[:,:-1]
y = data[:,10]
y = y.astype('int')
X = X.astype('float')
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = RandomForestClassifier(n_estimators=12)
#log_reg =LogisticRegression()
knn =XGBClassifier()



knn.fit(X_train, y_train)
print(knn.predict(X_test))

pickle.dump(knn,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


# In[6]:


#import forest_fire
import pickle
import numpy as np
import pandas as pd
model= pickle.load(open('model.pkl','rb'))

#final=[np.array([47,37,31,17,73,35,20,0,30.3,30.2])]
#final=[np.array([67,53,47,41,72,39,36,3,30.1,30])]
final=[np.array([59,41,45,35,93,55,10,0,30.3,30.1])]
print(final)
prediction=model.predict_proba(final)
output="{0:.2f}".format(prediction[0][1],2)

if output>str(0.5):
       print('Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output))
else:
       print('Your Forest is safe.\n Probability of fire occuring is {}'.format(output))


# In[7]:


import csv 
file = open("without_avg_parray.csv")
numpy_array = np.loadtxt(file, delimiter=",")
probability_array=[]
model= pickle.load(open('model.pkl','rb'))
for row in numpy_array:
    prediction=model.predict_proba([row])
    output="{0:.2f}".format(prediction[0][1],2)
    probability_array.append(output) 

print(len(probability_array),"\n")
print(probability_array)


# In[10]:


import pandas as pd 
import numpy as np

df = pd.read_csv("DATA.csv")
df["fire probability"] = probability_array
df.to_csv("mapping_file.csv", index=False)


# In[11]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon


# In[12]:


ca_map = gpd.read_file("tl_2016_06_cousub/tl_2016_06_cousub.shp")
stats=pd.read_csv("mapping_file.csv")
stats.head(80)


# UTTARAKHAND

# In[1]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon


# In[2]:


ukh_map = gpd.read_file("uttarakhand/india-village-census-2001-UK.shp")
stats=pd.read_csv("for_map.csv")


# In[3]:


ukh_map


# In[4]:


fig,ax=plt.subplots(figsize=(15,15))
ukh_map.plot(ax=ax)
plt.savefig('Map')


# In[5]:


stats


# In[6]:


stats=stats[['Latitude','Longitude','Fire']]


# In[7]:


stats


# In[8]:


crs = {'init':'EPSG:4326'}
geometry = [Point(xy) for xy in zip(stats['Longitude'], stats['Latitude'])]
geo_df = gpd.GeoDataFrame(stats, 
                          crs = crs, 
                          geometry = geometry)


# In[9]:


geo_df.head()


# In[11]:


fig, ax = plt.subplots(figsize = (10,10))
ukh_map.to_crs(epsg=4326).plot(ax=ax, color='lightgrey')
geo_df.plot(ax=ax,markersize = 10)
ax.set_title('uttarakhand')


# In[12]:


geo_df['temp'] = np.log(geo_df['Fire'])
fig, ax = plt.subplots(figsize = (15,15))
ukh_map.to_crs(epsg=4326).plot(ax=ax, color='green')
orig_map=plt.cm.get_cmap('autumn')
reversed_map = orig_map.reversed()
geo_df.plot(column = 'Fire', ax=ax, cmap = reversed_map,
            legend = True, legend_kwds={'shrink': 0.5, 'label':"Probability of fire occurance",}, 
            markersize =20)
ax.set_xlim([78.5, 79.5])
ax.set_ylim([29.2, 29.8])
ax.set_title('uttarkhand fire hotspots')
plt.savefig('Heat Map_uttarakhand')


# In[ ]:




