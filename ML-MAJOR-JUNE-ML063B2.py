#!/usr/bin/env python
# coding: utf-8

# <h2 id="data">MACHINE LEARNING PROJECT</h2>
# verzeo machine learning main project batch no:ML063B2
# 

# In[1]:


import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import preprocessing 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 10,10


# In[2]:


df =pd.read_csv('C:\\Users\\Shreyas Venishetty\\Desktop\\GOAL STREET\\Machine-Learning-master\\DATA\\Information.csv',engine ='python')


# In[3]:


df.head(3)


# In[4]:


df.info()


# <h2 id="data">Divide the dataset</h2>
# Here the dataset is divided into three df,male and female dataset for easy handling.

# In[5]:


female=df.loc[df.gender=='female']
male = df.loc[df.gender=='male'] 
brand=df.loc[df.gender=='brand']
df = df[df["gender"].isin(['male','female'])]


# <h2 id="data">label encoding</h2>
# Label encoding the gender column

# In[6]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['gender2']=le.fit_transform(df['gender'])


# <h2 id="data_exploration_for_linear_regression_1">Data Exploration and feature selection</h2>
# exploring the data and get rid of unwanted columns.

# In[7]:


df=df[['_trusted_judgments','gender2', 'gender:confidence',
       'profile_yn:confidence', 'fav_number',
       'retweet_count','tweet_count','text']]


# In[8]:


df.info()


# In[9]:


df.columns


# In[10]:


sb.heatmap(df.corr(), annot =True)


# <h2 id="data">COMMON WORDS</h2>
# Lets find the common words used by male and female. Here we did find the most used words by joining all the ['text'] values using the built in function "join" and count the words and found the answers

# common words used by male 

# In[11]:


male.head(2)


# In[12]:


pd.Series(' '.join(male.text).split()).value_counts()


# common words used by female 

# In[13]:


female.head(2)


# In[14]:


pd.Series(' '.join(female.text).split()).value_counts()


# <h3 id="data">therefore the common words used by male and female are "and" , "the" & "to"</h3>

# <h2 id="data">AVERAGE WORDS</h2>
# <p>Lets find the average words used by both gender</p>
# here we create a dictionary and iterate through the the "text" column and find how many times the each words occured after that sum of the count was found and divide it with column value.
# 

# In[15]:


number_of_words={}
for i in range(female.shape[0]):
    words=str(female['text'].values[i]).split(' ')
    for j in words:
        try:
            count=number_of_words[j]
            number_of_words[j]=count+1
        except:
            number_of_words[j]=1
            
number_of_words


# In[16]:


total_no_words=sum(number_of_words.values())


# In[17]:


female.shape


# In[18]:


average_number_words_female=total_no_words/female.shape[0]


# In[ ]:





# In[19]:


number_of_words={}
for i in range(male.shape[0]):
    words=str(male['text'].values[i]).split(' ')
    for j in words:
        try:
            count=number_of_words[j]
            number_of_words[j]=count+1
        except:
            number_of_words[j]=1
            
number_of_words


# In[20]:


total_no_words=sum(number_of_words.values())


# In[21]:


male.shape


# In[22]:


average_number_words_male=total_no_words/male.shape[0]


# In[23]:


print("The average number of words used by FEMALE in thier tweet is:",average_number_words_female)
print("The average number of words used by MALE in thier tweet is:",average_number_words_male)


# <h2 id="data"> Ensemble Machine learning Modelling </h2>

# assign independent and dependent variables. Here gender is the dependent variable 

# In[24]:


X = df[['_trusted_judgments','gender:confidence',
       'profile_yn:confidence', 'fav_number',
       'retweet_count','tweet_count']].values
Y =df[['gender2']].values


# TESTING AND SPLITTING THE DATASET

# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)


# <h1>KNN ALGORITHM</h1>
# finding the knn algorithm accuracy 

# In[26]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)


# In[27]:


from sklearn import metrics
y_pred = knn.predict(X_test)
print("Test set Accuracy: ",metrics.accuracy_score(Y_test, y_pred))


# <h1>SUPPORT VECTOR MACHINE ALGORITHM</h1>
# svm algorithm accuracy

# In[28]:


from sklearn.svm import SVC
svc = SVC(kernel='rbf')
# training Linear Regression model on training data
svc.fit(X_train, Y_train)# The coefficients


# In[29]:


y_pred = svc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print ("TEST ACCURACY:",metrics.accuracy_score(Y_test, y_pred))


# <h1>RANDOM FOREST ALGORITHM</h1>
# finding accuracy of random forest algorithm

# In[30]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
# training Linear Regression model on training data
rfc.fit(X_train, Y_train)


# In[31]:


y_pred = rfc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print ("TEST ACCURACY:",metrics.accuracy_score(Y_test, y_pred))


# <h1>ACCURACY OF THREE ALGORITHM</h1>
# from the above we find each algorithm accuracy and listed down

# <P>accuracy of each algorithm</P>
# <P>1.KNN             :53% </P>
# <P>2.SVM             :52% </P>
# <P>3.RANDOM FOREST   :54% </P>
#     
# <h3>Comparing the Accuracy of all three, the ML algorithms suits best for the given problem is RANDOM FOREST ALGORITHM</h3>  

# In[ ]:




