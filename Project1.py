
# coding: utf-8

# In[1]:


#import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from random import randrange, uniform


# In[2]:


#set directory
os.chdir("G:\edwisor")


# In[3]:


#Check current working directory
os.getcwd()


# In[4]:


#load data
train = pd.read_csv("Train_data.csv", sep = ',')
test = pd.read_csv("Test_data.csv", sep = ',')


# In[5]:


#combine datasets
data = train.append(test)
data = data.reset_index(drop=True)


# In[6]:


#check the first five observations of the dataset
data.head()


# In[7]:


#information of variables
data.info()


# In[8]:


#check summary of all continuous variables 
data.describe()


# In[9]:


#Count of unique values in all variables
data.nunique()


# In[10]:


#convert area code
data['area code'] = data['area code'].astype(object)


# In[11]:


#delete variable
del data['phone number']


# In[12]:


#Assigning levels to the categories

lis = []
for i in range(0, data.shape[1]):
    #print(i)
    if(data.iloc[:,i].dtypes == 'object'):
        data.iloc[:,i] = pd.Categorical(data.iloc[:,i])
        
        data.iloc[:,i] = data.iloc[:,i].cat.codes 
        data.iloc[:,i] = data.iloc[:,i].astype('object')
        
        lis.append(data.columns[i])


# In[13]:


data.head()


# In[14]:


#plot bar graph
ax = data['Churn'].value_counts().plot(kind='bar',
                                    figsize=(10,6),
                                    title="Churn")
ax.set_xlabel("Churn")
ax.set_ylabel("Frequency")
plt.show()


# In[15]:


var = data.groupby(['state','Churn']).Churn.count()
var.unstack().plot(kind='bar',stacked=True, figsize=(14,8), color=['green','blue'], grid=False)


# In[16]:


var = data.groupby(['area code','Churn']).Churn.count()
var.unstack().plot(kind='bar',stacked=True, figsize=(14,8), color=['green','blue'], grid=False)


# In[17]:


var = data.groupby(['international plan','Churn']).Churn.count()
var.unstack().plot(kind='bar',stacked=True, figsize=(14,8), color=['green','blue'], grid=False)


# In[18]:


var = data.groupby(['voice mail plan','Churn']).Churn.count()
var.unstack().plot(kind='bar',stacked=True, figsize=(14,8), color=['green','blue'], grid=False)


# In[19]:


#plot histogram of the continuous variables
plt.hist(data['account length'], color = "green")
plt.xlabel('account length')
plt.ylabel("Frequency")


# In[20]:


plt.hist(data['number vmail messages'], color = "blue")
plt.xlabel('Vmail messages')
plt.ylabel("Frequency")


# In[21]:


plt.hist(data['total day minutes'], color = "green")
plt.xlabel('total day minutes')
plt.ylabel("Frequency")


# In[22]:


plt.hist(data['total day calls'], color = "blue")
plt.xlabel('total day calls')
plt.ylabel("Frequency")


# In[23]:


plt.hist(data['total day charge'], color = "green")
plt.xlabel('total day charge')
plt.ylabel("Frequency")


# In[24]:


plt.hist(data['total eve minutes'], color = "blue")
plt.xlabel('total eve minutes')
plt.ylabel("Frequency")


# In[25]:


plt.hist(data['total eve calls'], color = "green")
plt.xlabel('total eve calls')
plt.ylabel("Frequency")


# In[26]:


plt.hist(data['total eve charge'], color = "blue")
plt.xlabel('total eve charge')
plt.ylabel("Frequency")


# In[27]:


plt.hist(data['total night minutes'], color = "green")
plt.xlabel('total night minutes')
plt.ylabel("Frequency")


# In[28]:


plt.hist(data['total night calls'], color = "blue")
plt.xlabel('total night calls')
plt.ylabel("Frequency")


# In[29]:


plt.hist(data['total night charge'], color = "green")
plt.xlabel('total night charge')
plt.ylabel("Frequency")


# In[30]:


plt.hist(data['total intl minutes'], color = 'blue')
plt.xlabel('total intl minutes')
plt.ylabel("Frequency")


# In[31]:


plt.hist(data['total intl calls'], color = 'green')
plt.xlabel('total intl calls')
plt.ylabel("Frequency")


# In[32]:


plt.hist(data['total intl charge'], color = 'blue')
plt.xlabel('total intl charge')
plt.ylabel("Frequency")


# In[33]:


plt.hist(data['number customer service calls'], color = 'green')
plt.xlabel('number customer service calls')
plt.ylabel("Frequency")


# In[34]:


#check missing values
missing_values = pd.DataFrame(data.isnull().sum())
print(missing_values)


# In[35]:


#create a copy of data
df = data.copy()


# In[36]:


df.head(2)


# In[37]:


#boxplot for outliers
plt.boxplot(data['account length'])
plt.xlabel('account length')


# In[38]:


#calculate 99th %ile
a = np.array(data['account length'])
p = np.percentile(a, 99) # return 50th percentile, e.g median.
print(p)


# In[39]:


#replace values in the variable
for i in range(len(data)):
      if  data["account length"].loc[i]>194:
          data["account length"].loc[i]=194


# In[40]:


plt.boxplot(data['account length'])
plt.xlabel('account length')


# In[41]:


plt.boxplot(data['number vmail messages'])
plt.xlabel('number vmail messages')


# In[42]:


a = np.array(data['number vmail messages'])
p = np.percentile(a, 98) # return 50th percentile, e.g median.
print(p)


# In[43]:


for i in range(len(data)):
      if  data["number vmail messages"].loc[i]>41:
          data["number vmail messages"].loc[i]=41


# In[44]:


plt.boxplot(data['number vmail messages'])
plt.xlabel('number vmail messages')


# In[45]:


plt.boxplot(data['total day minutes'])
plt.xlabel('total day minutes')


# In[46]:


#calculating 1st and 99th %ile
a = np.array(data['total day minutes'])
q = np.percentile(a, 1) 
p = np.percentile(a, 99) 
print(q)
print(p)


# In[49]:


#replace values in variables
for i in range(len(data)):
       if  data["total day minutes"].loc[i]<54.2:
          data["total day minutes"].loc[i]=54.2
       if  data["total day minutes"].loc[i]>304.61:
          data["total day minutes"].loc[i]=304.61


# In[50]:


plt.boxplot(data['total day minutes'])
plt.xlabel('total day minutes')


# In[51]:


plt.boxplot(data['total day calls'])
plt.xlabel('total day calls')


# In[52]:


a = np.array(data['total day calls'])
q = np.percentile(a, 1) 
p = np.percentile(a, 99) 
print(q)
print(p)


# In[53]:


for i in range(len(data)):
      if  data["total day calls"].loc[i]<54:
          data["total day calls"].loc[i]=54
      if  data["total day calls"].loc[i]>146:
          data["total day calls"].loc[i]=146


# In[54]:


plt.boxplot(data['total day calls'])
plt.xlabel('total day calls')


# In[55]:


plt.boxplot(data['total day charge'])
plt.xlabel('total day charge')


# In[56]:


a = np.array(data['total day charge'])
q = np.percentile(a, 1) 
p = np.percentile(a, 99) 
print(q)
print(p)


# In[57]:


for i in range(len(data)):
      if  data["total day charge"].loc[i]<9.2:
          data["total day charge"].loc[i]=9.2
      if  data["total day charge"].loc[i]>51.78:
          data["total day charge"].loc[i]=51.78


# In[58]:


plt.boxplot(data['total day charge'])
plt.xlabel('total day charge')


# In[59]:


plt.boxplot(data['total eve minutes'])
plt.xlabel('total eve minutes')


# In[60]:


a = np.array(data['total eve minutes'])
q = np.percentile(a, 1) 
p = np.percentile(a, 99) 
print(q)
print(p)


# In[61]:


for i in range(len(data)):
      if  data["total eve minutes"].loc[i]<80.6:
          data["total eve minutes"].loc[i]=80.6
      if  data["total eve minutes"].loc[i]>318.8:
          data["total eve minutes"].loc[i]=318.8


# In[62]:


plt.boxplot(data['total eve minutes'])
plt.xlabel('total eve minutes')


# In[63]:


plt.boxplot(data['total eve calls'])
plt.xlabel('total eve calls')


# In[64]:


a = np.array(data['total eve calls'])
q = np.percentile(a, 1) 
p = np.percentile(a, 99) 
print(q)
print(p)


# In[65]:


for i in range(len(data)):
       if  data["total eve calls"].loc[i]<54:
          data["total eve calls"].loc[i]=54
       if  data["total eve calls"].loc[i]>147:
          data["total eve calls"].loc[i]=147


# In[66]:


plt.boxplot(data['total eve calls'])
plt.xlabel('total eve calls')


# In[67]:


plt.boxplot(data['total eve charge'])
plt.xlabel('total eve charge')


# In[68]:


a = np.array(data['total eve charge'])
q = np.percentile(a, 1) 
p = np.percentile(a, 99) 
print(q)
print(p)


# In[69]:


for i in range(len(data)):
      if  data["total eve charge"].loc[i]<6.85:
          data["total eve charge"].loc[i]=6.85
      if  data["total eve charge"].loc[i]>27.1:
          data["total eve charge"].loc[i]=27.1


# In[70]:


plt.boxplot(data['total eve charge'])
plt.xlabel('total eve charge')


# In[71]:


plt.boxplot(data['total night minutes'])
plt.xlabel('total night minutes')


# In[72]:


a = np.array(data['total night minutes'])
q = np.percentile(a, 1) 
p = np.percentile(a, 99) 
print(q)
print(p)


# In[73]:


for i in range(len(data)):
      if  data["total night minutes"].loc[i]<81.6:
          data["total night minutes"].loc[i]=81.6
      if  data["total night minutes"].loc[i]>318:
          data["total night minutes"].loc[i]=318


# In[74]:


plt.boxplot(data['total night minutes'])
plt.xlabel('total night minutes')


# In[75]:


plt.boxplot(data['total night calls'])
plt.xlabel('total night calls')


# In[76]:


a = np.array(data['total night calls'])
q = np.percentile(a, 1) 
p = np.percentile(a, 99) 
print(q)
print(p)


# In[77]:


for i in range(len(data)):
      if  data["total night calls"].loc[i]<54.99:
          data["total night calls"].loc[i]=54.99
      if  data["total night calls"].loc[i]>148:
          data["total night calls"].loc[i]=148


# In[78]:


plt.boxplot(data['total night calls'])
plt.xlabel('total night calls')


# In[79]:


plt.boxplot(data['total night charge'])
plt.xlabel('total night charge')


# In[80]:


a = np.array(data['total night charge'])
q = np.percentile(a, 1) 
p = np.percentile(a, 99) 
print(q)
print(p)


# In[81]:


for i in range(len(data)):
      if  data["total night charge"].loc[i]<3.67:
          data["total night charge"].loc[i]=3.67
      if  data["total night charge"].loc[i]>14.31:
          data["total night charge"].loc[i]=14.31


# In[82]:


plt.boxplot(data['total night charge'])
plt.xlabel('total night charge')


# In[83]:


plt.boxplot(data['total intl minutes'])
plt.xlabel('total intl minutes')


# In[84]:


a = np.array(data['total intl minutes'])
q = np.percentile(a, 1) 
p = np.percentile(a, 99) 
print(q)
print(p)


# In[85]:


for i in range(len(data)):
      if  data["total intl minutes"].loc[i]<3.5:
          data["total intl minutes"].loc[i]=3.5
      if  data["total intl minutes"].loc[i]>16.6:
          data["total intl minutes"].loc[i]=16.6


# In[86]:


plt.boxplot(data['total intl minutes'])
plt.xlabel('total intl minutes')


# In[87]:


plt.boxplot(df['total intl calls'])

plt.xlabel('total intl calls')


# In[88]:


a = np.array(data['total intl calls'])
p = np.percentile(a, 97) 
print(p)


# In[89]:


for i in range(len(data)):
      if  data["total intl calls"].loc[i]>10:
          data["total intl calls"].loc[i]=10


# In[90]:


plt.boxplot(data['total intl calls'])
plt.xlabel('total intl calls')


# In[91]:


plt.boxplot(data['total intl charge'])
plt.xlabel('total intl charge')


# In[92]:


a = np.array(data['total intl charge'])
q = np.percentile(a, 1) 
p = np.percentile(a, 99) 
print(q)
print(p)


# In[93]:


for i in range(len(data)):
      if  data["total intl charge"].loc[i]<0.95:
          data["total intl charge"].loc[i]=0.95
      if  data["total intl charge"].loc[i]>4.48:
          data["total intl charge"].loc[i]=4.48


# In[94]:


plt.boxplot(data['total intl charge'])
plt.xlabel('total intl charge')


# In[95]:


plt.boxplot(df['number customer service calls'])
plt.xlabel('Customer service calls')


# In[96]:


a = np.array(data['number customer service calls'])
p = np.percentile(a, 92) 
print(p)


# In[97]:


for i in range(len(data)):
      if  data["number customer service calls"].loc[i]>3:
          data["number customer service calls"].loc[i]=3


# In[98]:


plt.boxplot(data['number customer service calls'])
plt.xlabel('Customer service calls')


# In[99]:


df = data.copy()


# In[100]:


df.head()


# In[101]:


df.describe()


# In[102]:


#storing column names
cnames = ['number vmail messages', 'account length', 'total day minutes',
       'total day calls', 'total day charge', 'total eve minutes',
       'total eve calls', 'total eve charge', 'total night minutes',
       'total night calls', 'total night charge', 'total intl minutes', 'total intl calls','total intl charge',
        'number customer service calls']


# In[103]:


#correlation
df_corr = data.loc[:,cnames]


# In[104]:


#correlation plot
f, ax = plt.subplots(figsize=(12,8))

#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[105]:


#Chisquare test of independence
#Save categorical variables
cat_names = ['state', "area code", "international plan", "voice mail plan"]


# In[106]:


#loop for chi square values
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(data['Churn'], data[i]))
    print(p)


# In[107]:


#drop variables
data = data.drop(['area code', 'total day minutes', 'total eve minutes', 'total night minutes','total intl minutes'], axis=1)


# In[108]:


data.head(2)


# In[109]:


plt.hist(data['account length'], color = "green")
plt.xlabel('account length')
plt.ylabel("Frequency")


# In[110]:


plt.hist(data['number vmail messages'], color = "blue")
plt.xlabel('Vmail messages')
plt.ylabel("Frequency")


# In[111]:


plt.hist(data['total day calls'], color = "blue")
plt.xlabel('total day calls')
plt.ylabel("Frequency")


# In[112]:


plt.hist(data['total day charge'], color = "green")
plt.xlabel('total day charge')
plt.ylabel("Frequency")


# In[113]:


plt.hist(data['total eve calls'], color = "green")
plt.xlabel('total eve calls')
plt.ylabel("Frequency")


# In[114]:


plt.hist(data['total eve charge'], color = "blue")
plt.xlabel('total eve charge')
plt.ylabel("Frequency")


# In[115]:


plt.hist(data['total night calls'], color = "blue")
plt.xlabel('total night calls')
plt.ylabel("Frequency")


# In[116]:


plt.hist(data['total night charge'], color = "green")
plt.xlabel('total night charge')
plt.ylabel("Frequency")


# In[117]:


plt.hist(data['total intl calls'], color = 'green')
plt.xlabel('total intl calls')
plt.ylabel("Frequency")


# In[118]:


plt.hist(data['total intl charge'], color = 'blue')
plt.xlabel('total intl charge')
plt.ylabel("Frequency")


# In[119]:


plt.hist(data['number customer service calls'], color = 'green')
plt.xlabel('number customer service calls')
plt.ylabel("Frequency")


# In[122]:


df = data.copy()


# In[123]:


#Import Libraries for decision tree
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split


# In[124]:


#replace target categories with Yes or No
data['Churn'] = data['Churn'].replace(0, 'No')
data['Churn'] = data['Churn'].replace(1, 'Yes')


# In[125]:


#Divide data into train and test
X = data.values[:, 0:14]
Y = data.values[:,14]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3)


# In[126]:


#Decision Tree
C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)


# In[127]:


#predict new test cases
C50_Predictions = C50_model.predict(X_test)


# In[128]:


#build confusion matrix
CM = pd.crosstab(y_test, C50_Predictions)
(CM)


# In[129]:


#Create dot file to visualise tree  #http://webgraphviz.com/
dotfile = open("pt.dot", 'w')
df = tree.export_graphviz(C50_model, out_file=dotfile, feature_names = data.columns)


# In[130]:


#accuracy of model
accuracy_score(y_test, C50_Predictions)*100


# In[131]:


(90*100)/(90+116)


# In[132]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators = 20).fit(X_train, y_train)


# In[133]:


RF_Predictions = RF_model.predict(X_test)


# In[134]:


#build confusion matrix
# from sklearn.metrics import confusion_matrix 
# CM = confusion_matrix(y_test, y_pred)
CM = pd.crosstab(y_test, RF_Predictions)
CM


# In[ ]:


#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]


# In[ ]:


#check accuracy of model
((TP+TN)*100)/(TP+TN+FP+FN)


# In[ ]:


#False Negative rate 
(FN*100)/(FN+TP)


# In[135]:


#Naive Bayes
from sklearn.naive_bayes import GaussianNB

#Naive Bayes implementation
NB_model = GaussianNB().fit(X_train, y_train)


# In[136]:


#predict test cases
NB_Predictions = NB_model.predict(X_test)


# In[137]:


#Build confusion matrix
CM = pd.crosstab(y_test, NB_Predictions)


# In[138]:


CM


# In[ ]:


#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]


# In[ ]:


#check accuracy of model
((TP+TN)*100)/(TP+TN+FP+FN)


# In[ ]:


#False Negative rate 
(FN*100)/(FN+TP)


# In[139]:


#KNN implementation
from sklearn.neighbors import KNeighborsClassifier

KNN_model = KNeighborsClassifier(n_neighbors = 9).fit(X_train, y_train)


# In[140]:


#predict test cases
KNN_Predictions = KNN_model.predict(X_test)


# In[141]:


#build confusion matrix
CM = pd.crosstab(y_test, KNN_Predictions)


# In[142]:


CM


# In[ ]:


#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]


# In[ ]:


#check accuracy of model
((TP+TN)*100)/(TP+TN+FP+FN)


# In[ ]:


#False Negative rate 
(FN*100)/(FN+TP)


# In[143]:


#Let us prepare data for logistic regression
#replace target categories with Yes or No
data['Churn'] = data['Churn'].replace('No', 0)
data['Churn'] = data['Churn'].replace('Yes', 1)


# In[144]:


data.head()


# In[145]:


#Create logistic data. Save target variable first
data_logit = pd.DataFrame(data['Churn'])


# In[146]:


data_logit.head()


# In[147]:


cnames = ['account length',
 'number vmail messages',
  'total day calls',
 'total day charge',
  'total eve calls',
 'total eve charge',
 'total night calls',
 'total night charge',
  'total intl calls',
 'total intl charge',
 'number customer service calls']


# In[148]:


#Add continous variables
data_logit = data_logit.join(data[cnames])


# In[149]:


data_logit.head()


# In[150]:


##Create dummies for categorical variables
cat_names = ["state", "voice mail plan", "international plan"]

for i in cat_names:
    temp = pd.get_dummies(data[i], prefix = i)
    data_logit = data_logit.join(temp)


# In[151]:


data_logit.head()


# In[152]:


Sample_Index = np.random.rand(len(data_logit)) < 0.7

train = data_logit[Sample_Index]
test = data_logit[~Sample_Index]


# In[153]:


#select column indexes for independent variables
train_cols = train.columns[1:67]


# In[154]:


train_cols


# In[155]:


#Built Logistic Regression
import statsmodels.api as sm

logit = sm.Logit(train['Churn'], train[train_cols]).fit()


# In[156]:


logit.summary()


# In[157]:


#Predict test data
test['Actual_prob'] = logit.predict(test[train_cols])

test['ActualVal'] = 1
test.loc[test.Actual_prob < 0.5, 'ActualVal'] = 0


# In[158]:


test.head()


# In[159]:


#Build confusion matrix
CM = pd.crosstab(test['Churn'], test['ActualVal'])


# In[160]:


CM


# In[ ]:


#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]


# In[ ]:


#accuracy
((TP+TN)*100)/(TP+TN+FP+FN)


# In[ ]:


#False negative rate
(FN*100)/(FN+TP)

