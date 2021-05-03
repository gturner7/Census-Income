#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# # EDA For DataSet Including Outliers

# In[28]:


salaryWithOuliers = pd.read_csv('C:/Users/Kevin Uruena/Desktop/Final_Data_With_Outliers.csv')


# In[14]:


salaryWithOuliers.shape


# In[30]:


salaryWithOuliers.info()


# In[31]:


#summary statistics
salaryWithOuliers.describe(include='all')


# In[86]:


fig = go.Figure()

fig.add_trace(go.Histogram(x=salaryWithOuliers['INDUSTRY'],
                          xbins=dict(
                          start=0,
                          end=10000,
                          size=10),
                          opacity=1))

fig.update_layout(title_text='Industry Distribution',
                 xaxis_title='Industry',
                 yaxis_title='Count',
                 bargap=0.05,
                 xaxis={'showgrid':False},
                 yaxis={'showgrid':False},
                 template='seaborn',
                 height=600,
                 width=1000)
fig.show()


# In[81]:


fig = go.Figure()

fig.add_trace(go.Histogram(x=salaryWithOuliers['PEIOOCC'],
                          xbins=dict(
                          start=0,
                          end=10000,
                          size=10),
                          opacity=1))

fig.update_layout(title_text='PEIOOCC Distribution',
                 xaxis_title='PEIOOCC',
                 yaxis_title='Count',
                 bargap=0.05,
                 xaxis={'showgrid':False},
                 yaxis={'showgrid':False},
                 template='seaborn',
                 height=600,
                 width=1000)
fig.show()


# In[35]:


#Visualizations
trace1 = go.Bar(x=salaryWithOuliers['A_CLSWKR'].value_counts().index, y=salaryWithOuliers['A_CLSWKR'].value_counts(), 
                text=(salaryWithOuliers['A_CLSWKR'].value_counts()/len(salaryWithOuliers['A_CLSWKR'])*100))

trace2 = go.Bar(x=salaryWithOuliers['A_HGA'].value_counts().index, y=salaryWithOuliers['A_HGA'].value_counts(), 
                text=(salaryWithOuliers['A_HGA'].value_counts()/len(salaryWithOuliers['A_HGA'])*100))

trace3 = go.Bar(x=salaryWithOuliers['A_MJIND'].value_counts().index, y=salaryWithOuliers['A_MJIND'].value_counts(), 
                text=(salaryWithOuliers['A_MJIND'].value_counts()/len(salaryWithOuliers['A_MJIND'])*100))

trace4 = go.Bar(x=salaryWithOuliers['A_USLHRS'].value_counts().index, y=salaryWithOuliers['A_USLHRS'].value_counts(), 
                text=(salaryWithOuliers['A_USLHRS'].value_counts()/len(salaryWithOuliers['A_USLHRS'])*100))



fig = make_subplots(rows=2, cols=2, specs=[[{'type':'bar'},{'type':'bar'}],
                                          [{'type':'bar'},{'type':'bar'}]],
                   subplot_titles=('Work Class Code Distribution','Education Distribution','Occupation Distribution',
                                  'Hours Per Week Distribution'))
fig.append_trace(trace1,1,1)
fig.append_trace(trace2,1,2)
fig.append_trace(trace3,2,1)
fig.append_trace(trace4,2,2)

fig['layout'].update(height=1100, width=1200,title='Feature Analysis')

fig.update_traces(marker_color=['wheat','yellowgreen','plum','pink','lightsalmon','lightseagreen','lightcyan','lightcoral','olive','mintcream','mistyrose','cornflowerblue'],
                  textposition='outside',texttemplate='%{text:.4s}%')
fig.show()


# In[60]:


#Visualizations
trace1 = go.Bar(x=salaryWithOuliers['GESTFIPS'].value_counts().index, y=salaryWithOuliers['GESTFIPS'].value_counts(), 
                text=(salaryWithOuliers['GESTFIPS'].value_counts()/len(salaryWithOuliers['GESTFIPS'])*100))

fig = make_subplots(rows=1, cols=1, specs=[[{'type':'bar'}]],
                   subplot_titles=('State Code Distribution'))
fig.append_trace(trace1,1,1)


fig['layout'].update(height=1100, width=1200,title='Feature Analysis')

fig.update_traces(marker_color=['wheat','yellowgreen','plum','pink','lightsalmon','lightseagreen','lightcyan','lightcoral','olive','mintcream','mistyrose','cornflowerblue'],
                  textposition='outside',texttemplate='%{text:.4s}%')
fig.show()


# In[18]:


fig = go.Figure()

fig.add_trace(go.Histogram(x=salaryWithOuliers['A_AGE'],
                          xbins=dict(
                          start=0,
                          end=95,
                          size=1),
                          opacity=1))

fig.update_layout(title_text='Age Distribution',
                 xaxis_title='Age',
                 yaxis_title='Count',
                 bargap=0.05,
                 xaxis={'showgrid':False},
                 yaxis={'showgrid':False},
                 template='seaborn',
                 height=600,
                 width=1000)
fig.show()


# In[69]:


colors=['mediumturquoise','lightgreen','seagreen',"rgb(114, 78, 145)",'palegreen','olive','gold','darkorange']

traces1 = go.Pie(values=salaryWithOuliers['DIV_YN'].value_counts(), labels=salaryWithOuliers['DIV_YN'].value_counts().index, marker_colors=['mediumturquoise','lightgreen','seagreen',"rgb(114, 78, 145)",'palegreen','olive'])

traces2 = go.Pie(values=salaryWithOuliers['PRCITSHP'].value_counts(), labels=salaryWithOuliers['PRCITSHP'].value_counts().index, marker_colors=['lightcyan','cyan','royalblue','darkblue','steelblue','lightblue'])

traces3 = go.Pie(values=salaryWithOuliers['PRDTRACE'].value_counts(), labels=salaryWithOuliers['PRDTRACE'].value_counts().index,marker_colors=['pink','plum','coral','salmon'])
traces4 = go.Pie(values=salaryWithOuliers['A_SEX'].value_counts(), labels=salaryWithOuliers['A_SEX'].value_counts().index, marker_colors=['gold','darkorange'])

fig = make_subplots(rows=2, cols =2, specs=[[{'type':'domain'}, {'type':'domain'}],
                                           [{'type':'domain'},{'type':'domain'}]],
                   subplot_titles=( 'Dividend Distribution','Citizenship Distribution','Race Distribution','Gender Distribution'))

fig.append_trace(traces1,1,1)
fig.append_trace(traces2,1,2)
fig.append_trace(traces3,2,1)
fig.append_trace(traces4,2,2)

fig['layout'].update(height=1000, title='Feature Analysis', titlefont_size=20)
fig.update_traces(hole=.4, pull=[0,0,0.2,0,0], hoverinfo='label+percent', marker_line=dict(color='black', width=2),)

fig.show()


# In[61]:


colors=['mediumturquoise','lightgreen','seagreen',"rgb(114, 78, 145)",'palegreen','olive','gold','darkorange']

traces1 = go.Pie(values=salaryWithOuliers['WEXP'].value_counts(), labels=salaryWithOuliers['WEXP'].value_counts().index, marker_colors=['mediumturquoise','lightgreen','seagreen',"rgb(114, 78, 145)",'palegreen','olive'])

traces2 = go.Pie(values=salaryWithOuliers['CLWK'].value_counts(), labels=salaryWithOuliers['CLWK'].value_counts().index, marker_colors=['lightcyan','cyan','royalblue','darkblue','steelblue','lightblue'])


fig = make_subplots(rows=1, cols =2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                   subplot_titles=( 'WEXP Distribution','CLWK Distribution'))

fig.append_trace(traces1,1,1)
fig.append_trace(traces2,1,2)


fig['layout'].update(height=1000, title='Feature Analysis', titlefont_size=20)
fig.update_traces(hole=.4, pull=[0,0,0.2,0,0], hoverinfo='label+percent', marker_line=dict(color='black', width=2),)

fig.show()


# In[20]:


fig = go.Figure()

fig.add_trace(go.Histogram(x=salaryWithOuliers['PEARNVAL'],
                          xbins=dict(
                          start=0,
                          end=2500000,
                          size=25000),
                          opacity=1))

fig.update_layout(title_text='Salary Distribution',
                 xaxis_title='Salary',
                 yaxis_title='Count',
                 bargap=0.05,
                 xaxis={'showgrid':False},
                 yaxis={'showgrid':False},
                 template='seaborn',
                 height=600,
                 width=1000)
fig.show()


# In[21]:


#Correlation between salary and features
fig, ax = plt.subplots(figsize=(10,10))
fig.suptitle('Correlation between Salary and Predictor Variables',fontsize=20)
ax=sns.heatmap(salaryWithOuliers.corr()[["PEARNVAL"]].sort_values("PEARNVAL"),vmax=1, vmin=-1, cmap="YlGnBu", annot=True, ax=ax);
ax.invert_yaxis()


# In[62]:


fig = px.scatter(salaryWithOuliers, x=salaryWithOuliers['A_HRS1'], y=salaryWithOuliers['PEARNVAL'], title='A_HRS1 VS Salary')
fig.show()


# In[63]:


fig = px.scatter(salaryWithOuliers, x=salaryWithOuliers['INDUSTRY'], y=salaryWithOuliers['PEARNVAL'], title='Industry VS Salary')
fig.show()


# In[80]:


fig = px.scatter(salaryWithOuliers, x=salaryWithOuliers['OCCUP'], y=salaryWithOuliers['PEARNVAL'], title='OCCUP VS Salary')
fig.show()


# In[68]:


fig = px.scatter(salaryWithOuliers, x=salaryWithOuliers['A_MJIND'], y=salaryWithOuliers['PEARNVAL'], title='Occupation VS Salary',
                 size=salaryWithOuliers['A_HGA'])
fig.show()


# In[25]:


#salary and occupation code with education distribution as well
plot = sns.stripplot(x=salaryWithOuliers['A_MJIND'], y=salaryWithOuliers['PEARNVAL'], hue=salaryWithOuliers['A_HGA'], data=salaryWithOuliers, 
              palette='ocean', 
              jitter=True, edgecolor='none', alpha=.20, size = 10)
plot.get_legend().set_visible(False)
sns.despine()


# Drawing the side color bar
normalize = mcolors.Normalize(vmin=salaryWithOuliers['A_HGA'].min(), vmax=salaryWithOuliers['A_HGA'].max())
colormap = cm.ocean

for n in salaryWithOuliers['A_HGA']:
    plt.plot(color=colormap(normalize(n)))

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(salaryWithOuliers['A_HGA'])
plt.colorbar(scalarmappaple)


# In[26]:


fig = px.scatter(salaryWithOuliers, x=salaryWithOuliers['A_HGA'], y=salaryWithOuliers['PEARNVAL'], title='Education VS Salary')
fig.show()


# In[64]:


fig = px.scatter(salaryWithOuliers, x=salaryWithOuliers['A_USLHRS'], y=salaryWithOuliers['PEARNVAL'], title='Hours Per Week VS Salary')
fig.show()


# In[87]:


fig = px.scatter(salaryWithOuliers, x=salaryWithOuliers['GESTFIPS'], y=salaryWithOuliers['PEARNVAL'], title='Location VS Salary')
fig.show()


# In[88]:


fig = px.scatter(salaryWithOuliers, x=salaryWithOuliers['A_AGE'], y=salaryWithOuliers['PEARNVAL'], title='Age VS Salary')
fig.show()


# In[67]:


fig = px.scatter(salaryWithOuliers, x=salaryWithOuliers['PRDTRACE'], y=salaryWithOuliers['PEARNVAL'], title='Race VS Salary')
fig.show()


# In[31]:


fig = px.scatter(salaryWithOuliers, x=salaryWithOuliers['A_SEX'], y=salaryWithOuliers['PEARNVAL'], title='Gender VS Salary')
fig.show()


# In[65]:


fig = px.scatter(salaryWithOuliers, x=salaryWithOuliers['PENATVTY'], y=salaryWithOuliers['PEARNVAL'], title='Country of Origin VS Salary')
fig.show()


# In[33]:


#Identifying Outliers
sns.boxplot(salaryWithOuliers['PEARNVAL']).set(title='Possible Outliers of Salary')


# In[34]:


# Using IQR method to determine upper and lower bounds
Q1 = np.percentile(salaryWithOuliers['PEARNVAL'], 25, 
                   interpolation = 'midpoint') 
  
Q3 = np.percentile(salaryWithOuliers['PEARNVAL'], 75,
                   interpolation = 'midpoint') 
IQR = Q3 - Q1 
print("Q1:",Q1)
print("Q3:",Q3)
print("IQR:",IQR)


# # Algorithms - DataSet Including Ouliers

# In[35]:


#Linear Regression
salaryWithOuliers = pd.read_csv('C:/Users/Kevin Uruena/Desktop/Final_Data_With_Outliers.csv')
y = salaryWithOuliers.iloc[:,-1]
x = salaryWithOuliers.iloc[:,:-1]

print(x.shape)
print(y.shape)


# In[36]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)


# In[37]:


lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)


# In[38]:


predict = lin_reg.predict(x_test)
accuracy = lin_reg.score(x_test, y_test)
print("Linear Regression Accuracy: %f" % (accuracy))


# In[39]:


rmse = np.sqrt(mean_squared_error(y_test, predict))
print("Linear Regression RMSE: %f" % (rmse))


# In[40]:


#Random Forest
random_forest = RandomForestRegressor(random_state = 42) 
random_forest.fit(x_train, y_train)


# In[41]:


y_pred = random_forest.predict(x_test) 
accuracy = random_forest.score(x_test, y_test)
print("Random Forest Accuracy: %f" % (accuracy))


# In[42]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Random Forest RMSE: %f" % (rmse))


# In[43]:


#Gradient Boosting
gradientB = GradientBoostingRegressor(learning_rate = 0.01, max_depth = 8, n_estimators = 500)
gradientB.fit(x_train, y_train)


# In[44]:


predict = gradientB.predict(x_test)
accuracy = gradientB.score(x_test, y_test)
print("Gradient Boosting Accuracy: %f" % (accuracy))


# In[45]:


rmse = np.sqrt(mean_squared_error(y_test, predict))
print("Gradient Boosting RMSE: %f" % (rmse))


# In[46]:


#XGB
data_dmatrix = xgb.DMatrix(data=x,label=y)
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.97, learning_rate = 0.03,
                max_depth = 8, alpha = 0.02, n_estimators = 500,
                           gamma=1.84,
                           reg_lambda = 2.10,
                           min_child_weight=0.0026,
                           subsample=0.6,
                           silent=0,
                           num_round=1024)


# In[47]:


xg_reg.fit(x_train,y_train)


# In[48]:


preds = xg_reg.predict(x_test)
accuracy = xg_reg.score(x_test, y_test)
print("X Gradient Boosting Accuracy: %f" % (accuracy))


# In[49]:


rmse = np.sqrt(mean_squared_error(y_test, preds))
print("X Gradient Boosting RMSE: %f" % (rmse))


# # New DataSet - No Ouliers (range between -20K to 100K)

# In[50]:


salaryNoOutliers= pd.read_csv('C:/Users/Kevin Uruena/Desktop/Final_Data_No_Outliers.csv')


# In[51]:


salaryNoOutliers.shape


# In[52]:


#summary statistics
salaryNoOutliers.describe(include='all')


# In[53]:


fig = go.Figure()

fig.add_trace(go.Histogram(x=salaryNoOutliers['PEARNVAL'],
                          xbins=dict(
                          start=0,
                          end=2500000,
                          size=10000),
                          opacity=1))

fig.update_layout(title_text='Salary Distribution Without Outliers',
                 xaxis_title='Salary',
                 yaxis_title='Count',
                 bargap=0.05,
                 xaxis={'showgrid':False},
                 yaxis={'showgrid':False},
                 template='seaborn',
                 height=600,
                 width=1000)
fig.show()


# In[54]:



plot = sns.stripplot(x=salaryNoOutliers['A_MJIND'], y=salaryNoOutliers['PEARNVAL'], hue=salaryNoOutliers['A_HGA'], data=salaryNoOutliers, 
              palette='ocean', 
              jitter=True, edgecolor='none', alpha=.60)
plot.get_legend().set_visible(False)
sns.despine()

# Drawing the side color bar
normalize = mcolors.Normalize(vmin=salaryNoOutliers['A_HGA'].min(), vmax=salaryNoOutliers['A_HGA'].max())
colormap = cm.ocean

for n in salaryNoOutliers['A_HGA']:
    plt.plot(color=colormap(normalize(n)))

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(salaryNoOutliers['A_HGA'])
plt.colorbar(scalarmappaple)


# In[55]:


#Correlation between salary and features for new dataset
fig, ax = plt.subplots(figsize=(10,10))
fig.suptitle('Correlation between Salary and Predictor Variables - No Outliers',fontsize=20)
ax=sns.heatmap(salaryNoOutliers.corr()[["PEARNVAL"]].sort_values("PEARNVAL"),vmax=1, vmin=-1, cmap="YlGnBu", annot=True, ax=ax);
ax.invert_yaxis()


# In[56]:


#Clustering - defining linkage matrix
train_data, test_data, _ = np.split(salaryNoOutliers.sample(frac=1, random_state = 123),
                                   [int(0.95* len(salaryNoOutliers)), int(len(salaryNoOutliers))])
test_data.to_csv('test-data-small.csv', index=False, header=True, sep=',')


# In[57]:


salaryNoOutliersSmall= pd.read_csv('test-data-small.csv')
salaryNoOutliersSmall.shape


# In[58]:


z = linkage(salaryNoOutliersSmall, 'ward')


# In[59]:


#Plot Dendrogram
plt.figure(figsize = (25, 10))
plt.title('Cluster with All Salary')
plt.ylabel('distance')
dendrogram(
   z,
   labels = salaryNoOutliersSmall.index,
   leaf_rotation = 0.,
   leaf_font_size = 18.,
)
plt.show()


# # Algorithms With No Outliers

# In[60]:


#Linear Regression
y = salaryNoOutliers.iloc[:,-1]
x = salaryNoOutliers.iloc[:,:-1]

print(x.shape)
print(y.shape)


# In[61]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)


# In[62]:


lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)


# In[63]:


predict = lin_reg.predict(x_test)
accuracy = lin_reg.score(x_test, y_test)
print("Linear Regression Accuracy: %f" % (accuracy))


# In[64]:


rmse = np.sqrt(mean_squared_error(y_test, predict))
print("Linear Regression RMSE: %f" % (rmse))


# In[65]:


#Random Forest
random_forest = RandomForestRegressor(random_state = 42) 
random_forest.fit(x_train, y_train)


# In[66]:


y_pred = random_forest.predict(x_test) 
accuracy = random_forest.score(x_test, y_test)
print("Random Forest Accuracy: %f" % (accuracy))


# In[67]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Random Forest RMSE: %f" % (rmse))


# In[68]:


#Gradient Boosting
gradientB = GradientBoostingRegressor(learning_rate = 0.01, max_depth = 8, n_estimators = 500)
gradientB.fit(x_train, y_train)


# In[69]:


predict = gradientB.predict(x_test)
accuracy = gradientB.score(x_test, y_test)
print("Gradient Boosting Accuracy: %f" % (accuracy))


# In[70]:


rmse = np.sqrt(mean_squared_error(y_test, predict))
print("Gradient Boosting RMSE: %f" % (rmse))


# In[71]:


#XGB
data_dmatrix = xgb.DMatrix(data=x,label=y)
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.97, learning_rate = 0.03,
                max_depth = 8, alpha = 0.02, n_estimators = 500,
                           gamma=1.84,
                           reg_lambda = 2.10,
                           min_child_weight=0.0026,
                           subsample=0.6,
                           silent=0,
                           num_round=1024)


# In[72]:


xg_reg.fit(x_train,y_train)


# In[73]:


preds = xg_reg.predict(x_test)
accuracy = xg_reg.score(x_test, y_test)
print("X Gradient Boosting Accuracy: %f" % (accuracy))


# In[74]:


rmse = np.sqrt(mean_squared_error(y_test, preds))
print("X Gradient Boosting RMSE: %f" % (rmse))


# In[75]:



from sklearn.metrics import explained_variance_score

print((explained_variance_score(preds, y_test)))


# In[81]:


#Visualizing XGB
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [50, 50]
plt.show()


# # Experiments - DS With Ouliers
# 

# In[17]:


from sklearn.preprocessing import power_transform
salaryWithOuliers = pd.read_csv('C:/Users/Kevin Uruena/Desktop/Final_Data_With_Outliers.csv')


# In[18]:


fig, ax = plt.subplots(1, 3, figsize=(15,5))
sns.distplot(salaryWithOuliers['PRDTRACE'], ax=ax[0])
sns.distplot(salaryWithOuliers['PEARNVAL'], ax=ax[1])
sns.distplot(salaryWithOuliers['A_MJIND'], ax=ax[2])
plt.tight_layout()
plt.show()

