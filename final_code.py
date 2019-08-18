#!/usr/bin/env python
# coding: utf-8

# In[1]:


######impot needed packages 
import pandas as pd
from ann_visualizer.visualize import ann_viz;
from nnv import NNV
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dropout
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale,normalize,minmax_scale
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense,Activation
from sklearn.datasets import make_circles
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.activations import relu,sigmoid
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Input, Embedding, Flatten
from keras.models import Model
from keras.utils import plot_model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from keras.constraints import maxnorm


# In[2]:


#######importing dataset
df=pd.read_csv('farm_or_fallow.csv')


# In[8]:


df1=df.astype('category')


# In[4]:


######all data informations 
def datainfo(df):
    print("the first 5 row of dataframe is :")
    print(df.head())
    print('\n\n')
    print('##################################################################################################')
    print("the type of entitines  of dataframe are :")
    df.info()
    print('\n\n')
    print('##################################################################################################')
    print("the statistical information of dataframe are :")
    print('\n\n')
    print(df.describe())
    print('\n\n')
    print('##################################################################################################')
    print("the index  of dataframe are :")
    print('\n\n')
    print(df.index)
    print('\n\n')
    print('##################################################################################################')
    print("the column's names of dataframe are :")
    print('\n\n')
    print(df.columns)
    print('\n\n')
    print('##################################################################################################')
    print("the shape of dataframe is:")
    print('\n\n')
    print(df.shape)
    print("Test if there any missing values in DataFrame:")
    print('\n\n')
    print(df.isnull().values.any())


# In[5]:


df.columns


# In[7]:


datainfo(df)


# In[8]:


datainfo(df1)


# In[ ]:


######data preparation 


# In[3]:


#######extract target and feauture for cell 1
X=df.iloc[:,[0,1,10]]
Y=df.iloc[:,19]


# In[17]:


#######extract target and feauture for cell 1
X1s=df1.iloc[:,[0,1,10]]
Y1s=df1.iloc[:,19]


# In[16]:


###################add Y1 to X1 as class column
X1['class']=Y1


# In[12]:


X1s['class']=Y1s


# In[4]:


######data extract for all cells
F=df.iloc[:,0:19]
h=df.iloc[:,19:]


# In[9]:


F1=df1.iloc[:,0:19]
h1=df1.iloc[:,19:]


# In[6]:


######target for all cells
listy=[]

for i in range(0,9):
    y=df.iloc[:,19+i]
    listy.append(y)
    


# In[10]:


listy1=[]

for i in range(0,9):
    y1=df1.iloc[:,19+i]
    listy1.append(y1)


# In[11]:


######features for all cells
listx=[]
for j in range(0,9):
    x=df.iloc[:,[0,1+j,10+j]]
    listx.append(x)


# In[12]:


######features for all cells
listx1=[]
for j in range(0,9):
    x1=df1.iloc[:,[0,1+j,10+j]]
    listx1.append(x1)


# In[ ]:


#####################################exploratory data analysis


# In[22]:


###### Describe the data for cell one 
print('Dataset stats: \n', X1.describe())


# In[23]:


###### Describe the data for cell one 
print('Dataset stats: \n', X1.describe())


# In[24]:


df.describe().transpose()


# In[25]:


df1.describe().transpose()


# In[26]:


########data distrubition class cor choices 
a=['O_Choice_c1','O_Choice_c2', 'O_Choice_c3', 'O_Choice_c4', 'O_Choice_c5','O_Choice_c6', 'O_Choice_c7', 'O_Choice_c8', 'O_Choice_c9']
n=len(a)
fig,ax = plt.subplots(n,1, figsize=(6,n*2), sharex=True)
for i in range(n):
    plt.sca(ax[i])
    col2 = a[i]
    sns.countplot(df[col2].values).set_title(a[i])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.xlabel('class')
    plt.ylabel('count')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# In[27]:


########data distrubition class cor choices 
a=['O_Choice_c1','O_Choice_c2', 'O_Choice_c3', 'O_Choice_c4', 'O_Choice_c5','O_Choice_c6', 'O_Choice_c7', 'O_Choice_c8', 'O_Choice_c9']
n=len(a)
fig,ax = plt.subplots(n,1, figsize=(6,n*2), sharex=True)
for i in range(n):
    plt.sca(ax[i])
    col2 = a[i]
    sns.countplot(df1[col2].values).set_title(a[i])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.xlabel('class')
    plt.ylabel('count')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# In[28]:


#####count plot for Neighbors columns
b=['I_Neigh_c1', 'I_Neigh_c2', 'I_Neigh_c3', 'I_Neigh_c4','I_Neigh_c5', 'I_Neigh_c6', 'I_Neigh_c7', 'I_Neigh_c8', 'I_Neigh_c9']
n=len(b)
fig,ax = plt.subplots(n,1, figsize=(6,n*2), sharex=True)
for i in range(n):
    plt.sca(ax[i])
    col = b[i]
    sns.countplot(df[col].values).set_title(b[i])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.xlabel('class')
    plt.ylabel('count')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    


# In[29]:


#####count plot for Neighbors columns
b=['I_Neigh_c1', 'I_Neigh_c2', 'I_Neigh_c3', 'I_Neigh_c4','I_Neigh_c5', 'I_Neigh_c6', 'I_Neigh_c7', 'I_Neigh_c8', 'I_Neigh_c9']
n=len(b)
fig,ax = plt.subplots(n,1, figsize=(6,n*2), sharex=True)
for i in range(n):
    plt.sca(ax[i])
    col = b[i]
    sns.countplot(df1[col].values).set_title(b[i])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.xlabel('class')
    plt.ylabel('count')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# In[30]:


#####count plot for Neighbors columns
c=['I_Hist_c1', 'I_Hist_c2', 'I_Hist_c3', 'I_Hist_c4','I_Hist_c5', 'I_Hist_c6', 'I_Hist_c7', 'I_Hist_c8', 'I_Hist_c9']
n=len(c)
fig,ax = plt.subplots(n,1, figsize=(6,n*2), sharex=True)
for i in range(n):
    plt.sca(ax[i])
    col1 = c[i]
    sns.countplot(df[col1].values).set_title(c[i])
    # Add title and axis names
    plt.xlabel('class')
    plt.ylabel('count')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# In[31]:


#####count plot for Neighbors columns
c=['I_Hist_c1', 'I_Hist_c2', 'I_Hist_c3', 'I_Hist_c4','I_Hist_c5', 'I_Hist_c6', 'I_Hist_c7', 'I_Hist_c8', 'I_Hist_c9']
n=len(c)
fig,ax = plt.subplots(n,1, figsize=(6,n*2), sharex=True)
for i in range(n):
    plt.sca(ax[i])
    col1 = c[i]
    sns.countplot(df1[col1].values).set_title(c[i])
    # Add title and axis names
    plt.xlabel('class')
    plt.ylabel('count')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# In[ ]:


#######################modeling 


# In[ ]:


#######################keras


# In[13]:


# Creates a model given an activation and learning rate
def create_model(learning_rate=0.01, activation='relu'):
  
  	# Create an Adam optimizer with the given learning rate
  	opt = Adam(lr=learning_rate)
  	
  	# Create your binary classification model  
  	model = Sequential()
  	model.add(Dense(128, input_shape=(3,), activation=activation))
  	model.add(Dense(256, activation=activation))
  	model.add(Dense(1, activation='sigmoid'))
  	
  	# Compile your model with your optimizer, loss, and metrics
  	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
  	return model


# In[14]:


##Import KerasClassifier from keras wrappers

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model)
# random_search.fit(X,y) takes too long! But would start the search.
# Define the parameters to try out
params = {'activation':['relu', 'tanh','sigmoid','softmax'], 'batch_size':[32, 128, 256], 'epochs':[50, 100, 200], 'learning_rate':[0.1, 0.01, 0.001]}


# In[15]:


# Create a randomize search cv object and fit it on the data to obtain the results
random_search = RandomizedSearchCV(model, param_distributions=params, cv=3) 
grid_result = random_search.fit(X1s,Y1s)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


######################NEW########################


# In[18]:


# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=3, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# In[19]:


def create_model2(layers,activation):
    model=Sequential()
    for i,nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=X_train.shape[1]))
            model.add(Activation(activation))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
    model.add(Dense(1))   
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model
model = KerasClassifier(build_fn=create_model2, verbose=0)  


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.30, random_state=1000)

layers=[[50],[150,100],[155,55,25]]
activations=['sigmoid','relu']
param_grid = dict(layers=layers,activation=activations)
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=3)
grid_result = grid.fit(X_train,y_train)


# In[24]:


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[15]:


##################Tune Batch Size and Number of Epochs##############################
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 128]
epochs = [10, 50, 100,150,200]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X,Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


##################Tune Batch Size and Number of Epochs##############################
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 128]
epochs = [10, 50, 100,150,200]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X,Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[19]:


#######################Tune the Training Optimization Algorithm################################
# Function to create model, required for KerasClassifier
def create_model(optimizer='adam'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=3, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[21]:


########################Tune Learning Rate and Momentum###########################
# Function to create model, required for KerasClassifier
def create_model(learn_rate=0.01, momentum=0):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=3, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	optimizer = SGD(lr=learn_rate, momentum=momentum)
	model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
	return model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(learn_rate=learn_rate, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[23]:


########################Tune Network Weight Initialization###########################
# Function to create model, required for KerasClassifier
def create_model(init_mode='uniform'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=3, kernel_initializer=init_mode, activation='relu'))
	model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[24]:


######################Tune the Neuron Activation Function###########################
# create model
# Function to create model, required for KerasClassifier
def create_model(activation='relu'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=3, kernel_initializer='uniform', activation=activation))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[13]:


# Function to create model, required for KerasClassifier
def create_model(dropout_rate=0.0, weight_constraint=0):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=3, kernel_initializer='uniform', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
	model.add(Dropout(dropout_rate))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[31]:


###########################################Tune the Number of Neurons in the Hidden Layer
# Function to create model, required for KerasClassifier
def create_model(neurons=1):
	# create model
	model = Sequential()
	model.add(Dense(neurons, input_dim=3, kernel_initializer='uniform', activation='relu', kernel_constraint=maxnorm(4)))
	model.add(Dropout(0.2))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
neurons = [10,25, 50,100,128, 256, 300]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[16]:


def neuralnetwork(X_train,X_test,y_train, y_test):
    model = Sequential()
    model.add(Dense(150, input_dim=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=128,verbose=0)
    # evaluate the model
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # plot loss during training
    pyplot.subplot(211)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['acc'], label='train')
    pyplot.plot(history.history['val_acc'], label='test')
    pyplot.legend()
    pyplot.show()


# In[35]:


for h in range(0,9):
    X_train, X_test, y_train, y_test = train_test_split(listx[h],listy[h], test_size=0.30, random_state=1000)
    print('the result of model base on cell number',h+1)
    neuralnetwork(X_train,X_test,y_train, y_test)


# In[17]:


for h in range(0,9):
    X_train, X_test, y_train, y_test = train_test_split(listx1[h],listy1[h], test_size=0.30, random_state=1000)
    print('the result of model base on cell number',h+1)
    neuralnetwork(X_train,X_test,y_train, y_test)


# In[88]:


def neuralnetwork2(X_train,X_test,y_train, y_test):
    global y_probas
    global cm
    global cr
    global y_pred
    model = Sequential()
    model.add(Dense(150, input_dim=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=128,verbose=0)
    # Predicting the Test set results
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)
    y_probas = model.predict_proba(X_test)
    cr=classification_report(y_test, y_pred)
    # Predicting the Test set results
    print(cm)
    print(cr)


# In[87]:


#########################Confusion matrix for each cell
for h in range(0,9):
    X_train, X_test, y_train, y_test = train_test_split(listx1[h],listy1[h], test_size=0.30, random_state=1000)
    print('the result of model base on cell number',h+1)
    neuralnetwork2(X_train,X_test,y_train, y_test)
    # Predicting the Test set results
    print(cm)
    print(cr)
    


# In[78]:


for h in range(0,9):
    X_train, X_test, y_train, y_test = train_test_split(listx1[h],listy1[h], test_size=0.30, random_state=1000)
    print('the result of model base on cell number',h+1)
    neuralnetwork2(X_train,X_test,y_train, y_test)
   
   


# In[ ]:


#########################cross validition for each cell


# In[53]:


from keras import models
def create_network():
    
    # Start neural network
    network = models.Sequential()
    
    network.add(Dense(150, input_dim=3, activation='relu', kernel_initializer='he_uniform'))
    network.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.01, momentum=0.9)
    network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
    # Return compiled network
    return network


# In[55]:


neural_network = KerasClassifier(build_fn=create_network, 
                                 epochs=10, 
                                 batch_size=100, 
                                 verbose=0)


# In[59]:


for h in range(0,9):
    print('the result of 5-fold cross validition  base on cell number',h+1)
    print('\n')
    cross_value=cross_val_score(neural_network, listx1[h],listy1[h], cv=5)
    print('\n',cross_value)
    print("the mean of cross validiation is :",np.mean(cross_value))
    print('##############################################################')


# In[1]:





# In[7]:


grid_result = random_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


############################second way #################################################


# In[8]:


# Instantiate a Sequential model
model = Sequential()

# Add a hidden layer of 64 neurons and a 20 neuron's input
model.add(Dense(64, input_shape=(19,), activation='relu'))

# Add an output layer of 3 neurons with sigmoid activation
model.add(Dense(9, activation='sigmoid'))

# Compile your model with adam and binary crossentropy loss
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


# In[5]:


################Train/test split for all cells method
X1_train, X1_test, y1_train, y1_test  = train_test_split(F,h, test_size=0.30, random_state=1111)


# In[12]:


# Train for 100 epochs using a validation split of 0.2
model.fit(X1_train, y1_train,batch_size=128,epochs=100)

# Predict on sensors_test and round up the predictions
preds = model.predict(X1_test)
preds_rounded = np.round(preds)

# Print rounded preds
print('Rounded Predictions: \n', preds_rounded)

# Evaluate your model's accuracy on the test data
accuracy = model.evaluate(X1_test,y1_test)

# Print accuracy
print('Accuracy:', accuracy)


# In[62]:


# Creates a model given an activation and learning rate
def create_model1(learning_rate=0.01, activation='relu'):
  
  	# Create an Adam optimizer with the given learning rate
  	opt = Adam(lr=learning_rate)
  	
  	# Create your binary classification model  
  	model = Sequential()
  	model.add(Dense(128, input_shape=(19,), activation=activation))
  	model.add(Dense(256, activation=activation))
  	model.add(Dense(9, activation='sigmoid'))
  	
  	# Compile your model with your optimizer, loss, and metrics
  	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
  	return model


# In[ ]:


##################Confusion Matrix for all cells methods

X2_train, X2_test, y2_train, y2_test  = train_test_split(F,h, test_size=0.30, random_state=1111)
model = Sequential()
model.add(Dense(150, input_dim=19, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(9, activation='sigmoid'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X2_train,y2_train)
# Predicting the Test set results
y_pred = model.predict(X2_test)
y_pred = (y_pred > 0.5)


cr=classification_report(y2_test, y_pred)
# Predicting the Test set results
print(cr)


# In[ ]:





# In[22]:


model = Sequential()
model.add(Dense(50, input_dim=19, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(9, activation='sigmoid'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model
history = model.fit(X1_train, y1_train, validation_data=(X1_test, y1_test), epochs=100, batch_size=128,verbose=0)
# evaluate the model
_, train_acc = model.evaluate(X1_train, y1_train, verbose=0)
_, test_acc = model.evaluate(X1_test, y1_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.show()


# In[ ]:


#####################cv=5 for all cells
def create_network():
    
    # Start neural network
    network = models.Sequential()
    
    network.add(Dense(150, input_dim=19, activation='relu', kernel_initializer='he_uniform'))
    network.add(Dense(9, activation='sigmoid'))
    opt = SGD(lr=0.01, momentum=0.9)
    network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
    # Return compiled network
    return network


# In[ ]:


neural_network = KerasClassifier(build_fn=create_network, 
                                 epochs=10, 
                                 batch_size=100, 
                                 verbose=0)


# In[ ]:


cross_value=cross_val_score(neural_network,F,h, cv=5)
print('\n',cross_value)


# In[ ]:


#####Visualize neural network for all cells method
from nnv import NNV
layers_List = [
    {"title":"input\n(relu)", "units": 19, "color": "darkBlue"},
    {"title":"hidden 1\n(relu)", "units": 150},
    {"title":"hidden 2\n(relu)", "units": 50},
    {"title":"output\n(sigmoid)", "units": 9,"color": "darkBlue"},
]

NNV(layers_List, max_num_nodes_visible=13, node_radius=20, spacing_layer=260, font_size=10).render(save_to_file="my_example_2.pdf")


# In[ ]:


###################################################################################################################


# In[ ]:


#####################Third way-MLP###############


# In[17]:


########Test algoritm for cell one#######
F1=df.iloc[:,[0,1,10]]
h1=df.iloc[:,19]

X2_train, X2_test, y2_train, y2_test  = train_test_split(F1,h1, test_size=0.30, random_state=1111)


# In[19]:


#######bulding algorithm for cell one 
clf = MLPClassifier(hidden_layer_sizes=(100,3), max_iter=500, alpha=0.05,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)


# In[20]:


###############fit model to cell one 
clf.fit(X2_train, y2_train)


# In[21]:


y2_pred = clf.predict(X2_test)
accuracy_score(y2_test, y2_pred)


# In[ ]:


################method one for  MLP foreach cell


# In[34]:


############finding hyper parameter

parameter_space = {
  'hidden_layer_sizes': [(100,1), (100,2), (100,3)],
       'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
'activation': ["logistic", "relu", "tanh"],
    'learning_rate': ["constant", "invscaling", "adaptive"],
}


# In[35]:


mlp = MLPClassifier(max_iter=100)
clf1 = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)


# In[36]:


clf1.fit(X2_train,y2_train)


# In[37]:


# Best paramete set
print('Best parameters found:\n', clf1.best_params_)

# All results
means = clf1.cv_results_['mean_test_score']
stds = clf1.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf1.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


# In[43]:


###### mlp for all cell
dlf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha= 0.05,solver='sgd',activation='tanh', verbose=10,  random_state=21,tol=0.000000001)
dlf.fit(X2_train,y2_train)                   
y2_true, y2_pred = y2_test , dlf.predict(X2_test)



# In[44]:


print('Results on the test set:')
print(classification_report(y2_true, y2_pred))


# In[46]:


cm = confusion_matrix(y2_test, y2_pred)
cm
sns.heatmap(cm, center=True)
plt.show()


# In[45]:


import scikitplot as skplt

y1_probas = dlf.predict_proba(X2_test)
skplt.metrics.plot_roc(y2_test, y1_probas)


# In[58]:


###############define function for making process easier for MLP
def mlpclassi(X_train,X_test,y_train, y_test):
    dlf = MLPClassifier(hidden_layer_sizes=(100, 3), max_iter=500, alpha= 0.05,solver='sgd',activation='tanh', verbose=10,  random_state=21,tol=0.000000001)
    dlf.fit(X_train,y_train)                   
    y2_true, y2_pred = y_test , dlf.predict(X_test)
    print('Results on the test set:')
    print(classification_report(y2_true, y2_pred))
    cm = confusion_matrix(y_test, y2_pred)
    sns.heatmap(cm, center=True)
    y1_probas = dlf.predict_proba(X_test)
    skplt.metrics.plot_roc(y_test, y1_probas)
    plt.show()
    


# In[ ]:


###############fit model
mlp = MLPClassifier(hidden_layer_sizes=(100, 3), max_iter=500, alpha= 0.05,solver='sgd',activation='tanh',)
mlp.fit(X_train,y_train)


# In[59]:


####################fit model and get result for each cell + Confusion matrix and ROC
for h in range(0,9):
    X_train, X_test, y_train, y_test = train_test_split(listx[h],listy[h], test_size=0.30, random_state=1000)
    print('the result of model base on cell number',h+1)
    mlpclassi(X_train,X_test,y_train, y_test)
    


# In[ ]:


##############cv=5 (crosss alidation for mlp for method one(each cells))


# In[ ]:


for h in range(0,9):
    print('the result of model base on cell number',h+1)
    print('####################################################################################################################################################################################\n')
    dlf = MLPClassifier(hidden_layer_sizes=(100, 3), max_iter=500, alpha= 0.05,solver='sgd',activation='tanh', verbose=10,  random_state=21,tol=0.000000001)
    # Compute 10-fold cross-validation scores: cv_scores
    cv_scores = cross_val_score(dlf,listx[h],listy[h],cv=5)
    # Print the 5-fold cross-validation scores
    print(cv_scores)
    print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
    print('######################################################################################################################################################################################')


# In[ ]:


#####################MLP for method 2(all cells)


# In[ ]:


#############train and test 


# In[ ]:


dlf = MLPClassifier(hidden_layer_sizes=(100, 3), max_iter=500, alpha= 0.05,solver='sgd',activation='tanh', verbose=10,  random_state=21,tol=0.000000001)
dlf.fit(X2_train,y2_train)  
y7_pred = dlf.predict(X2_test)
y8_pred = dlf.predict(X2_train)
accuracy_score(y2_test, y7_pred)
print("  test accuracy:",accuracy_score(y2_test, y7_pred))
print(" training accuracy    :",accuracy_score(y2_train,y8_pred))


# In[ ]:





# In[ ]:


#####CV=5 for method 2


# In[ ]:


cv_scores1 = cross_val_score(dlf,X,Y,cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores1)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores1)))


# In[ ]:


####################################################################################


# In[ ]:


############XGBOOSTER 


# In[26]:


# Create the parameter grid: gbm_param_grid
gbm_param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'n_estimators':[50, 100, 150, 200],
    'max_depth': [1,5,10,15,20],
    'learning_rate':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
}

# Instantiate the regressor: gbm
gbm = xgb.XGBClassifier(objective="reg:logistic")

# Perform grid search: grid_mse
grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid,
                        scoring="neg_log_loss", cv=5, verbose=1)
grid_mse.fit(X1,Y1)
print("Best: %f using %s" % (grid_mse.best_score_, grid_mse.best_params_))


# In[62]:


hdmatrix = xgb.DMatrix(X1,Y1)
params = {"objective":"reg:logistic",'colsample_bytree': 0.7, 'max_depth': 5,'learning_rate': 0.1,'n_estimators':500}
cv_results = xgb.cv(dtrain=hdmatrix, params=params, nfold=3, num_boost_round=20, metrics="auc", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)
# Print the AUC
Acurracy_mean=cv_results["test-auc-mean"].iloc[-1].mean()  
print(Acurracy_mean)


# In[78]:


gbm1 = xgb.XGBClassifier(objective="reg:logistic",colsample_bytree=0.7, max_depth=5,learning_rate=0.1,n_estimators=500)
y_pred = cross_val_predict(gbm1,X1,Y1, cv=10)
conf_mat = confusion_matrix(Y1, y_pred)


# In[51]:


n_estimators1 = [50, 100, 150, 200]
max_depth1 = [5,10,15,20]

print(max_depth1)
param_grid1 = dict(max_depth1=max_depth, n_estimators=n_estimators)
scores1= np.array(means).reshape(len(max_depth1), len(n_estimators1))
for i, value in enumerate(max_depth1):
    pyplot.plot(n_estimators, scores1[i], label='depth: ' + str(value))
pyplot.legend()
pyplot.xlabel('n_estimators')
pyplot.ylabel('Log Loss')
pyplot.savefig('n_estimators_vs_max_depth.png')


# In[85]:


####define Xboosting function which we can run for each cell k-fold method
def booster(E,G):
    global Accuracy_mean_test
    global Accuracy_mean_train
    global Accuracy_std_test
    global Accuracy_std_train
    gbm = xgb.XGBClassifier(objective="reg:logistic")
    hdmatrix = xgb.DMatrix(E,G)
    params = {"objective":"reg:logistic",'colsample_bytree': 0.7, 'max_depth': 5,'learning_rate': 0.1,'n_estimators':500}
    cv_results = xgb.cv(dtrain=hdmatrix, params=params, nfold=3, num_boost_round=20, metrics="auc", as_pandas=True, seed=123)
    # Print cv_results
    print(cv_results)
     # Print the AUC
    print('#########################the Acurracy of model#####################################################')
    Accuracy_mean_test=cv_results["test-auc-mean"].iloc[-1].mean()
    print('the mean accurarcy of test is :',Accuracy_mean_test)
    Accuracy_mean_train=cv_results["train-auc-mean"].iloc[-1].mean()
    print('the mean accurarcy of train is :',Accuracy_mean_train)
    Accuracy_std_train=cv_results["train-auc-std"].iloc[-1].mean()
    print('the mean accurarcy of std train is :',Accuracy_std_train)
    Accuracy_std_test=cv_results["test-auc-std"].iloc[-1].mean()
    print('the mean accurarcy of std test is :',Accuracy_std_test)
    return Accuracy_mean_test,Accuracy_mean_train,Accuracy_std_test,Accuracy_std_train


# In[ ]:


#####Avrage of accuracy for all nine models 
Accuracy_test=[]
Accuracy_train=[]
Accuracy_sdtest=[]
Accuracy_sdtrain=[]
#####the results for each cell 
for h in range(0,9):
    print('the result of model base on cell number',h+1)
    booster(listx[h],listy[h])
    print('##################################################################################################')
    Accuracy_test.append(Accuracy_mean_test)
    Accuracy_train.append(Accuracy_mean_train)
    Accuracy_sdtest.append(Accuracy_std_test)
    Accuracy_sdtrain.append(Accuracy_std_train)


# In[ ]:


print(np.mean(Accuracy_test),np.mean(Accuracy_train))


# In[ ]:


def splitbooster(J,Z) : 
    # Create the training and test sets
    X_train,X_test,y_train,y_test= train_test_split(J,Z, test_size=0.3, random_state=123)
    # Instantiate the XGBClassifier: xg_cl
    gbm1 = xgb.XGBClassifier(objective="reg:logistic",colsample_bytree=0.7, max_depth=5,learning_rate=0.1,n_estimators=500)
    # Fit the classifier to the training set
    gbm1.fit(X_train,y_train)
    # Predict the labels of the test set: preds
    preds = gbm1.predict(X_test)
    preds1 = gbm1.predict(X_train)
    # Compute the accuracy: accuracy
    accuracy_test = float(np.sum(preds==y_test))/y_test.shape[0]
    accuracy_train = float(np.sum(preds1==y_train))/y_train.shape[0]
    print("accuracy of test dataset: %f" % (accuracy_test))
    print("accuracy of train dataset: %f" % (accuracy_train))


# In[ ]:


for h in range(0,9):
    print('##################################################################################################')
    print('the result of model base on cell number',h+1)
    print('##################################################################################################')
    splitbooster(listx[h],listy[h])


# In[ ]:


##################confusion matrix for method one(for each cell)


# In[ ]:


for h in range(0,9):
    print('##################################################################################################')
    print('the result of model base on cell number',h+1)
    print('##################################################################################################')
    X_train, X_test, y_train, y_test = train_test_split(listx[h],listy[h], test_size=0.30)
    gmb= xgb.XGBClassifier(objective="reg:logistic",colsample_bytree=0.7, max_depth=5,learning_rate=0.1,n_estimators=500)
    gmb.fit(X_train, y_train)
    y2_true, y2_pred = y_test , dlf.predict(X_test)
    print('Results on the test set:')
    print(classification_report(y2_true, y2_pred))
    cm = confusion_matrix(y_test, y2_pred)
    sns.heatmap(cm, center=True)
    y1_probas = dlf.predict_proba(X_test)
    skplt.metrics.plot_roc(y_test, y1_probas)
    plt.show()
    print('##################################################################################################')


# In[ ]:


######################Method2 k-fold method 
F1=np.array(F)
h1=np.array(h)
booster(F1,h1)


# In[ ]:


################confusion amtrix for method 2


# In[ ]:


from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
X_train, X_test, y_train, y_test = train_test_split(F,h, test_size=0.30)
multioutput = MultiOutputClassifier(xgb.XGBClassifier(objective='reg:logistic')).fit(X_train, y_train)
y2_true, y2_pred = y_test , multioutput.predict(X_test)
print('Results on the test set:')
print(classification_report(y2_true, y2_pred))
cm = multilabel_confusion_matrix(y_test, y2_pred)
sns.heatmap(cm, center=True)
y1_probas =multioutput.predict_proba(X_test)
skplt.metrics.plot_roc(y_test, y1_probas)
plt.show()


# In[ ]:


#############################################################################################


# In[ ]:


##################################Random forest


# In[ ]:


##########hyper parameter 


# In[ ]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)


# In[ ]:


# Fit the grid search to the data
grid_search.fit(X,Y)
grid_search.best_params_
print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[ ]:


# Fit the grid search to the data
grid_search.fit(X1,Y1)
grid_search.best_params_


# In[ ]:


best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, test_features, test_labels)
print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))


# In[ ]:


rfc1=RandomForestClassifier(bootstrap=True, class_weight=None, 

            max_depth=100, max_features=2, 

            min_impurity_decrease=0.0, min_impurity_split=None, 

            min_samples_leaf=4, min_samples_split=8, 

            n_estimators=100, n_jobs=1,
            random_state=10, verbose=1) 

rfc1.fit(X2_train, y2_train) 

 rfc1_pred = rfc1.predict(X2_test) 

print(confusion_matrix(y2_test,rfc1_pred)) 

print(classification_report(y2_test,rfc1_pred))
print(rfc1.score(X2_test,y2_test))


# In[ ]:


##################################


# In[ ]:





# In[ ]:


#####define randomeforest function 


# In[ ]:


def randomforest(X1_train, X1_test, y1_train, y1_test):
    forest = RandomForestClassifier(n_estimators=10, random_state=1)
    forest.fit(X1_train,y1_train)
    y2_pred=forest.predict(X1_test)
    y3_pred =forest.predict(X1_train)
    print(" training accuracy    :",accuracy_score(y1_train, y3_pred))
    print("  test accuracy:",accuracy_score(y1_test,y2_pred))
    


# In[ ]:


############# the result of cross validation and test,train/split method for each cell 


# In[ ]:


for h in range(0,9):
    print('the result of model base on cell number',h+1)
    X_train, X_test, y_train, y_test = train_test_split(listx[h],listy[h], test_size=0.30)
    randomforest(X_train, X_test, y_train, y_test )
    print('########################################################################\n')
    ## Compute 5-fold cross-validation scores: cv_scores
    cv_scores = cross_val_score(forest,listx[h],listy[h],cv=5)
    # Print the 5-fold cross-validation scores
    print(cv_scores)
    print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
    print('####################################################################################################################################################################################\n')


# In[ ]:


########confusion matrix for method one (each cell)


# In[ ]:


for h in range(0,9):
    print('the result of model base on cell number',h+1)
    X_train, X_test, y_train, y_test = train_test_split(listx[h],listy[h], test_size=0.30)
    forest = RandomForestClassifier(n_estimators=10, random_state=1)
    forest.fit(X_train,y_train)
    y2_true, y2_pred = y_test , forest.predict(X_test)
    print('Results on the test set:')
    print(classification_report(y2_true, y2_pred))
    cm = confusion_matrix(y_test, y2_pred)
    sns.heatmap(cm, center=True)
    y1_probas = forest.predict_proba(X_test)
    skplt.metrics.plot_roc(y_test, y1_probas)
    plt.show()


# In[ ]:


################method 2


# In[ ]:


##############cross validation/testt&train for method 2


# In[ ]:


F1=np.array(F)
h1=np.array(h)
X1_train, X1_test, y1_train, y1_test  = train_test_split(F1,h1, test_size=0.30, random_state=1111)
forest = RandomForestClassifier(n_estimators=10, random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=9)
multi_target_forest.fit(X1_train,y1_train)
yp=multi_target_forest.predict(X1_test)
accuracy_score(y1_test, yp)
y8 = multi_target_forest.predict(X1_train)
print("  test accuracy:",accuracy_score(y1_test, yp))
print(" training accuracy    :",accuracy_score(y1_train,y8))
# Compute 10-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(multi_target_forest,F1,h1,cv=10)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 10-Fold CV Score: {}".format(np.mean(cv_scores)))


# In[ ]:


######################confusion matrix for method 2 


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(F,h, test_size=0.30)
forest = RandomForestClassifier(n_estimators=10, random_state=1)
forest.fit(X_train,y_train)
y2_true, y2_pred = y_test , forest.predict(X_test)
print('Results on the test set:')
print(classification_report(y2_true, y2_pred))
cm = confusion_matrix(y_test, y2_pred)
y1_probas = forest.predict_proba(X_test)


# In[ ]:


#############visualize randomfroest for cell one


# In[ ]:


# Model (can also use single decision tree)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10)

# Train
model.fit(X,Y)
# Extract single tree
estimator = model.estimators_[2]

from sklearn.tree import export_graphviz
# Export as dot file
str_tree=export_graphviz(estimator, out_file='tree.dot', 
                feature_names = X.columns,
                class_names = Y,
                rounded = True, proportion = False, 
                precision = 2, filled = True)
get_ipython().system('dot -Tpng tree.dot -o tree.png -Gdpi=15')


# In[ ]:


#############functon to  visuluze each/all cells to
def rfvisu(A,B):
     model = RandomForestClassifier(n_estimators=10)
     # Train
      model.fit(A,B)
     # Extract single tree
     estimator = model.estimators_[2]
    # Export as dot file
     str_tree=export_graphviz(estimator, out_file='tree.dot', 
                feature_names = X.columns,
                class_names = Y,
                rounded = True, proportion = False, 
                precision = 2, filled = True)
   get_ipython().system('dot -Tpng tree.dot -o tree.png -Gdpi=15')


# In[ ]:




