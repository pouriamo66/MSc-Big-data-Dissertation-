#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale,normalize,minmax_scale
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_circles
from keras.optimizers import SGD
from keras.optimizers import Adam
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


# In[2]:


df=pd.read_csv("farm_or_fallow.csv")
X1=df.iloc[:,[0,1,10]]
Y1=df.iloc[:,19]
X1['class']=Y1
X=df.iloc[:,[0,1,10]]
y=X1['class']
y=y.astype('category')
F=df.iloc[:,0:19]
h=df.iloc[:,19:]


# In[49]:


h


# In[55]:


listy=[]

for i in range(0,9):
    y=df.iloc[:,19+i]
    listy.append(y)
    


# In[56]:


listx=[]
for j in range(0,9):
    x=df.iloc[:,[0,1+j,10+j]]
    listx.append(x)


# In[4]:


# Use pairplot and set the hue to be our class
sns.pairplot(X1, hue='class') 

# Show the plot
plt.show()

# Describe the data
print('Dataset stats: \n', X1.describe())

# Count the number of observations of each class
print('Observations per class: \n',X1['class'].value_counts())


# In[25]:


# Use pairplot and set the hue to be our class
sns.pairplot(df, hue='O_Choice_c1') 

# Show the plot
plt.show()

# Describe the data
print('Dataset stats: \n', df.describe())

# Count the number of observations of each class
print('Observations per class: \n', df['O_Choice_c1'].value_counts())


# In[12]:


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


# In[15]:


##Import KerasClassifier from keras wrappers

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model)
# random_search.fit(X,y) takes too long! But would start the search.
# Define the parameters to try out
params = {'activation':['relu', 'tanh','sigmoid','softmax'], 'batch_size':[32, 128, 256], 'epochs':[50, 100, 200], 'learning_rate':[0.1, 0.01, 0.001]}


# In[16]:


# Create a randomize search cv object and fit it on the data to obtain the results
random_search = RandomizedSearchCV(model, param_distributions=params, cv=5) 
grid_result = random_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# In[26]:


def neuralnetwork(X_train,X_test,y_train, y_test):
    model = Sequential()
    model.add(Dense(50, input_dim=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # fit model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32,verbose=0)
    # evaluate the model
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
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


# In[27]:


for h in range(0,9):
    X_train, X_test, y_train, y_test = train_test_split(listx[h],listy[h], test_size=0.30, random_state=1000)
    print('the result of model base on cell number',h+1)
    neuralnetwork(X_train,X_test,y_train, y_test)


# In[18]:


def neuralnetwork(X_train,X_test,y_train, y_test):
    model = Sequential()
    model.add(Dense(50, input_dim=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # fit model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, verbose=0)
    # evaluate the model
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
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





# In[ ]:


model = Sequential()
model.add(Dense(50, input_dim=3, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, verbose=0)
# evaluate the model
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
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


# In[25]:


# Instantiate a Sequential model
model = Sequential()

# Add a hidden layer of 64 neurons and a 20 neuron's input
model.add(Dense(2, input_shape=(3,), activation='sigmoid'))

# Compile your model
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Display a summary of your model
model.summary()


# In[ ]:


##There are 180 parameters, 171 from the connection of our input layer to our hidden layer 
###and 9 from the bias weight of each neuron in the hidden layer.


# In[26]:


# Train your model for 20 epochs
model.fit(X_train, y_train,epochs=100)

# Evaluate your model accuracy on the test set
accuracy = model.evaluate(X_test,y_test)[1]

# Print accuracy
print('Accuracy:',accuracy)


# In[118]:


X_test


# In[21]:


# Predict on sensors_test and round up the predictions
# Predict on X_small_test
preds = model.predict(X_test)
preds


# In[22]:


def plot_accuracy(acc,val_acc):
  # Plot training & validation accuracy values
  plt.figure()
  plt.plot(acc)
  plt.plot(val_acc)
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()


# In[23]:


def plot_loss(loss,val_loss):
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()


# In[24]:


# Train your model and save it's history
history = model.fit(X_train,y_train, epochs=50,
              validation_data=(X_test,y_test))

# Plot train vs test loss during training
plot_loss(history.history['loss'], history.history['val_loss'])

# Plot train vs test accuracy during training
plot_accuracy(history.history['acc'],history.history['val_acc'])


# In[128]:


# Import the early stopping callback
from keras.callbacks import EarlyStopping

# Define a callback to monitor val_acc
monitor_val_acc = EarlyStopping(monitor='val_acc', 
                                patience=5)

# Train your model using the early stopping callback
model.fit(X_train, y_train, 
          epochs=1000, validation_data=(X_test, y_test),
          callbacks=[monitor_val_acc])


# In[129]:


# Import the EarlyStopping and ModelCheckpoint callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Early stop on validation accuracy
monitor_val_acc = EarlyStopping(monitor='val_acc', patience=3)

# Save the best model as best_banknote_model.hdf5
modelCheckpoint = ModelCheckpoint('best_banknote_model.hdf5', save_best_only=True)

# Fit your model for a stupid amount of epochs
history = model.fit(X_train, y_train,
                    epochs=10000000,
                    callbacks=[monitor_val_acc,modelCheckpoint],
                    validation_data=(X_test,y_test))


# In[ ]:


########################################


# In[7]:


model = Sequential()
model.add(Dense(50, input_dim=3, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, verbose=0)
# evaluate the model
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
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


# In[6]:


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


# In[7]:


##Import KerasClassifier from keras wrappers
from keras.wrappers.scikit_learn import KerasClassifier

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model)
# random_search.fit(X,y) takes too long! But would start the search.
#show_results()
# Define the parameters to try out
params = {'activation':['relu', 'tanh'], 'batch_size':[32, 128, 256], 
         'epochs':[50, 100, 200], 'learning_rate':[0.1, 0.01, 0.001]}

# Create a randomize search cv object and fit it on the data to obtain the results
random_search = RandomizedSearchCV(model, param_distributions=params, cv=5)


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


##Import KerasClassifier from keras wrappers
from keras.wrappers.scikit_learn import KerasClassifier

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model1)
# random_search.fit(F,h) takes too long! But would start the search.
#show_results()
# Define the parameters to try out
params = {'activation':['relu', 'tanh','sigmoid','softmax'], 'batch_size':[32, 128, 256], 
         'epochs':[50, 100,200], 'learning_rate':[0.1, 0.01, 0.001]}

# Create a randomize search cv object and fit it on the data to obtain the results
random_search = RandomizedSearchCV(model, param_distributions=params, cv=5)
grid_result = random_search.fit(F, h)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
   print("%f (%f) with: %r" % (mean, stdev, param))


# In[17]:


# Train for 100 epochs using a validation split of 0.2
model.fit(X1_train, y1_train,batch_size=128,validation_split=0.2,epochs=100)

# Predict on sensors_test and round up the predictions
preds = model.predict(X1_test)
preds_rounded = np.round(preds)

# Print rounded preds
print('Rounded Predictions: \n', preds_rounded)

# Evaluate your model's accuracy on the test data
accuracy1 = model.evaluate(X1_test,y1_test)
# Print accuracy
print('Accuracy:', accuracy1)


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


# In[17]:


#####################Third way-MLP###############
F1=df.iloc[:,[0,1,10]]
h1=df.iloc[:,19]

X2_train, X2_test, y2_train, y2_test  = train_test_split(F1,h1, test_size=0.30, random_state=1111)


# In[19]:


clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)


# In[20]:



clf.fit(X2_train, y2_train)


# In[21]:


y2_pred = clf.predict(X2_test)
accuracy_score(y2_test, y2_pred)


# In[34]:


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


dlf = MLPClassifier(hidden_layer_sizes=(100, 3), max_iter=500, alpha= 0.05,solver='sgd',activation='tanh', verbose=10,  random_state=21,tol=0.000000001)
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
    


# In[59]:


for h in range(0,9):
    X_train, X_test, y_train, y_test = train_test_split(listx[h],listy[h], test_size=0.30, random_state=1000)
    print('the result of model base on cell number',h+1)
    mlpclassi(X_train,X_test,y_train, y_test)
    


# In[70]:


Y2=df.iloc[1,19:]
print(Y2)

