# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 08:57:10 2022

this script is to build a model for customer segmentation

credit to :
    HackerEarth HackLive: Customer Segmentation | Kaggle

@author: Afiq Sabqi
"""


import os
import pickle
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import scipy.stats as ss
import missingno as msno
import matplotlib.pyplot as plt

from sklearn import preprocessing
from tensorflow.keras import Sequential
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras.layers import Dropout,Dense,Input,BatchNormalization

#%%                               STATIC

DATA_PATH=os.path.join(os.getcwd(),'dataset','train.csv')

LE_FILE_PATH=os.path.join(os.getcwd(),'model','LE.pkl')
OHE_PICKLE_PATH=os.path.join(os.getcwd(),'model','ohe.pkl')

log_dir=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_PATH=os.path.join(os.getcwd(),'logs',log_dir)

def plot_con(df,continuous_data):
    for con in continuous_data:
        plt.figure()
        sns.distplot(df[con])
        plt.show()

def plot_cat(df,categorical_data):
    for cat in categorical_data:
        plt.figure()
        sns.countplot(df[cat])
        plt.show()


def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


#%%                           DATA LOADING

df=pd.read_csv(DATA_PATH)

#%%                          DATA INSPECTION

df.info()

df.describe().T
df.duplicated().sum()
# no duplicate value

df.isna().sum()
msno.matrix(df)
# from isna() and msno we can see that column prev_camp_contact
# has a lot off nan value make it no use features. can drop entire column
# ID also not important hence can drop entire column

categorical_data = ['job_type','marital','education','default','housing_loan',
                    'personal_loan','communication_type','month',
                    'prev_campaign_outcome','term_deposit_subscribed']
continuous_data = ['customer_age','balance','day_of_month',
                   'last_contact_duration','num_contacts_in_campaign',
                   'num_contacts_prev_campaign']

#%%                           DATA CLEANING

# drop id and days_since_prev_camp_ctc
df.drop('id', axis=1, inplace=True)
df.drop('days_since_prev_campaign_contact', axis=1, inplace=True)

# filling with most common class
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

df.isna().sum()
msno.matrix(df)
# from msno no more NaNs value

#%%                         FEATURES SELECTION

##         Label Encoding
le=preprocessing.LabelEncoder()

for cat in categorical_data:
    df[cat]=le.fit_transform(df[cat])

with open(LE_FILE_PATH,'wb') as file:
      pickle.dump(le,file)

##         general view of correlation from graph

# from def plot_con (static section)
plot_con(df,continuous_data)  
plot_cat(df,categorical_data) 


##          correlation analysis

for con in continuous_data:
    print(con)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[con],axis=-1),df['term_deposit_subscribed'])
    print(lr.score(np.expand_dims(df[con],
                                  axis=-1),df['term_deposit_subscribed']))

# from def cramers_corrected_stat (static section)
for cat in categorical_data:
    print(cat)
    confusion_mat = pd.crosstab(df[cat],
                                df['term_deposit_subscribed']).to_numpy()
    print(cramers_corrected_stat(confusion_mat))

# from correlation analysis above, anything below 10% is deselect
X=df.loc[:,['customer_age','balance','day_of_month','last_contact_duration',
              'num_contacts_in_campaign','num_contacts_prev_campaign',
              'job_type','housing_loan','communication_type','month',
              'prev_campaign_outcome']]
y=df['term_deposit_subscribed']
# X=df.drop(labels=['term_deposit_subscribed'],axis=1)


#%%                                PREPROCESSING


ohe=OneHotEncoder(sparse=False)
y=ohe.fit_transform(np.expand_dims(y,axis=1))

with open(OHE_PICKLE_PATH,'wb') as file:
    pickle.dump(ohe,file)

X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                  test_size=0.3,
                                                  random_state=123)

nb_features=np.shape(X)[1:]
nb_classes=len(np.unique(y_train,axis=0))


model=Sequential()                # to create container
model.add(Input(shape=(nb_features)))
model.add(Dense(128,activation='linear',name='Hidden_Layer1'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128,activation='linear',name='Hidden_Layer2'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(nb_classes,activation='softmax',name='Output_layer'))

model.summary()

plot_model(model)

# after creating model, compile the model(wrapping)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

tensorboard_callback=TensorBoard(log_dir=LOG_PATH)

# early_stopping_callback=EarlyStopping(monitor='loss',patience=10)

hist=model.fit(X_train,y_train,
              validation_data=(X_test,y_test),
              batch_size=123,epochs=100,
              callbacks=[tensorboard_callback])
              
              
hist.history.keys()
training_loss=hist.history['loss']
training_acc=hist.history['acc']
validation_acc=hist.history['val_acc']
validation_loss=hist.history['val_loss']

plt.figure()
plt.plot(training_loss)
plt.plot(validation_loss)
plt.legend(['train_loss','val_loss'])
plt.show()

plt.figure()
plt.plot(training_acc)
plt.plot(validation_acc)
plt.legend(['train_acc','val_acc'])
plt.show()

results=model.evaluate(X_test,y_test)
print(results)


y_true=y_test
y_pred=model.predict(X_test)

y_true=np.argmax(y_true,axis=1)
y_pred=np.argmax(y_pred,axis=1)

print(classification_report(y_true,y_pred))
print(accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

#%%                          DISCUSSION

'''

    *f1 score shows a 90% accuracy.
    
    *with 100 epoch train, the model learn well and gives a val_acc
    near the train acc.
    
    *change in number of hidden layer/output/ and number of epochs
    did not make the model train more accurate.
    
    *a number of features is drop since it give very low correlation 
    to the term_deposit_subscribed which is below 10%.
    
    *also id and days_since_prev_campaign_contact columns is drop
    because of not important(id) and days_since_prev_campaign_contact
    have many NaNs value that make it not acceptable as data
    
    *selection of features gives a difference about 1% accuracy
    
    *the graph also shows in tensorboard as shown in tensorboard.png
    

'''













































