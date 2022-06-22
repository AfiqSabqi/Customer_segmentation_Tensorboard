# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:23:09 2022

this project is to create modules for customer segmentation project
so that it will look neat

@author: afiq Sabqi
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout,Dense,Input,BatchNormalization

class EDA():
    def __init__(self):
        pass

    def plot_con(self,df,continuous_data):
        for con in continuous_data:
            plt.figure()
            sns.distplot(df[con])
            plt.show()
    
    def plot_cat(self,df,categorical_data):
        for cat in categorical_data:
            plt.figure()
            sns.countplot(df[cat])
            plt.show()

class ModelCreation():
    def __init__(self):
        pass
    
    def simple_tens_layer(self,X,y_train,num_nodes=128,drop_rate=0.3):
        model=Sequential()                # to create container
        model.add(Input(shape=(np.shape(X)[1:])))
        model.add(Dense(num_nodes,activation='linear',name='Hidden_Layer1'))
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
        model.add(Dense(num_nodes,activation='linear',name='Hidden_Layer2'))
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
        model.add(Dense(len(np.unique(y_train,axis=0))
                        ,activation='softmax',name='Output_layer'))
        model.summary()
        return model


