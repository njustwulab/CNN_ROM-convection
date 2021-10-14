# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 20:53:47 2019

@author: Peng-JZ
"""
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from CNN.model import Model, get_US, get_data

valmodel = 'car'

trained_model="./model"

font = {'family' : 'Times New Roman',
     'weight' : 'bold',
     'size' : 22}

def evaluate(modelname):
    with tf.Graph().as_default():
        predata, Nandata, true = get_data(modelname)
        U, S = get_US(modelname)
        images = predata.reshape(1,250,250,1)
        images = images.astype(np.float32)
        model = Model()
        y_pre = model.inference(images,keep_prob=1.0)
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess,trained_model+"/model.ckpt")
            predict_result = sess.run(y_pre)
            reshape_pre = (predict_result*Nandata).reshape(250,250)
            label = true*S+U
            pre = reshape_pre*S+U
            E = np.abs(label-pre)/np.abs(label)
            return reshape_pre, true, E


def main(argv=None):
    results, trues, error = evaluate(valmodel)
    cm = plt.cm.jet
    fig = plt.figure(figsize=(20,6))
    plt.subplot(1,3,1)
    plt.contourf(results, cmap = cm, levels = 100, extent=(0,1,0,1))
    plt.tick_params(labelsize=16)
    plt.title('Network prediction',fontdict=font)
    
    plt.subplot(1,3,2)
    plt.contourf(trues, cmap = cm, levels = 100, extent=(0,1,0,1))
    plt.tick_params(labelsize=16)
    plt.title('Numerical simulation',fontdict=font)
    
    plt.subplot(1,3,3)
    cm2 = plt.cm.coolwarm
    G3 = plt.contourf(error, cmap = cm2, levels = 100, extent=(0,1,0,1))
    plt.tick_params(labelsize=16)
    plt.title('Relative error',fontdict=font)    
    fig.subplots_adjust(right=0.9)
    l = 0.92
    b = 0.12
    w = 0.015
    h = 1 - 2*b
    rect = [l,b,w,h]
    cbar_ax = fig.add_axes(rect) 
    cb = plt.colorbar(G3, cax=cbar_ax)
    cb.ax.tick_params(labelsize=16)
    
if __name__ == '__main__':
    tf.app.run()