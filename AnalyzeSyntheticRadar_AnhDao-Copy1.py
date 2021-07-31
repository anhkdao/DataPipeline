#!/usr/bin/env python
# coding: utf-8

# # Notebook for analyzing synthetic radar results
# 
# 
# Before running this notebook, download the pretrained models as described in the `README`
# 

# In[5]:


import os
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
import sys
sys.path.append('../src/')
import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


import streamlit as st
st.title('Analyze Synthetic Radar Data')


# In[2]:





# In[2]:




# In[5]:




# In[2]:




# ## Plot metrics

# In[6]:


# Read the metrics output during training
log_files= {
    'mse':'/Users/anhdao/neurips-2020-sevir/logs/sample-mse/metrics.csv',
    'mse+vgg':'/Users/anhdao/neurips-2020-sevir/logs/sample-mse-vgg/metrics.csv',
    'gan':'/Users/anhdao/neurips-2020-sevir/logs/sample-gan-mae/metrics.csv'
}
metrics={ k:pd.read_csv(v) for k,v in log_files.items()}


# In[7]:


# MSE
fig,ax=plt.subplots(2,3,figsize=(15,6))
def plot_metrics_row(df,metric_names,ax):
    for k,m in enumerate(metric_names):
        df[[m,'val_'+m]].plot(ax=ax[k])
        ax[k].grid(True,alpha=.25)

plot_metrics_row(metrics['mse'],['pod74','sucr74','csi74'],ax[0])
plot_metrics_row(metrics['mse'],['pod133','sucr133','csi133'],ax[1])


# With MSE loss, the model converges in relatively few epochs when using the full SEVIR training dataset.  In this case, each epoch represents 75 batches of 32 SEVIR samples.  In this case, roughly 20 epochs seems like enough for peak performance on the validation set (orange line) in a variety of metrics.  The training loss continues to improve over time, but this is likely overfitting based on the stagnant validation curve. 

# In[8]:


# mse+mse
fig,ax=plt.subplots(2,3,figsize=(15,6))
def plot_metrics_row(df,metric_names,ax):
    for k,m in enumerate(metric_names):
        df[[m,'val_'+m]].plot(ax=ax[k])
        ax[k].grid(True,alpha=.25)

plot_metrics_row(metrics['mse+vgg'],['output_layer_1_pod74','output_layer_1_sucr74','output_layer_1_csi74'],ax[0])
plot_metrics_row(metrics['mse+vgg'],['output_layer_1_pod133','output_layer_1_sucr133','output_layer_1_csi133'],ax[1])


# This model also converged in ~20 epochs.

# In[9]:


# gan
fig,ax=plt.subplots(1,4,figsize=(15,4))
def plot_metrics_row(df,metrics,ax):
    for k,m in enumerate(metrics):
        df[[m]].plot(ax=ax[k])
        ax[k].grid(True,alpha=.25)

plot_metrics_row(metrics['gan'],['gen_total_loss', 'gen_gan_loss', 'gen_l1_loss', 'disc_loss'],ax)


# The four losses involved with training the cGAN version of the loss.  No validation set was used for this experiement.  At around spoch 50, the discriminator started competiting with the generator.   

# ## Generate sample images

# This runs the trained model on samples from the test set.  We will use basic color maps for this demo (if the paper is accepted we will apply the same colormap and style shown in the paper)

# In[10]:


# Load weights from best model on val set
mse_weights_file = '/Users/anhdao/Downloads/mse_weights.h5'
mse_model = tf.keras.models.load_model(mse_weights_file,compile=False,custom_objects={"tf": tf})

mse_vgg_weights_file = '/Users/anhdao/Downloads/mse_vgg_weights.h5'
mse_vgg_model = tf.keras.models.load_model(mse_vgg_weights_file,compile=False,custom_objects={"tf": tf})

gan_weights_file = '/Users/anhdao/Downloads/gan_mae_weights.h5'
gan_model = tf.keras.models.load_model(gan_weights_file,compile=False,custom_objects={"tf": tf})


# ## Load sample test data
# 
# To download sample test data, go to https://www.dropbox.com/s/7o3jyeenhrgrkql/synrad_testing.h5?dl=0
#  and save file to `data/sample/synrad_testing.h5`

# In[1]:


"""
data reader for synrad using SEVIR
"""

import logging
import os
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
import h5py
import numpy as np


def get_data(train_data, pct_validation=0.2, rank=0, size=1, end=None ):
    logging.info('Reading datasets')
    train_IN, train_OUT = read_data(train_data, rank=rank, size=size, end=end)

    # Make the validation dataset the last pct_validation of the training data
    val_idx = int((1-pct_validation)*len(train_OUT['vil']))
    val_IN={}
    val_OUT={}
    for k in train_IN:
        train_IN[k],val_IN[k]=train_IN[k][:val_idx],train_IN[k][val_idx:]
    for k in train_OUT:
        train_OUT[k],val_OUT[k]=train_OUT[k][:val_idx],train_OUT[k][val_idx:]
    
    logging.info('data loading completed')
    return (train_IN,train_OUT,val_IN,val_OUT)


def read_data(filename, rank=0, size=1, end=None,dtype=np.float32):
    x_keys = ['ir069','ir107','lght']
    y_keys = ['vil']
    s = np.s_[rank:end:size]
    with h5py.File(filename, 'r') as hf:
        IN  = {k:hf[k][s].astype(np.float32) for k in x_keys}
        OUT = {k:hf[k][s].astype(np.float32) for k in y_keys}
    return IN,OUT


# In[3]:


# Load a part of the test dataset

x_test,y_test = read_data('/Users/anhdao/Downloads/synrad_testing.h5',end=1000)


# ## Visualize results on some test samples

# In[11]:


# Run model on test set
def run_synrad(model,x_test,batch_size=32):
    return model.predict([x_test[k] for k in ['ir069','ir107','lght']],batch_size=batch_size)
y_pred_mse     = run_synrad(mse_model,x_test)
y_pred_mse_vgg = run_synrad(mse_vgg_model,x_test)
y_pred_gan     = run_synrad(gan_model,x_test)


# In[16]:


import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def get_cmap(type,encoded=True):
    if type.lower()=='vis':
        cmap,norm = vis_cmap(encoded)
        vmin,vmax=(0,10000) if encoded else (0,1)
    elif type.lower()=='vil':
        cmap,norm=vil_cmap(encoded)
        vmin,vmax=None,None
    elif type.lower()=='ir069':
        cmap,norm=c09_cmap(encoded)
        vmin,vmax=(-8000,-1000) if encoded else (-80,-10)
    elif type.lower()=='lght':
        cmap,norm='hot',None
        vmin,vmax=0,5
    else:
        cmap,norm='jet',None
        vmin,vmax=(-7000,2000) if encoded else (-70,20)

    return cmap,norm,vmin,vmax


def vil_cmap(encoded=True):
    cols=[   [0,0,0],
             [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
             [0.1568627450980392,  0.7450980392156863,  0.1568627450980392],
             [0.09803921568627451, 0.5882352941176471,  0.09803921568627451],
             [0.0392156862745098,  0.4117647058823529,  0.0392156862745098],
             [0.0392156862745098,  0.29411764705882354, 0.0392156862745098],
             [0.9607843137254902,  0.9607843137254902,  0.0],
             [0.9294117647058824,  0.6745098039215687,  0.0],
             [0.9411764705882353,  0.43137254901960786, 0.0],
             [0.6274509803921569,  0.0, 0.0],
             [0.9058823529411765,  0.0, 1.0]]
    lev = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0]
    #TODO:  encoded=False
    nil = cols.pop(0)
    under = cols[0]
    over = cols.pop()
    cmap=mpl.colors.ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = mpl.colors.BoundaryNorm(lev, 10)
    return cmap,norm
       
    
def vis_cmap(encoded=True):
    cols=[[0,0,0],
             [0.0392156862745098, 0.0392156862745098, 0.0392156862745098],
             [0.0784313725490196, 0.0784313725490196, 0.0784313725490196],
             [0.11764705882352941, 0.11764705882352941, 0.11764705882352941],
             [0.1568627450980392, 0.1568627450980392, 0.1568627450980392],
             [0.19607843137254902, 0.19607843137254902, 0.19607843137254902],
             [0.23529411764705882, 0.23529411764705882, 0.23529411764705882],
             [0.27450980392156865, 0.27450980392156865, 0.27450980392156865],
             [0.3137254901960784, 0.3137254901960784, 0.3137254901960784],
             [0.35294117647058826, 0.35294117647058826, 0.35294117647058826],
             [0.39215686274509803, 0.39215686274509803, 0.39215686274509803],
             [0.43137254901960786, 0.43137254901960786, 0.43137254901960786],
             [0.47058823529411764, 0.47058823529411764, 0.47058823529411764],
             [0.5098039215686274, 0.5098039215686274, 0.5098039215686274],
             [0.5490196078431373, 0.5490196078431373, 0.5490196078431373],
             [0.5882352941176471, 0.5882352941176471, 0.5882352941176471],
             [0.6274509803921569, 0.6274509803921569, 0.6274509803921569],
             [0.6666666666666666, 0.6666666666666666, 0.6666666666666666],
             [0.7058823529411765, 0.7058823529411765, 0.7058823529411765],
             [0.7450980392156863, 0.7450980392156863, 0.7450980392156863],
             [0.7843137254901961, 0.7843137254901961, 0.7843137254901961],
             [0.8235294117647058, 0.8235294117647058, 0.8235294117647058],
             [0.8627450980392157, 0.8627450980392157, 0.8627450980392157],
             [0.9019607843137255, 0.9019607843137255, 0.9019607843137255],
             [0.9411764705882353, 0.9411764705882353, 0.9411764705882353],
             [0.9803921568627451, 0.9803921568627451, 0.9803921568627451],
             [0.9803921568627451, 0.9803921568627451, 0.9803921568627451]]
    lev=np.array([0.  , 0.02, 0.04, 0.06, 0.08, 0.1 , 0.12, 0.14, 0.16, 0.2 , 0.24,
       0.28, 0.32, 0.36, 0.4 , 0.44, 0.48, 0.52, 0.56, 0.6 , 0.64, 0.68,
       0.72, 0.76, 0.8 , 0.9 , 1.  ])
    if encoded:
        lev*=1e4
    nil = cols.pop(0)
    under = cols[0]
    over = cols.pop()
    cmap=mpl.colors.ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = mpl.colors.BoundaryNorm(lev, cmap.N)
    return cmap,norm


def ir_cmap(encoded=True):
    cols=[[0,0,0],[1.0, 1.0, 1.0],
     [0.9803921568627451, 0.9803921568627451, 0.9803921568627451],
     [0.9411764705882353, 0.9411764705882353, 0.9411764705882353],
     [0.9019607843137255, 0.9019607843137255, 0.9019607843137255],
     [0.8627450980392157, 0.8627450980392157, 0.8627450980392157],
     [0.8235294117647058, 0.8235294117647058, 0.8235294117647058],
     [0.7843137254901961, 0.7843137254901961, 0.7843137254901961],
     [0.7450980392156863, 0.7450980392156863, 0.7450980392156863],
     [0.7058823529411765, 0.7058823529411765, 0.7058823529411765],
     [0.6666666666666666, 0.6666666666666666, 0.6666666666666666],
     [0.6274509803921569, 0.6274509803921569, 0.6274509803921569],
     [0.5882352941176471, 0.5882352941176471, 0.5882352941176471],
     [0.5490196078431373, 0.5490196078431373, 0.5490196078431373],
     [0.5098039215686274, 0.5098039215686274, 0.5098039215686274],
     [0.47058823529411764, 0.47058823529411764, 0.47058823529411764],
     [0.43137254901960786, 0.43137254901960786, 0.43137254901960786],
     [0.39215686274509803, 0.39215686274509803, 0.39215686274509803],
     [0.35294117647058826, 0.35294117647058826, 0.35294117647058826],
     [0.3137254901960784, 0.3137254901960784, 0.3137254901960784],
     [0.27450980392156865, 0.27450980392156865, 0.27450980392156865],
     [0.23529411764705882, 0.23529411764705882, 0.23529411764705882],
     [0.19607843137254902, 0.19607843137254902, 0.19607843137254902],
     [0.1568627450980392, 0.1568627450980392, 0.1568627450980392],
     [0.11764705882352941, 0.11764705882352941, 0.11764705882352941],
     [0.0784313725490196, 0.0784313725490196, 0.0784313725490196],
     [0.0392156862745098, 0.0392156862745098, 0.0392156862745098],
     [0.0, 0.803921568627451, 0.803921568627451]]
    lev=np.array([-110. , -105.2,  -95.2,  -85.2,  -75.2,  -65.2,  -55.2,  -45.2,
        -35.2,  -28.2,  -23.2,  -18.2,  -13.2,   -8.2,   -3.2,    1.8,
          6.8,   11.8,   16.8,   21.8,   26.8,   31.8,   36.8,   41.8,
         46.8,   51.8,   90. ,  100. ])
    if encoded:
        lev*=1e2
    nil = cols.pop(0)
    under = cols[0]
    over = cols.pop()
    cmap=mpl.colors.ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = mpl.colors.BoundaryNorm(lev, cmap.N)
    return cmap,norm         


def c09_cmap(encoded=True):
    cols=[
    [1.000000, 0.000000, 0.000000],
    [1.000000, 0.031373, 0.000000],
    [1.000000, 0.062745, 0.000000],
    [1.000000, 0.094118, 0.000000],
    [1.000000, 0.125490, 0.000000],
    [1.000000, 0.156863, 0.000000],
    [1.000000, 0.188235, 0.000000],
    [1.000000, 0.219608, 0.000000],
    [1.000000, 0.250980, 0.000000],
    [1.000000, 0.282353, 0.000000],
    [1.000000, 0.313725, 0.000000],
    [1.000000, 0.349020, 0.003922],
    [1.000000, 0.380392, 0.003922],
    [1.000000, 0.411765, 0.003922],
    [1.000000, 0.443137, 0.003922],
    [1.000000, 0.474510, 0.003922],
    [1.000000, 0.505882, 0.003922],
    [1.000000, 0.537255, 0.003922],
    [1.000000, 0.568627, 0.003922],
    [1.000000, 0.600000, 0.003922],
    [1.000000, 0.631373, 0.003922],
    [1.000000, 0.666667, 0.007843],
    [1.000000, 0.698039, 0.007843],
    [1.000000, 0.729412, 0.007843],
    [1.000000, 0.760784, 0.007843],
    [1.000000, 0.792157, 0.007843],
    [1.000000, 0.823529, 0.007843],
    [1.000000, 0.854902, 0.007843],
    [1.000000, 0.886275, 0.007843],
    [1.000000, 0.917647, 0.007843],
    [1.000000, 0.949020, 0.007843],
    [1.000000, 0.984314, 0.011765],
    [0.968627, 0.952941, 0.031373],
    [0.937255, 0.921569, 0.050980],
    [0.901961, 0.886275, 0.074510],
    [0.870588, 0.854902, 0.094118],
    [0.835294, 0.823529, 0.117647],
    [0.803922, 0.788235, 0.137255],
    [0.772549, 0.756863, 0.160784],
    [0.737255, 0.725490, 0.180392],
    [0.705882, 0.690196, 0.200000],
    [0.670588, 0.658824, 0.223529],
    [0.639216, 0.623529, 0.243137],
    [0.607843, 0.592157, 0.266667],
    [0.572549, 0.560784, 0.286275],
    [0.541176, 0.525490, 0.309804],
    [0.509804, 0.494118, 0.329412],
    [0.474510, 0.462745, 0.349020],
    [0.752941, 0.749020, 0.909804],
    [0.800000, 0.800000, 0.929412],
    [0.850980, 0.847059, 0.945098],
    [0.898039, 0.898039, 0.964706],
    [0.949020, 0.949020, 0.980392],
    [1.000000, 1.000000, 1.000000],
    [0.964706, 0.980392, 0.964706],
    [0.929412, 0.960784, 0.929412],
    [0.890196, 0.937255, 0.890196],
    [0.854902, 0.917647, 0.854902],
    [0.815686, 0.894118, 0.815686],
    [0.780392, 0.874510, 0.780392],
    [0.745098, 0.850980, 0.745098],
    [0.705882, 0.831373, 0.705882],
    [0.670588, 0.807843, 0.670588],
    [0.631373, 0.788235, 0.631373],
    [0.596078, 0.764706, 0.596078],
    [0.560784, 0.745098, 0.560784],
    [0.521569, 0.721569, 0.521569],
    [0.486275, 0.701961, 0.486275],
    [0.447059, 0.678431, 0.447059],
    [0.411765, 0.658824, 0.411765],
    [0.376471, 0.635294, 0.376471],
    [0.337255, 0.615686, 0.337255],
    [0.301961, 0.592157, 0.301961],
    [0.262745, 0.572549, 0.262745],
    [0.227451, 0.549020, 0.227451],
    [0.192157, 0.529412, 0.192157],
    [0.152941, 0.505882, 0.152941],
    [0.117647, 0.486275, 0.117647],
    [0.078431, 0.462745, 0.078431],
    [0.043137, 0.443137, 0.043137],
    [0.003922, 0.419608, 0.003922],
    [0.003922, 0.431373, 0.027451],
    [0.003922, 0.447059, 0.054902],
    [0.003922, 0.462745, 0.082353],
    [0.003922, 0.478431, 0.109804],
    [0.003922, 0.494118, 0.137255],
    [0.003922, 0.509804, 0.164706],
    [0.003922, 0.525490, 0.192157],
    [0.003922, 0.541176, 0.215686],
    [0.003922, 0.556863, 0.243137],
    [0.007843, 0.568627, 0.270588],
    [0.007843, 0.584314, 0.298039],
    [0.007843, 0.600000, 0.325490],
    [0.007843, 0.615686, 0.352941],
    [0.007843, 0.631373, 0.380392],
    [0.007843, 0.647059, 0.403922],
    [0.007843, 0.662745, 0.431373],
    [0.007843, 0.678431, 0.458824],
    [0.007843, 0.694118, 0.486275],
    [0.011765, 0.705882, 0.513725],
    [0.011765, 0.721569, 0.541176],
    [0.011765, 0.737255, 0.568627],
    [0.011765, 0.752941, 0.596078],
    [0.011765, 0.768627, 0.619608],
    [0.011765, 0.784314, 0.647059],
    [0.011765, 0.800000, 0.674510],
    [0.011765, 0.815686, 0.701961],
    [0.011765, 0.831373, 0.729412],
    [0.015686, 0.843137, 0.756863],
    [0.015686, 0.858824, 0.784314],
    [0.015686, 0.874510, 0.807843],
    [0.015686, 0.890196, 0.835294],
    [0.015686, 0.905882, 0.862745],
    [0.015686, 0.921569, 0.890196],
    [0.015686, 0.937255, 0.917647],
    [0.015686, 0.952941, 0.945098],
    [0.015686, 0.968627, 0.972549],
    [1.000000, 1.000000, 1.000000]]
    
    return ListedColormap(cols),None


# In[14]:


# Plot using default cmap

def visualize_result(y_test,y_preds,idx,ax):
    cmap_dict = lambda s: {'cmap':get_cmap(s,encoded=True)[0], 'norm':get_cmap(s,encoded=True)[1],
                           'vmin':get_cmap(s,encoded=True)[2], 'vmax':get_cmap(s,encoded=True)[3]}
    ax[0].imshow(x_test['ir069'][idx,:,:,0],**cmap_dict('ir069'))
    ax[1].imshow(x_test['ir107'][idx,:,:,0],**cmap_dict('ir107'))
    ax[2].imshow(x_test['lght'][idx,:,:,0],cmap='hot',vmin=0,vmax=10)
    ax[3].imshow(y_test['vil'][idx,:,:,0],**cmap_dict('vil'))
    for k in range(len(y_preds)):
        if isinstance(y_preds[k],(list,)):
            yp=y_preds[k][0]
        else:
            yp=y_preds[k]
        ax[4+k].imshow(yp[idx,:,:,0],**cmap_dict('vil'))
    for i in range(len(ax)):
        ax[i].xaxis.set_ticks([])
        ax[i].yaxis.set_ticks([])
    


# In[17]:


test_idx = [123,456,789]
N=len(test_idx)
fig,ax = plt.subplots(N,7,figsize=(12,4))
for k,i in enumerate(test_idx):
    visualize_result(y_test,[y_pred_mse,y_pred_mse_vgg,y_pred_gan], i, ax[k] )

ax[0][0].set_title('Input ir069')
ax[0][1].set_title('Input ir107')
ax[0][2].set_title('Input lght')
ax[0][3].set_title('Truth')
ax[0][4].set_title('Output\nMSE Loss')
ax[0][5].set_title('Output\nMSE+VGG Loss')
ax[0][6].set_title('Output\nGAN+MAE Loss')
output = plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.05,
                    wspace=0.35)


# In[ ]:




