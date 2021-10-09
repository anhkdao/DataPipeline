from typing import Optional

#from fastapi import FastAPI
import streamlit as st
import os
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
import sys
import h5py
#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import uvicorn

#app = FastAPI()

log_files= {
    'mse':'./logs/sample-mse/metrics.csv',
    'mse+vgg':'./logs/sample-mse-vgg/metrics.csv',
    'gan':'./logs/sample-gan-mae/metrics.csv'
}
metrics={ k:pd.read_csv(v) for k,v in log_files.items()}
for k in metrics.keys():
	st.text(k)
	st.dataframe(metrics[k])
	st.line_chart(metrics[k])


#if __name__=='__main__':
#	uvicorn.run(app,host='0.0.0.0',port=8081)
#fig,ax=plt.subplots(2,3,figsize=(15,6))
#def plot_metrics_row(df,metric_names,ax):
#	for k,m in enumerate(metric_names):
#		df[[m,'val_'+m]].plot(ax=ax[k])
#		ax[k].grid(True,alpha=.25)

#@app.get("/")
#def radar():
#	st.write("Here's our first attempt at using data to create a table:")
#	a = st.write(pd.DataFrame({
#		'first column': [1, 2, 3, 4],
#		'second column': [10, 20, 30, 40]
#	}),unsafe_allow_html=True)
#	return a
#
#for k,df in metrics:
#	st.write("Here's our first attempt at using data to create a table:")
#	st.write(df,unsafe_allow_html=True)
#
#