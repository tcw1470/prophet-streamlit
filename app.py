import copy
import json

import streamlit as st

st.set_page_config(
    page_title="Forecast visualizer", page_icon="", initial_sidebar_state="collapsed"
)
st.title( "Forecast visualizer" )
st.markdown("# ")
st.markdown("## ")
st.write(
    "Hello!"
)
import plotly.express as px

import polars as pol
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Create a Prophet model instance
model = Prophet()

     
@st.cache_data
def load_new_data(filename): # private_repository requires URL of resources in relative form
   df = pd.read_csv(filename)
   return df

df = load_new_data(filename='https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')

st.header("Data in the current dataframe")
st.write("(uniformly sampled 10 rows for display)")

n, ncols = df.shape
k = n//10
st.dataframe( df.iloc[::k,:] )

fl = st.file_uploader(":file_folder: Upload new dataframe with exact column names",type=(["csv","txt","xlsx","xls"]))
if fl is not None:
   filename = fl.name
   st.write( 'Reading in ' + filename )
   try:
      df = load_new_data(filename);
   except Exception as e:
      st.write( e ) 
   try:
      df.rename(columns={ df.columns[0]: "ds" }, inplace = True)
      df.rename(columns={ df.columns[1]: "y" }, inplace = True)
   except Exception as e:
      st.write( e ) 


st.header("Data from the loaded file")
st.scatter_chart(data=df, 
                 x='ds', 
                 y=['y' ],
                 color=['#FF0000'],
                 size = 2 
                 )      

st.write("Below we use Python's prophet to forecast...")

# Train the model with the prepared data
model.fit(df)

# Create a dataframe for the future period to be predicted
future_df = model.make_future_dataframe(periods=10, freq='MS')

# Perform prediction using the model
forecast = model.predict(future_df)

# ------------------ Visualization of prediction results using Prophet ------------------
fig = model.plot(forecast)
try:
    st.header("Predictive performance visualized")
    st.plotly_chart(fig, use_container_width=True)     
except:
    pass

try:
    fig2 = model.plot_components(forecast);
    st.header("Weekly and seasonality trends")
    st.plotly_chart(fig2, use_container_width=True)     
except:
    pass

# ------------------ Visualization of prediction results using Prophet ------------------
# Perform cross-validation to evaluate model performance
cv_results = cross_validation(model=model,
                              initial=pd.to_timedelta(30*20, unit='D'),
                              period=pd.to_timedelta(30 * 5, unit='D'),
                              horizon=pd.to_timedelta(30*12, unit='D')
                              )


# Getting the min and max date 
endDate = pd.to_datetime( df["ds"] ).max()
try:
    startDate = endDate - pd.DateOffset( months = 12 )
except:
    startDate = endDate - pd.DateOffset( days = 30 )

# Set a 'cutoff' date for performance evaluation
cv_results['cutoff'] = pd.to_datetime(startDate)

# Calculate performance metrics using performance_metrics
df_performance = performance_metrics(cv_results)
st.header("Evaluation of model-fitting: results of cross-validation")
st.dataframe( df_performance )
