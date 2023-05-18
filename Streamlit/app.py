import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.regression import load_model, predict_model
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(layout="wide")
filename = 'new_gb_pipeline.pkl'
# Load the training data
training_data = pd.read_csv('train.csv')  # Adjust this to your training data file

def predict(model, features_df):
    predictions_df = predict_model(estimator=model, data=features_df)
    predictions = predictions_df['prediction_label'][0]
    return predictions
#loaded_model = pickle.load(open(filename, 'rb'))

model = load_model('new_gb_pipeline')

from PIL import Image
image=Image.open('blueberries1.jpg')
st.image(image, use_column_width = False)

st.title('Wild Blueberry Yield Prediction Web App')
st.write('This is a web app to predict the yield of wild blueberries based on\
         several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction of the regressor.')

st.sidebar.info('This app is created by Saadia Humayun (ERP: 27269), as part of my Machine Learning Project.')
st.sidebar.success('View Github repository here: https://github.com/saadiahumayun/streamlit-pycaret-webapp')

clonesize = st.sidebar.slider(label = 'Clone Size', min_value = 10.0,
                          max_value = 40.0 ,
                          value = 11.0,
                          step = 0.5)

honeybee = st.sidebar.slider(label = 'Honeybee', min_value = 0.00,
                          max_value = 0.00 ,
                          value = 1.00,
                          step = 0.1)
                          
bumbles = st.sidebar.slider(label = 'Bumbles', min_value = 0.2,
                          max_value = 0.40 ,
                          value = 0.25,
                          step = 0.01)                          

andrena = st.sidebar.slider(label = 'Andrena', min_value = 0.20,
                          max_value = 0.75 ,
                          value = 0.38,
                          step = 0.01)

osmia = st.sidebar.slider(label = 'Osmia', min_value = 0.20,
                          max_value = 0.75 ,
                          value = 0.50,
                          step = 0.01)
   
MaxOfUpperTRange = st.sidebar.slider(label = 'Upper T-Range (Max.)', min_value = 65.0,
                          max_value = 95.0,
                          value = 77.4,
                          step = 0.1)

MinOfUpperTRange = st.sidebar.slider(label = 'Upper T-Range (Min.)', min_value = 40.0,
                          max_value = 60.0 ,
                          value = 57.2,
                          step = 0.1)

AverageOfUpperTRange = st.sidebar.slider(label = 'Average Upper T-Range', min_value = 58.0,
                          max_value = 80.0 ,
                          value = 64.7,
                          step = 0.1)

MaxOfLowerTRange = st.sidebar.slider(label = 'Lower T-Range (Max.)', min_value = 50.0,
                          max_value = 70.0,
                          value = 55.8,
                          step = 0.1)

MinOfLowerTRange = st.sidebar.slider(label = 'Lower T-Range (Min.)', min_value = 24.0,
                          max_value = 35.0 ,
                          value = 27.0,
                          step = 0.1)

AverageOfLowerTRange = st.sidebar.slider(label = 'Average Lower T-Range', min_value = 41.0,
                          max_value = 56.0 ,
                          value = 50.8,
                          step = 0.1)
                          
RainingDays = st.sidebar.slider(label = 'Raining Days', min_value = 1.00,
                          max_value = 35.0,
                          value = 16.0,
                          step = 1.0)

AverageRainingDays = st.sidebar.slider(label = 'Average Raining Days', min_value = 0.05,
                          max_value = 0.60,
                          value = 0.26,
                          step = 0.1)

fruitset = st.sidebar.slider(label = 'Fruit set', min_value = 0.1900,
                          max_value = 0.6500,
                          value = 0.4290,
                          step = 0.0001)

fruitmass = st.sidebar.slider(label = 'Fruit mass', min_value = 0.300,
                          max_value = 0.550,
                          value = 0.431,
                          step = 0.001)

seeds = st.sidebar.slider(label = 'Seeds', min_value = 20.0,
                          max_value = 50.0,
                          value = 38.45,
                          step = 0.01)

features = {'clonesize':clonesize, 'honeybee': honeybee,
            'bumbles': bumbles, 'andrena': andrena,
            'osmia': osmia, 'MaxOfUpperTRange': MaxOfUpperTRange,
            'MinOfUpperTRange': MinOfUpperTRange, 'AverageOfUpperTRange': AverageOfUpperTRange,
            'MaxOfLowerTRange': MaxOfLowerTRange,
            'MinOfLowerTRange': MinOfLowerTRange, 'AverageOfLowerTRange': AverageOfLowerTRange,
            'RainingDays': RainingDays, 'AverageRainingDays': AverageRainingDays,'fruitset': fruitset, 'fruitmass': fruitmass, 'seeds': seeds
            }
 

features_df = pd.DataFrame([features])
st.subheader ('Please adjust feature values from the sidebar.')
st.dataframe(features_df)
# Load the LIME explainer model
explainer = LimeTabularExplainer(training_data.values, feature_names=features_df, mode='regression')


if st.button('Predict'):
    
    predictions = predict(model, features_df)
    # Generate explanations using LIME
    explanation = explainer.explain_instance(features_df.values[0], model.predict, num_features=8)

    # Interpret and display the explanation
    top_features = explanation.as_list()
    
    st.write('Based on feature values, your blueberry yield is '+ str(predictions), ' tonnes.') 
    
    st.subheader('LIME Explanation:')
    for feature in top_features:
        st.write(f"Feature: {feature[0]}, Weight: {feature[1]}")

