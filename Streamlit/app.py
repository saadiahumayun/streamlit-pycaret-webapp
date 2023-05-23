import streamlit as st
import pandas as pd
import numpy as np
import pickle
from utils import DataLoader
from sklearn.ensemble import GradientBoostingRegressor
from pycaret.regression import load_model, predict_model
from interpret.blackbox import LimeTabular
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import shap
import streamlit.components.v1 as components

# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()

# Create and train the Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor()
gb_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gb_regressor.predict(X_test)

st.set_page_config(layout="wide")
#filename = 'new_gb_pipeline.pkl'
# Load the training data
# training_data= pd.read_csv('train.csv')

def predict(data):
    #predictions_df = predict_model(estimator=model, data=features_df)
    predictions_df = gb_regressor.predict(features_df)
    predictions = predictions_df[0]
    #predictions = predictions_df['prediction_label'][0]
    return predictions.flatten()
#loaded_model = pickle.load(open(filename, 'rb'))

#model = load_model('new_gb_pipeline')

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

if st.button('Predict'):
    
    predictions = predict(features_df)
    st.write('Based on feature values, your blueberry yield is '+ str(predictions), ' tonnes.')


arr = features_df.to_numpy()

#def predict_fn(arr):
    #return predictions


st.button('Explain with SHAP')
    

    # %% Create SHAP explainer
explainer = shap.TreeExplainer(gb_regressor)
# Calculate shapley values for test data
shap_values = explainer.shap_values(features_df.iloc[0])
features_df.iloc[0]

# %% Investigating the values (classification problem)
# class 0 = contribution to class 1
# class 1 = contribution to class 2
print(shap_values[0].shape)
shap_values

# %% >> Visualize local predictions
shap.initjs()
# Force plot
prediction = gb_regressor.predict(features_df.iloc[0])
print(f"The GB predicted: {prediction}")
shap.force_plot(explainer.expected_value[1],
                shap_values[1],
                features_df.iloc[0]) # for values

# %% >> Visualize global features
# Feature summary
shap.summary_plot(shap_values, X_train)

    
    

if st.button('Explain with LIME'):
    # Load the LIME explainer model
    
    def predict_row(input_data):
        return gb_regressor.predict(input_data)

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,  
        mode='regression')


    # asking for explanation for LIME model
    i=0
    exp = explainer.explain_instance(features_df.iloc[0].values, 
                                     predict_row, num_features=16)
    
    
    top_features = exp.as_list()
    
    st.subheader('LIME Explanation:')
    
    for feature in top_features:
        st.write(f"Feature: {feature[0]}, Weight: {feature[1]}")
        
    # Display explainer HTML object
    components.html(exp.as_html(), height=800)
    #exp.as_pyplot_figure()

        
  
        

    
