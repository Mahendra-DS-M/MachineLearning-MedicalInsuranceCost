###### Libraries #######

# Base Libraries
import pandas as pd
import numpy as np

# Deployment Library
import streamlit as st

# Model Pickled File Library
import joblib

############# Data File ###########

data = pd.read_excel("Medical Cost.xlsx")

data = data.dropna(axis=0).reset_index(drop=True)

########### Loading Trained Model Files ########
model = joblib.load("medicalcost_rfreg.pkl")


########## UI Code #############

# Ref: https://docs.streamlit.io/library/api-reference

# Title
st.header("Estimation of Medical Insurance Amount for the Given Person Details:")

# Image
st.image("health-Insurance.png")

# Description
st.write("""Built a Predictive model in Machine Learning to estimate the medical insurance charges a person can get.
         Sample Data taken as below shown.
""")

# Data Display
st.dataframe(data.head())
st.write("From the above data , charges is the prediction variable")

###### Taking User Inputs #########
st.subheader("Enter Below Details to Get the Estimation Charges:")

col1, col2, col3 = st.columns(3) # value inside brace defines the number of splits
col4, col5, col6 = st.columns(3)


with col1:
    age = st.number_input("Enter Person Age:")
    st.write(age)

with col2:
    gender = st.selectbox("Enter Person Gender:",data.gender.unique())
    st.write(gender)

with col3:
    bmi = st.number_input("Enter Person BMI:")
    st.write(bmi)

with col4:
    childs = st.number_input("Enter Number of Childs for Person:")
    st.write(childs)

with col5:
    smoker = st.selectbox("Enter Whether the Person Smoker or not:", data.smoker.unique())
    st.write(smoker)

with col6:
    region = st.selectbox("Enter Region of Person:", data.region.unique())
    st.write(region)

###### Predictions #########

if st.button("Estimate"):
    st.write("Data Given:")
    values = [age, gender, bmi, childs, smoker, region]
    record =  pd.DataFrame([values],
                           columns = ['age', 'gender',
                                      'bmi', 'children',
                                      'smoker', 'region'])
    
    st.dataframe(record)
    charges = round(model.predict(record)[0],2)
    charges = str(charges)+" $"
    st.subheader("Estimated Charges:")
    st.subheader(charges)