import numpy as np
import pandas as pd
import pickle
import streamlit as st
# from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu


loaded_model = pickle.load(open("trained_model.sav",'rb'))

def diabetes_prediction(input_data):
    # scaler = StandardScaler()

    input_data_as_np_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_np_array.reshape(1, -1)

    # std_data = scaler.transform(input_data_reshaped)
    # print(std_data)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'person is not diabetic'
    else:
        return 'person is diabetic'

def main():

    # giving a title
    st.title('Diabetes prediction web app')

    # getting the input data from user
    # columns for input fields :
    col1, col2, col3 = st.columns(3)

    with col1 :
        Pregnancies = st.text_input('Number of pregnancies')
    with col2:
        Glucose = st.text_input('Glucose level')
    with col3:
        BloodPressure = st.text_input('BP value')
    with col1:
        SkinThickness = st.text_input('SkinThickness value')
    with col2:
        Insulin = st.text_input('Insulin level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes pedigree function value')
    with col2:
        Age = st.text_input('Age')





    # code for prediction

    diagnosis = ''

    # creating a button for prediction

    if st.button('Diabetes test result'):
        diagnosis = diabetes_prediction([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

    st.success(diagnosis)

if __name__ == '__main__':
    main()

with st.sidebar:

    selected = option_menu("Multiple Disease Predictive system",
                           ['Diabetes Prediction'
                            ],
                           icons = ['activity'],
                           default_index=0)

# icons name have to be taken from bootstrap

if (selected =='Diabetes Prediction'):

    st.title('Diabetes Prediction using ML')


# streamlit run "filepath"

