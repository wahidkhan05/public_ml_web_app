import pickle
import streamlit as st
from streamlit_option_menu import option_menu
 

loaded_model = pickle.load(open("trained_model.sav",'rb'))

with st.sidebar:

    selected = option_menu("Multiple Disease Predictive system",
                           ['Diabetes Prediction'
                            ],
                           icons = ['activity'],
                           default_index=0)

# icons name have to be taken from bootstrap

if (selected =='Diabetes Prediction'):
    # giving a title
    st.title('Diabetes Prediction using ML')




    # std_data = scaler.transform(input_data_reshaped)
    # print(std_data)


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
        diagnosis = loaded_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])


        if (diagnosis[0] == 1):
            diagnosis = 'person is diabetic'
        else:
            diagnosis = 'person is not diabetic'

    st.success(diagnosis)
# streamlit run "filepath"

print("end of page")

