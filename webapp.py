import numpy as np
import pickle
import streamlit as st
import sklearn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def cancer_prediction(input_data):
    # Loading the saved model
    loaded_model = pickle.load(open('trained_model.sav', 'rb'))
    # Change the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # Reshape the numpy array as we are predicting for one data point
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    # Fit the scaler to your data
    scaler.fit(input_data_reshaped)
    # Standardizing the input data
    input_data_std = scaler.transform(input_data_reshaped)
    # Giving the probability of the model
    prediction = loaded_model.predict(input_data_std)
    print(prediction)
    #building the condition
    if prediction[0] == 0:
        return 'The cell is Malignant (cancer cell)'
    else:
        return 'The cell is Benign (normal cell)'

#building the stream lit

def main():
    #giving the title
    st.title('Cancer Prediction')
    #creating the input data options--gettiing the input data from the user
    smoothness_mean=st.text_input('smoothness_mean')
    compactness_mean=st.text_input('compactness_mean')
    symmetry_mean=st.text_input('symmetry_mean')
    fractal_dimension_mean=st.text_input('fractal_dimension_mean')
    texture_se=st.text_input('texture_se')
    smoothness_se=st.text_input('smoothness_se')
    compactness_se=st.text_input('compactness_se')
    concavity_se=st.text_input('concavity_se')
    concave_points_se=st.text_input('concave_points_se')
    symmetry_se=st.text_input('symmetry_se')
    fractal_dimension_se=st.text_input('fractal_dimension_se')
    smoothness_worst=st.text_input('smoothness_worst')
    compactness_worst=st.text_input('compactness_worst')
    concavity_worst=st.text_input('concavity_worst')
    symmetry_worst=st.text_input('symmetry_worst')
    fractal_dimension_worst=st.text_input('fractal_dimension_worst')

    #code for prediction
    diagnosis=''

    #creating the button
    if st.button('Cancer test result'):
        diagnosis=cancer_prediction([smoothness_mean,compactness_mean,symmetry_mean,fractal_dimension_mean,texture_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,smoothness_worst,compactness_worst,concavity_worst,symmetry_worst,fractal_dimension_worst])

    
    st.success(diagnosis)



if __name__=='__main__':
    main()