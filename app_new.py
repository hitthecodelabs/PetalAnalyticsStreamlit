import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
.custom-font {
    font-family: 'Helvetica', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# Header with Image
st.image("iris_flowers_image.jpg", use_column_width=True)  # Replace with actual image path
st.write("# Simple Iris Flower Prediction App")
st.markdown("This app predicts the **Iris flower** type!")

# Sidebar
st.sidebar.header('User Input Parameters')
st.sidebar.markdown("Adjust the features to predict the Iris flower type:")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader('User Input parameters')
    st.dataframe(df.style.set_properties(**{'background-color': '#f0f0f0'}))  # Example of custom styling

# [Your existing model prediction code]

with col2:
    st.subheader('Prediction')
    st.markdown(f"<div class='custom-font'>Predicted Flower: {iris.target_names[prediction][0]}</div>", unsafe_allow_html=True)
    st.write('Confidence: {:.2f}%'.format(max(prediction_proba[0]) * 100))

# Interactive Bar Chart for Prediction Probabilities
st.subheader('Prediction Probability')
fig = px.bar(x=iris.target_names, y=prediction_proba[0], labels={'x': 'Flower Type', 'y': 'Probability'})
st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown('<p class="big-font custom-font">Iris Flower Prediction App © 2023</p>', unsafe_allow_html=True)
