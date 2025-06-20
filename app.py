import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load & siapkan data
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df = df.fillna(0)

# Fit model
X = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']
model = DecisionTreeClassifier().fit(X, y)

# Streamlit UI
st.title("Prediksi Kelangsungan Hidup Penumpang Titanic")

pclass = st.selectbox("Kelas Tiket (Pclass)", [1, 2, 3])
sex = st.selectbox("Jenis Kelamin", ['male', 'female'])
age = st.slider("Umur", 0, 100, 25)

if st.button("Prediksi"):
    input_df = pd.DataFrame([[pclass, 1 if sex == 'male' else 0, age]], columns=['Pclass', 'Sex', 'Age'])
    prediction = model.predict(input_df)[0]
    st.success("Survived" if prediction == 1 else "Did Not Survive")
