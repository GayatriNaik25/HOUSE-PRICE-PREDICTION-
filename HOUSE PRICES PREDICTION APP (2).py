import numpy as np
import matplotlib as plt
import pandas as pd
import streamlit as st
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

dataset = pd.read_csv(r"D:\DATA SCIENCE CLASS NOTES\ml\6th- slr\SLR - House price prediction\House_data.csv")

st.title("House Price App")

st.write('Dataset perivew')
st.write(dataset.head())

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

#Spliting the dataset into the training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=4,weights="distance",p=1)
classifier.fit(X_train,y_train)

age =st.number_input('Enter value:',min_value=18,max_value=100,step=1)
salary=st.number_input('Enter a estimate salary:',min_value=150000 ,max_value=200000,step=1000)

if st.button('Predict'):
    input_data = np.array([[age, salary]])
    input_data_scaled = sc.transform(input_data)  # Scale the input
    prediction = classifier.predict(input_data_scaled)

    # Display the result
    st.success(f'The predicted house price is: ${prediction[0]:,.2f}')

# Optional: Display model accuracy
accuracy = classifier.score(X_test, y_test)
st.write(f'Model R² Score: {accuracy:.2f}')
