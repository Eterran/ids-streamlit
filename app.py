import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

df = pd.read_csv('study_performance.csv')

st.title('Student Performance Dashboard')

le_gender = preprocessing.LabelEncoder()
le_race_ethnicity = preprocessing.LabelEncoder()
le_parental_level_of_education = preprocessing.LabelEncoder()
le_lunch = preprocessing.LabelEncoder()
le_test_preparation_course = preprocessing.LabelEncoder()

df['gender'] = le_gender.fit_transform(df['gender'])
df['race_ethnicity'] = le_race_ethnicity.fit_transform(df['race_ethnicity'])
df['parental_level_of_education'] = le_parental_level_of_education.fit_transform(df['parental_level_of_education'])
df['lunch'] = le_lunch.fit_transform(df['lunch'])
df['test_preparation_course'] = le_test_preparation_course.fit_transform(df['test_preparation_course'])

st.sidebar.header('User Input Features')
gender = st.sidebar.selectbox('Gender', list(le_gender.classes_))
race_ethnicity = st.sidebar.selectbox('Race/Ethnicity', list(le_race_ethnicity.classes_))
parental_level_of_education = st.sidebar.selectbox('Parental Level of Education', list(le_parental_level_of_education.classes_))
lunch = st.sidebar.selectbox('Lunch', list(le_lunch.classes_))
test_preparation_course = st.sidebar.selectbox('Test Preparation Course', list(le_test_preparation_course.classes_))

selected_features = {
    'gender': le_gender.transform([gender])[0],
    'race_ethnicity': le_race_ethnicity.transform([race_ethnicity])[0],
    'parental_level_of_education': le_parental_level_of_education.transform([parental_level_of_education])[0],
    'lunch': le_lunch.transform([lunch])[0],
    'test_preparation_course': le_test_preparation_course.transform([test_preparation_course])[0]
}

X = df[['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']]
y = df[['math_score', 'reading_score', 'writing_score']]

scaler = MinMaxScaler().set_output(transform="pandas")
x_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X), columns = X.columns)

X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=24424)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(pd.DataFrame([selected_features]))

st.write(f'Predicted Math Score: {prediction[0][0]:.2f}')
st.write(f'Predicted Reading Score: {prediction[0][1]:.2f}')
st.write(f'Predicted Writing Score: {prediction[0][2]:.2f}')

st.subheader('Score Distributions')
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.histplot(df['math_score'], kde=True, ax=ax[0])
ax[0].set_title('Math Score Distribution')
sns.histplot(df['reading_score'], kde=True, ax=ax[1])
ax[1].set_title('Reading Score Distribution')
sns.histplot(df['writing_score'], kde=True, ax=ax[2])
ax[2].set_title('Writing Score Distribution')
st.pyplot(fig)

st.subheader('Correlation Heatmap')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)