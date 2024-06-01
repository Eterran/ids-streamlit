import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics, preprocessing
import xgboost as xgb

def normalize(scores):
    max_score = max(scores)
    return [score / max_score for score in scores]

df = pd.read_csv('study_performance.csv')

st.markdown("""
# Student Performance Dashboard
Predict your scores based on your personal and study details.
""")

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

st.sidebar.header('Enter Your Details')
gender = st.sidebar.selectbox('Gender', list(le_gender.classes_), index=list(le_gender.classes_).index('female'))
race_ethnicity = st.sidebar.selectbox('Race/Ethnicity', list(le_race_ethnicity.classes_), index=list(le_race_ethnicity.classes_).index('group A'))
parental_level_of_education = st.sidebar.selectbox('Parental Level of Education', list(le_parental_level_of_education.classes_), index=list(le_parental_level_of_education.classes_).index('some high school'))
lunch = st.sidebar.selectbox('Lunch', list(le_lunch.classes_), index=list(le_lunch.classes_).index('free/reduced'))
test_preparation_course = st.sidebar.selectbox('Test Preparation Course', list(le_test_preparation_course.classes_), index=list(le_test_preparation_course.classes_).index('none'))

selected_features = {
    'gender': le_gender.transform([gender])[0],
    'race_ethnicity': le_race_ethnicity.transform([race_ethnicity])[0],
    'parental_level_of_education': le_parental_level_of_education.transform([parental_level_of_education])[0],
    'lunch': le_lunch.transform([lunch])[0],
    'test_preparation_course': le_test_preparation_course.transform([test_preparation_course])[0]
}

user_math_score = {'math_score': st.sidebar.number_input('Enter your Math Score', min_value=0, max_value=100, value=50)}
user_reading_score = {'reading_score': st.sidebar.number_input('Enter your Reading Score', min_value=0, max_value=100, value=50)}
user_writing_score = {'writing_score': st.sidebar.number_input('Enter your Writing Score', min_value=0, max_value=100, value=50)}

X = df[['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'math_score', 'reading_score', 'writing_score']]

X_math = pd.concat([X.iloc[:, :-3], X.iloc[:, -2:]], axis=1) # select all columns except writing_score
y_math = X.iloc[:, -3]
math_best_params = {'colsample_bytree': 0.6, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 1.0}
model_math = xgb.XGBRegressor(objective='reg:squarederror', random_state=60, **math_best_params).fit(X_math, y_math)

X_reading = pd.concat([X.iloc[:, :-2], X.iloc[:, -1:]], axis=1)
y_reading = X.iloc[:, -2]
model_reading = LinearRegression().fit(X_reading, y_reading)

X_writing = pd.concat([X.iloc[:, :-1]], axis=1)
y_writing = X.iloc[:, -1]
writing_best_params = {'colsample_bytree_writing': 0.6, 'gamma_writing': 0, 'learning_rate_writing': 0.01, 'max_depth_writing': 3, 'n_estimators_writing': 50, 'subsample_writing': 0.6}
model_writing = xgb.XGBRegressor(objective='reg:squarederror', random_state=60, **writing_best_params).fit(X_writing, y_writing)

user_input = [selected_features['gender'], selected_features['race_ethnicity'], selected_features['parental_level_of_education'],
              selected_features['lunch'], selected_features['test_preparation_course']]
user_input_math = user_input + [user_reading_score['reading_score'], user_writing_score['writing_score']]
user_input_reading = user_input + [user_math_score['math_score'], user_writing_score['writing_score']]
user_input_writing = user_input + [user_math_score['math_score'], user_reading_score['reading_score']]

if user_math_score and user_reading_score and user_writing_score:
    with st.spinner('Predicting...'):
        predicted_math_score = min(max(model_math.predict([user_input_math])[0], 0), 100)
        predicted_reading_score = min(max(model_reading.predict([user_input_reading])[0], 0), 100)
        predicted_writing_score = min(max(model_writing.predict([user_input_writing])[0], 0), 100)
    
    st.subheader('Predicted Scores')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'**Math Score:** {predicted_math_score:.0f}')
    with col2:
        st.markdown(f'**Reading Score:** {predicted_reading_score:.0f}')
    with col3:
        st.markdown(f'**Writing Score:** {predicted_writing_score:.0f}')

    avg_predicted_score = (predicted_math_score + predicted_reading_score + predicted_writing_score) / 3
    avg_user_score = (user_math_score['math_score'] + user_reading_score['reading_score'] + user_writing_score['writing_score']) / 3
    st.write(f'Average Predicted Score: {avg_predicted_score:.0f}')
    st.write(f'Your Average Score: {avg_user_score:.0f}')
    
    st.subheader('Suggestions and insights')
    
    if user_math_score['math_score'] > predicted_math_score:
        st.write('Your math score is above the predicted average. Keep it up!')
    elif user_math_score['math_score'] <= predicted_math_score:
        st.write('Your math score is below the predicted average. Consider additional study.')
        user_input_with_standard_lunch_and_completed_course = user_input_math.copy()
        user_input_with_standard_lunch_and_completed_course[4] = le_lunch.fit_transform(['standard'])[0]
        user_input_with_standard_lunch_and_completed_course[5] = le_test_preparation_course.fit_transform(['completed'])[0]
        
        user_input_with_standard_lunch = user_input_math.copy()
        user_input_with_standard_lunch[4] = le_lunch.fit_transform(['standard'])[0]
        
        user_input_with_completed_course = user_input_math.copy()
        user_input_with_completed_course[5] = le_test_preparation_course.fit_transform(['completed'])[0]
        
        if lunch != 'standard' and test_preparation_course != 'completed':
            predicted_math_score_with_standard_lunch_and_completed_course = model_math.predict([user_input_with_standard_lunch_and_completed_course])[0]
            
            if predicted_math_score_with_standard_lunch_and_completed_course > user_math_score['math_score']:
                st.write(f'If you ate standard lunch and completed the preparation course, your math score could increase up to {predicted_math_score_with_standard_lunch_and_completed_course:.0f}.')
        else:
            if lunch != 'standard':
                predicted_math_score_with_standard_lunch = model_math.predict([user_input_with_standard_lunch])[0]
                
                if predicted_math_score_with_standard_lunch > user_math_score['math_score']:
                    st.write(f'If you ate standard lunch, your math score could increase up to {predicted_math_score_with_standard_lunch:.0f}.')
                    
            if test_preparation_course != 'completed':
                predicted_math_score_with_completed_course = model_math.predict([user_input_with_completed_course])[0]
                
                if predicted_math_score_with_completed_course > user_math_score['math_score']:
                    st.write(f'If you completed the preparation course, your math score could increase up to {predicted_math_score_with_completed_course:.0f}.')
    st.write('')

    if user_reading_score['reading_score'] > predicted_reading_score:
        st.write('Your reading score is above the predicted average. Keep it up!')
    elif user_reading_score['reading_score'] <= predicted_reading_score:
        st.write('Your reading score is below the predicted average. Consider additional study.')
        
        if lunch != 'standard' and test_preparation_course != 'completed':
            user_input_with_standard_lunch_and_completed_course = user_input_reading.copy()
            user_input_with_standard_lunch_and_completed_course[4] = le_lunch.fit_transform(['standard'])[0]
            user_input_with_standard_lunch_and_completed_course[5] = le_test_preparation_course.fit_transform(['completed'])[0]
            predicted_reading_score_with_standard_lunch_and_completed_course = model_reading.predict([user_input_with_standard_lunch_and_completed_course])[0]
            
            if predicted_reading_score_with_standard_lunch_and_completed_course > user_writing_score['writing_score']:
                st.write(f'If you ate standard lunch and completed the preparation course, your reading score could increase up to {predicted_reading_score_with_standard_lunch_and_completed_course:.0f}.')
        else:
            if lunch != 'standard':
                user_input_with_standard_lunch = user_input_reading.copy()
                user_input_with_standard_lunch[4] = le_lunch.fit_transform(['standard'])[0]
                predicted_reading_score_with_standard_lunch = model_reading.predict([user_input_with_standard_lunch])[0]
                
                if predicted_reading_score_with_standard_lunch > user_writing_score['writing_score']:
                    st.write(f'If you ate standard lunch, your reading score could increase up to {predicted_reading_score_with_standard_lunch:.0f}.')
                    
            if test_preparation_course != 'completed':
                user_input_with_completed_course = user_input_reading.copy()
                user_input_with_completed_course[5] = le_test_preparation_course.fit_transform(['completed'])[0]
                predicted_reading_score_with_completed_course = model_reading.predict([user_input_with_completed_course])[0]
                
                if predicted_reading_score_with_completed_course > user_writing_score['writing_score']:
                    st.write(f'If you completed the preparation course, your reading score could increase up to {predicted_reading_score_with_completed_course:.0f}.')
    st.write('')
    
    if user_writing_score['writing_score'] > predicted_writing_score:
        st.write('Your writing score is above the predicted average. Keep it up!')
    elif user_writing_score['writing_score'] <= predicted_writing_score:
        st.write('Your writing score is below the predicted average. Consider additional study.')
        
        if lunch != 'standard' and test_preparation_course != 'completed':
            user_input_with_standard_lunch_and_completed_course = user_input_writing.copy()
            user_input_with_standard_lunch_and_completed_course[4] = le_lunch.fit_transform(['standard'])[0]
            user_input_with_standard_lunch_and_completed_course[5] = le_test_preparation_course.fit_transform(['completed'])[0]
            predicted_writing_score_with_standard_lunch_and_completed_course = model_writing.predict([user_input_with_standard_lunch_and_completed_course])[0]
            
            if predicted_writing_score_with_standard_lunch_and_completed_course > user_writing_score['writing_score']:
                st.write(f'If you ate standard lunch and completed the preparation course, your writing score could increase up to {predicted_writing_score_with_standard_lunch_and_completed_course:.0f}.')
        else:
            if lunch != 'standard':
                user_input_with_standard_lunch = user_input_writing.copy()
                user_input_with_standard_lunch[4] = le_lunch.fit_transform(['standard'])[0]
                predicted_writing_score_with_standard_lunch = model_writing.predict([user_input_with_standard_lunch])[0]
                
                if predicted_writing_score_with_standard_lunch > user_writing_score['writing_score']:
                    st.write(f'If you ate standard lunch, your writing score could increase up to {predicted_writing_score_with_standard_lunch:.0f}.')
            
            if test_preparation_course != 'completed':
                user_input_with_completed_course = user_input_writing.copy()
                user_input_with_completed_course[5] = le_test_preparation_course.fit_transform(['completed'])[0]
                predicted_writing_score_with_completed_course = model_writing.predict([user_input_with_completed_course])[0]
                
                if predicted_writing_score_with_completed_course > user_writing_score['writing_score']:
                    st.write(f'If you completed the preparation course, your writing score could increase up to {predicted_writing_score_with_completed_course:.0f}.')
    st.write('')
        
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split

# def display_metrics(model, X_test, y_test, target_name):
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     st.write(f'{target_name} Model Performance:')
#     st.write(f'Mean Squared Error: {mse:.2f}')
#     st.write(f'R² Score: {r2:.2f}')

# X_train_math, X_test_math, y_train_math, y_test_math = train_test_split(X_math, y_math, test_size=0.2, random_state=24424)
# X_train_reading, X_test_reading, y_train_reading, y_test_reading = train_test_split(X_reading, y_reading, test_size=0.2, random_state=24424)
# X_train_writing, X_test_writing, y_train_writing, y_test_writing = train_test_split(X_writing, y_writing, test_size=0.2, random_state=24424)

# model_math.fit(X_train_math, y_train_math)
# model_reading.fit(X_train_reading, y_train_reading)
# model_writing.fit(X_train_writing, y_train_writing)

# st.subheader('Model Performance Metrics')
# display_metrics(model_math, X_test_math, y_test_math, 'Math')
# display_metrics(model_reading, X_test_reading, y_test_reading, 'Reading')
# display_metrics(model_writing, X_test_writing, y_test_writing, 'Writing')