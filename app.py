import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics, preprocessing
import xgboost as xgb
import plotly.graph_objects as go

# def normalize(scores):
#     max_score = max(scores)
#     return [score / max_score for score in scores]

df = pd.read_csv('study_performance.csv')

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
parental_level_of_education = st.sidebar.selectbox('Parental Level of Education', list(le_parental_level_of_education.classes_), index=list(le_parental_level_of_education.classes_).index('some high school'))
lunch = st.sidebar.selectbox('Lunch', list(le_lunch.classes_), index=list(le_lunch.classes_).index('free/reduced'))
test_preparation_course = st.sidebar.selectbox('Test Preparation Course', list(le_test_preparation_course.classes_), index=list(le_test_preparation_course.classes_).index('none'))

most_common_ethnicity = df['race_ethnicity'].mode()[0]

selected_features = {
    'gender': le_gender.transform([gender])[0],
    'race_ethnicity': most_common_ethnicity,
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




avg_scores_standard = df[df['lunch'] == le_lunch.transform(['standard'])[0]][['math_score', 'reading_score', 'writing_score']].mean()
avg_scores_reduced = df[df['lunch'] == le_lunch.transform(['free/reduced'])[0]][['math_score', 'reading_score', 'writing_score']].mean()
avg_scores_completed = df[df['test_preparation_course'] == le_test_preparation_course.transform(['completed'])[0]][['math_score', 'reading_score', 'writing_score']].mean()
avg_scores_none = df[df['test_preparation_course'] == le_test_preparation_course.transform(['none'])[0]][['math_score', 'reading_score', 'writing_score']].mean()
avg_scores_combined = df[(df['lunch'] == le_lunch.transform(['standard'])[0]) & 
                         (df['test_preparation_course'] == le_test_preparation_course.transform(['completed'])[0])][['math_score', 'reading_score', 'writing_score']].mean()
avg_scores_combined_reduced_none = df[(df['lunch'] == le_lunch.transform(['free/reduced'])[0]) & 
                                      (df['test_preparation_course'] == le_test_preparation_course.transform(['none'])[0])][['math_score', 'reading_score', 'writing_score']].mean()

def create_bar_chart_with_colors(labels, scores1, scores2, name1, name2, color1, color2):
    fig = go.Figure(data=[
        go.Bar(name=name1, x=labels, y=scores1, marker=dict(color=color1),
               hoverinfo='text', text=[f'{name1}: {score:.2f}' for score in scores1]),
        go.Bar(name=name2, x=labels, y=scores2, marker=dict(color=color2),
               hoverinfo='text', text=[f'{name2}: {score:.2f}' for score in scores2])
    ])
    fig.update_layout(
        xaxis_title='Subject',
        yaxis_title='Average Score',
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

categories = ['Math', 'Reading', 'Writing']

fig_lunch = create_bar_chart_with_colors(categories, avg_scores_standard, avg_scores_reduced, 'Standard Lunch', 'Free/Reduced Lunch', 'rgb(0, 0, 226)', 'rgb(108, 178, 255)')
fig_prep_course = create_bar_chart_with_colors(categories, avg_scores_completed, avg_scores_none, 'Completed Prep', 'No Prep', 'rgb(0, 200, 0)', 'rgb(153, 255, 153)')
fig_combined = create_bar_chart_with_colors(categories, avg_scores_combined, avg_scores_combined_reduced_none, 'Standard Lunch & Completed Prep', 'Free/Reduced Lunch & No Prep', 'rgb(255, 51, 51)', 'rgb(255, 153, 153)')

def calculate_percentage_difference(scores1, scores2):
    return (scores1 - scores2) / scores2 * 100

percent_diff_lunch = calculate_percentage_difference(avg_scores_standard, avg_scores_reduced)
percent_diff_prep_course = calculate_percentage_difference(avg_scores_completed, avg_scores_none)
percent_diff_combined = calculate_percentage_difference(avg_scores_combined, avg_scores_combined_reduced_none)

tab1, tab2 = st.tabs(["Dashboard", "Performance Graphs"])

user_input = [selected_features['gender'], selected_features['race_ethnicity'], selected_features['parental_level_of_education'],
                selected_features['lunch'], selected_features['test_preparation_course']]
user_input_math = user_input + [user_reading_score['reading_score'], user_writing_score['writing_score']]
user_input_reading = user_input + [user_math_score['math_score'], user_writing_score['writing_score']]
user_input_writing = user_input + [user_math_score['math_score'], user_reading_score['reading_score']]

user_input_with_standard_lunch_and_completed_course = user_input_math.copy()
user_input_with_standard_lunch_and_completed_course[4] = le_lunch.transform(['standard'])[0]
user_input_with_standard_lunch_and_completed_course[5] = le_test_preparation_course.transform(['completed'])[0]

predicted_math_score_with_standard_lunch_and_completed_course = model_math.predict([user_input_with_standard_lunch_and_completed_course])[0]
predicted_reading_score_with_standard_lunch_and_completed_course = model_reading.predict([user_input_with_standard_lunch_and_completed_course])[0]
predicted_writing_score_with_standard_lunch_and_completed_course = model_writing.predict([user_input_with_standard_lunch_and_completed_course])[0]

if user_math_score and user_reading_score and user_writing_score:
    with st.spinner('Predicting...'):
        predicted_math_score = min(max(model_math.predict([user_input_math])[0], 0), 100)
        predicted_reading_score = min(max(model_reading.predict([user_input_reading])[0], 0), 100)
        predicted_writing_score = min(max(model_writing.predict([user_input_writing])[0], 0), 100)

fig_improvement = go.Figure(data=[
    go.Bar(name='Current Predicted Scores', x=['Math', 'Reading', 'Writing'], y=[predicted_math_score, predicted_reading_score, predicted_writing_score],
           hoverinfo='text', text=[f'Current: {score:.2f}' for score in [predicted_math_score, predicted_reading_score, predicted_writing_score]]),
    go.Bar(name='With Standard Lunch & Completed Prep', x=['Math', 'Reading', 'Writing'], y=[predicted_math_score_with_standard_lunch_and_completed_course, predicted_reading_score_with_standard_lunch_and_completed_course, predicted_writing_score_with_standard_lunch_and_completed_course],
           hoverinfo='text', text=[f'Improved: {score:.2f}' for score in [predicted_math_score_with_standard_lunch_and_completed_course, predicted_reading_score_with_standard_lunch_and_completed_course, predicted_writing_score_with_standard_lunch_and_completed_course]])
])

# Update layout
fig_improvement.update_layout(
    title='Potential Improvement with Standard Lunch & Completed Prep',
    xaxis_title='Subject',
    yaxis_title='Score',
    barmode='group'
)

with tab1:
    st.markdown("""
    # Student Performance Dashboard
    <span style='color:grey'><i>Predict your scores based on your personal and study details.</i></span>
    """, unsafe_allow_html=True)

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
        
        avg_improvement_lunch = avg_scores_standard - avg_scores_reduced
        avg_improvement_prep = avg_scores_completed - avg_scores_none
        avg_improvement_combined = avg_scores_combined - avg_scores_combined_reduced_none

        st.subheader('Suggestions and insights')

        if lunch != 'standard' or test_preparation_course != 'completed':
            potential_math_improvement = min(max(user_math_score['math_score'] + avg_improvement_combined[0] if lunch != 'standard' and test_preparation_course != 'completed' else (user_math_score['math_score'] + avg_improvement_lunch[0] if lunch != 'standard' else user_math_score['math_score'] + avg_improvement_prep[0]), 0), 100)
            potential_reading_improvement = min(max(user_reading_score['reading_score'] + avg_improvement_combined[1] if lunch != 'standard' and test_preparation_course != 'completed' else (user_reading_score['reading_score'] + avg_improvement_lunch[1] if lunch != 'standard' else user_reading_score['reading_score'] + avg_improvement_prep[1]), 0), 100)
            potential_writing_improvement = min(max(user_writing_score['writing_score'] + avg_improvement_combined[2] if lunch != 'standard' and test_preparation_course != 'completed' else (user_writing_score['writing_score'] + avg_improvement_lunch[2] if lunch != 'standard' else user_writing_score['writing_score'] + avg_improvement_prep[2]), 0), 100)

            st.write(f'If you ate standard lunch and completed the preparation course, your scores could increase to:')
            st.write(f'**Math:** {potential_math_improvement:.0f}')
            st.write(f'**Reading:** {potential_reading_improvement:.0f}')
            st.write(f'**Writing:** {potential_writing_improvement:.0f}')

            fig_improvement = go.Figure(data=[
                go.Bar(name='Current Scores', x=['Math', 'Reading', 'Writing'], y=[user_math_score['math_score'], user_reading_score['reading_score'], user_writing_score['writing_score']], marker=dict(color='rgb(204, 154, 242)'),
                    hoverinfo='text', text=[f'Current: {score:.0f}' for score in [user_math_score['math_score'], user_reading_score['reading_score'], user_writing_score['writing_score']]]),
                go.Bar(name='Potential Improvement', x=['Math', 'Reading', 'Writing'], y=[potential_math_improvement, potential_reading_improvement, potential_writing_improvement],  marker=dict(color='rgb(138, 43, 226)'),
                    hoverinfo='text', text=[f'Improved: {score:.0f}' for score in [potential_math_improvement, potential_reading_improvement, potential_writing_improvement]])
            ])

            fig_improvement.update_layout(
                title='Potential Improvement with Standard Lunch & Completed Prep',
                xaxis_title='Subject',
                yaxis_title='Score',
                barmode='group'
            )
            st.plotly_chart(fig_improvement)
        else:
            st.write("You already have the best possible conditions for improvement with a standard lunch and completed prep course. Here are some additional tips to further enhance your performance:")
            st.write("- **Consistent Study Routine:** Ensure you have a dedicated study schedule.")
            st.write("Reference: [Education Corner](https://www.educationcorner.com/study-skills.html)")
            st.write("- **Practice Tests:** Regularly take practice tests to identify weak areas.")
            st.write("Reference: [Test Prep Toolkit](https://www.testpreptoolkit.com/importance-of-practice-tests)")
            st.write("- **Healthy Lifestyle:** Maintain a balanced diet and regular exercise to boost cognitive function.")
            st.write("Reference: [Harvard Health](https://www.health.harvard.edu/staying-healthy/the-importance-of-a-healthy-lifestyle)")
            st.write("- **Seek Help:** Don't hesitate to ask teachers or tutors for help in challenging subjects.")
            st.write("Reference: [Edutopia](https://www.edutopia.org/article/how-ask-help)")

            st.write("Based on these suggestions, your potential percentage improvement could be:")
            st.write("**Consistent Study Routine:** Up to 10% improvement.")
            st.write("**Practice Tests:** Up to 10% improvement.")
            st.write("**Healthy Lifestyle:** Up to 5% improvement.")
            st.write("**Seek Help:** Up to 7% improvement.")
                
with tab2:
    st.markdown("""Did you know? Students who eat standard lunch and complete the preparation course tend to perform better.""")
    st.subheader('Average Performance by Lunch Type')
    st.plotly_chart(fig_lunch)
    st.write(f'Percentage Difference: Math: {percent_diff_lunch[0]:.2f}%, Reading: {percent_diff_lunch[1]:.2f}%, Writing: {percent_diff_lunch[2]:.2f}%')

    st.subheader('Average Performance by Preparation Course')
    st.plotly_chart(fig_prep_course)
    st.write(f'Percentage Difference: Math: {percent_diff_prep_course[0]:.2f}%, Reading: {percent_diff_prep_course[1]:.2f}%, Writing: {percent_diff_prep_course[2]:.2f}%')

    st.subheader('Average Performance by Combined Factors')
    st.plotly_chart(fig_combined)
    st.write(f'Percentage Difference: Math: {percent_diff_combined[0]:.2f}%, Reading: {percent_diff_combined[1]:.2f}%, Writing: {percent_diff_combined[2]:.2f}%')