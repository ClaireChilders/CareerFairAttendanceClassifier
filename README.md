**Claire Childers** — Career Fair Attendance Classifier using Decision Trees and Random Forest

> [!important] See the code here: <https://github.com/ClaireChilders/CareerFairAttendanceClassifier>

> [!note] `random_forest.py` holds the models training code and `preprocessing.py` defines the data preprocessing methods

> [!warning] The data used in this project is not included in the code because it holds sensitive student identifiers that is not permitted to be shared.

This project aims to utilize machine learning techniques (Decision Tree and Random Forest) to predict student attendance at upcoming career fairs. Through collaboration with the Oakland University Career and Life Design Center, extensive student engagement data from Handshake was utilized to train and evaluate the model's performance. 

This project not only focuses on predicting attendance, but also aims to uncover key factors influencing students' decisions to attend career fairs, providing valuable insights to future strategies to increase attendance rates.

---

## Idea

The main idea is to develop a machine learning model capable of predicting whether students would attend a career fair or not, based on various features extracted from student data.

## Outcomes

The main outcomes are to identify the most influential factors determining student attendance at career fairs, providing actionable insights to improve future attendance rates.

## Process

The project process involves several steps:

1. **Data Gathering**: Relevant student data, including Handshake activity and event attendance, along with career fair registration data is collected
2. **Data Processing**: Extensive data cleaning and feature engineering are performed to prepare the dataset for modeling. This includes handling null values, converting data types, and creating/adding binary features
3. **Model Selection and Training**: Initially, a Decision Tree model was trained and optimized, but the model was later switched to a Random Forest model for its ability to mitigate overfitting
4. **Hyperparameter Tuning**: Test various hyperparameters using GridSearchCV to optimize model performance
5. **Evaluation**: Assess the model using metrics like accuracy, precision, recall, and F1-score
6. **Feature Importance Analysis**: Generate feature importance rankings to determine the most and least significant predictors of attendance

## Relevant Data

The dataset included various student attributes, event attendance records, career fair data, and registration information, providing a comprehensive view of student engagement.

### Student Data

- `stu_is_archived` (saved to `student_data.csv`)
- `stu_is_activated` (saved to `student_data.csv`)
- `stu_is_visible` (saved to `student_data.csv`)
- `stu_is_work_study` (saved to `student_data.csv`)
- `stu_is_profile_complete` (saved to `student_data.csv`)
- `stu_grad_date` (saved to `student_data.csv`)
- `stu_creation_date` (saved to `student_data.csv`)
- `stu_login_date` (saved to `student_data.csv`)
- `stu_gpa` (saved to `student_data.csv`)
- `stu_majors` (saved to `student_data.csv`)
- `stu_colleges` (saved to `student_data.csv`)
- `stu_school_year` (saved to `student_data.csv`)
- `stu_applications` (saved to `student_counts_1.csv`)
- `stu_logins` (saved to `student_counts_1.csv`)
- `stu_appointments` (saved to `student_counts_1.csv`)
- `stu_attendances` (saved to `student_counts_2.csv`)
- `stu_work_experience` (saved to `student_counts_2.csv`)
- `stu_experiences` (saved to `student_counts_2.csv`)
- `appointment_types` (saved to `appointment_types.csv`)

### Student Event Attendance

`student_event_attendance.csv`

- `event_name`
- `event_type`
- `event_date`
- `event_categories`

### Student Fair Attendance

`student_fair_attendance.csv`

- `career_fair_date`
- `career_fair_name`

### Career Fair Data

`career_fair_data.csv`

- `career_fair_name`
- `career_fair_date`
- `career_fair_majors`

### Registration Data

`registration_data.csv`

- `is_pre_registered`
- `is_checked_in`
- `career_fair_name`
- `career_fair_date`

---

## Processed Data

The raw data undergoes extensive preprocessing, including null value handling, integer conversion, date conversion, and feature engineering, resulting in a cleaned dataset with over 980,000 rows and 120 features.

### 1. Load CSV Files

Open the CSV Files and merge them together

There should be a row for each career fair for every student. If the student never attended a career fair, there will be a row with all null values for that student.

The `appointment_df` only contains rows for students that have appointments. There can only be one row per student however, so we will just need to merge the appointment data with the student data to ensure that we have a row for each student.

The `registration_df` contains a row for each student that has registered for each career fair. There may be multiple rows for each student. We want a row for every student for every career fair, so we will need to merge the registration data with the student data.

To ensure that we have a row for each student for each career fair, we get the cross product of the student ids and career fair dates and merge that with the merged data. This ensures there is a row for each student for each career fair. Then, we merge the registration data with all that to add the registration columns to the merged data.

Merge career fair name and date to ensure that we have a unique identifier for each career fair (some have the same name but different dates)

### 2. Null Values

Fill all null values in "Yes/No" columns with "No"

Fill all count columns with 0

Fill all null colleges with "No College Designated"

Convert all "Yes/No" values to 1/0

### 3. Integer Conversion

The data contained some numerical strings in the format "1,000". These are converted to integers like 1000 

Then, convert all the numerical values to binary values based on threshold values for:

- Appointments
- Applications
- Logins
- Attendances
- Work Experiences
- Learning Experiences

### 4. Fair Attendance Cleaning

`stu_fair_attendance_df` contains a row for each career fair for each student that attended that career fair. We want to count the number of fairs before the fair date for each student and merge that with the student data.

For every row in the data, we want to lookup and count the number of fairs that the student attended before the career fair date for that row.

Process:

1. Get the cross product of the fair attendances and career fair dates to get a row for each attendance for each career fair.
2. Filter out the attendances that are after the career fair date to get the attendances that are before the career fair date.
3. Separate into main career fair and other career fair attendances. (main career fairs are the ones we are predicting)
4. Group by student id and career fair date and count the number of attendances before the career fair date.
5. Merge the counts with the data and fill null values with 0.
6. Convert the counts to binary values.

### 5. Event Attendance Cleaning

`stu_event_attendance_df` contains a row for each event for each student that attended that event. We want to count the number of events before the career fair date for each student and merge that with the student data.

Unlike the career fair attendance where we counted past attendance, we also want to consider the category of the event. Each event may have multiple categories.

Possible Event Categories:

- Academic
- Career fairs
- Conference
- Employers
- General
- Guidance
- Hiring
- Networking

We also want to consider specifically career fair prep sessions because they are likely to be more relevant to the career fair.

Process:

1. Get the cross product of the event attendances and career fair dates
2. Filter out the attendances that are after the career fair date
3. Separate categories into different dataframes
4. Separate career fair prep sessions into a separate dataframe
5. For each dataframe, group by student id and career fair date and count the number of attendances.
6. For the career fair prep session dataframe, add a boolean for whether that attendance/event is for that upcoming career fair (this can be determined as whether the event is within a month)
7. Merge the counts with the data and fill null values with 0.
8. Convert the counts to binary values.

### 6. Date Conversion

Date values are converted to boolean values based on certain thresholds like:

- How many years the account was created before or after the career fair
- How many days before the career fair the student logged into their Handshake account
- How many years until graduation relative to the career fair date

### 7. School Year Cleaning

Convert school year to a binary value. `stu_school_year` is a list of school years.

- Freshman
- Sophomore
- Junior
- Senior
- Alumni
- Masters
- Doctorate

### 8. College Cleaning

- Business
- Health
- Engineering
- Education
- Arts
- No College
- Other

### 9. Majors Cleaning

Since there are so many majors and colleges are already considered, the majors are cleaned by checking whether the student's major is included in the majors specified by employers at the career fair (i.e., whether the career fair includes the student's major)

### 10. Appointments Cleaning

Convert appointment list to whether or not the student has had certain appointment types and drop the original `appointment_types` column since we only need the binary values.

Appointment Types included: Walk-Ins, Resume Reviews, Career Fair Preparation, Career Exploration, Internship/Job Search, and Other.

### 11. GPA Cleaning

Convert to a range that the student's gpa falls in:

- 1.0 or below
- 1.0–1.5
- 1.5–2.0
- 2.0–2.5
- 2.5–3.0
- 3.0–3.5
- 3.5–4.0

---

## Model

I first tried out a decision tree model to predict whether or not a student will attend a career fair. The model was trained on this student data without any consideration of time dependencies or memory specific to that student. The model was then trained and tested on the same data to see how well it performed. I then used some hyperparameter optimizations to find the most optimal configuration.

After training the model in this way, I wanted to perform a practical test, so the most recent career fair data was removed from the training data and added to a validation data set. The model fit the training data fairly well (F1: 0.8), but struggled significantly more with the new unknown data (F1: 0.35).

I eventually decided to switch over to a random forest model after doing some research, because I learned that the use of multiple decision trees to make a prediction can help to solve the overfitting problem.

## Hyperparameters

I tried a lot of different hyperparameters to optimize the performance of the model, implementing a `GridSearchCV` to test out many options quickly. In the end I found that the following hyperparamters performed the best consistently

```
max_depth: None,
min_samples_leaf: 1,
min_samples_split: 8,
n_estimators: 3
```

With more time, I would like to try out a wider spread of hyperparameters. With how long it takes for the model to train, especially when multiple parameters are being tested at once, I didn't end up having a lot of time to try out a wide spread of parameters. There were many parameters that I did not end up trying anything other than default values because of this so there may be significant room for improvement given more time.

---

## Results

### Test Results

The model predicted student attendance for the 2024 Winter Career Fair with the following metrics:

- Mean Squared Error: `0.0071`
- Accuracy: `0.9929`
- F1 Score: `0.6281`
- Recall: `0.4875`
- Precision: `0.8824`

### Feature Importance Rankings

Analysis of feature importances revealed a few main insights:

- Pre-registration significantly increased attendance likelihood, highlighting the importance of promoting pre-registration to boost attendance rates
- Graduation timing and profile creation proximity to career fair date were significant predictors
- Higher engagement levels (indicated by application and past attendance counts), correlated with higher attendance likelihood
- GPA, work experience, and field of study were also influential factors to determining career fair attendance

### Comprehensive Feature Analysis

Below are the ranked feature importances. These show the weight that the model assigned to each feature.

Clearly, students who pre-register for the career are significantly more likely to attend. This stresses the importance of propoting pre-registration to boost attendance rates

The next most important predictor is whether the student graduated 4 or more years after the career fair. This makes sense because those alumni are probably significantly less likely to attend, so it becomes a good indicator

Next is the timing of the profile creation (ranks 3, 7, and 34). Profiles created closer to the career fair date are easier to predict. The direction of this is unclear — are students who created their account closer to the career fair date more likely or less likely to attend? This would be a good data point to look deeper into.

Students with more applications and past attendances (ranks 4, 8, 19, 23, 33) tend to be easier for the model to predict attendance. This suggests that the level of a students engagement is a good indicator for career fair attendance.

Higher GPA (ranks 10, 14, 30, 41, 47, 71) have higher importances according to the model. This could indicate that high-achieving students are more likely to attend career fairs. 

Students with work experience, especially multiple, are a good indicator for their attendance

The student's field of study is also a significant indicator. With business students having the highest importance (rank 5), engineering (rank 24), arts (rank 27), education (52), and health (54). This could give insight into what groups should be targeted more. Business and engineering students are already the biggest group that attends career fairs, so much so that a student being a business or engineering student makes it easier for the model to predict their attendance. Other groups are not so high, so this might indicate room for improvement.

There are many more takeaways that can come of these rankings and much more that this model could be used for within the scope of this project in the future.

```
Rank                                  Feature Importance
   1                       is_pre_registered    0.29719
   2                   grad_4+_years_post_cf    0.02111
   3                   created_1_year_pre_cf    0.01838
   4                    has_10+_applications    0.01731
   5                             is_business    0.01716
   6                     grad_1_year_post_cf    0.01636
   7                  created_2_years_pre_cf    0.01628
   8                       has_0_attendances    0.01446
   9                    grad_2_years_post_cf    0.01393
  10                             gpa_3.5-4.0    0.01336
  11         attended_0_past_guidance_events    0.01319
  12                  has_2_work_experiences    0.01299
  13                 has_3+_work_experiences    0.01249
  14                             gpa_3.0-3.5    0.01238
  15                 stu_is_profile_complete    0.01234
  16                               is_alumni    0.01211
  17                       stu_is_work_study    0.01210
  18           attended_1_other_fairs_before    0.01162
  19                    has_5-10_attendances    0.01161
  20             attended_1_main_fair_before    0.01141
  21           attended_0_other_fairs_before    0.01117
  22            attended_0_main_fairs_before    0.01105
  23                      has_1-2_attendance    0.01104
  24                          is_engineering    0.01094
  25                    has_1-5_appointments    0.01052
  26                    grad_3_years_post_cf    0.01016
  27                                 is_arts    0.01016
  28                 has_walk_in_appointment    0.01014
  29                  created_4_years_pre_cf    0.00990
  30                             gpa_2.5-3.0    0.00978
  31           has_resume_review_appointment    0.00972
  32                   has_1_work_experience    0.00967
  33                     has_10+_attendances    0.00943
  34                  created_3_years_pre_cf    0.00930
  35                   has_other_appointment    0.00925
  36                       has_10-100_logins    0.00889
  37                     login_7_days_pre_cf    0.00878
  38                  has_0_work_experiences    0.00868
  39                     has_3-5_attendances    0.00849
  40                      has_0_applications    0.00833
  41                   has_5-10_applications    0.00803
  42                         has_1-10_logins    0.00765
  43                               is_senior    0.00729
  44         attended_1_past_guidance_events    0.00728
  45                   has_5-10_appointments    0.00727
  46                      has_0_appointments    0.00724
  47                                  no_gpa    0.00714
  48                              is_masters    0.00689
  49                    has_1-5_applications    0.00663
  50                         has_100+_logins    0.00660
  51                  created_5_years_pre_cf    0.00630
  52                            is_education    0.00622
  53                 has_cf_prep_appointment    0.00619
  54                               is_health    0.00612
  55                      grad_1_year_pre_cf    0.00609
  56                               is_junior    0.00590
  57          attended_0_past_cf_prep_events    0.00576
  58         attended_2_past_guidance_events    0.00534
  59            attended_2_main_fairs_before    0.00503
  60              has_job_search_appointment    0.00496
  61                          stu_is_visible    0.00468
  62        attended_1_past_employers_events    0.00468
  63       attended_0_past_networking_events    0.00466
  64                              no_college    0.00447
  65       attended_1_past_networking_events    0.00445
  66        attended_0_past_employers_events    0.00442
  67           attended_2_other_fairs_before    0.00431
  68               attended_career_fair_prep    0.00401
  69                    login_30_days_pre_cf    0.00390
  70        attended_3+_past_guidance_events    0.00385
  71                             gpa_2.0-2.5    0.00381
  72                   login_90+_days_pre_cf    0.00374
  73                 has_learning_experience    0.00364
  74          attended_3+_other_fairs_before    0.00362
  75                            is_sophomore    0.00330
  76                            is_doctorate    0.00317
  77                    has_10+_appointments    0.00316
  78          attended_1_past_cf_prep_events    0.00316
  79      has_career_exploration_appointment    0.00309
  80         attended_1_past_academic_events    0.00290
  81         attended_0_past_academic_events    0.00287
  82                    grad_4+_years_pre_cf    0.00280
  83                           other_college    0.00242
  84                             is_freshman    0.00212
  85           attended_3+_main_fairs_before    0.00208
  86                         stu_is_archived    0.00208
  87                     grad_2_years_pre_cf    0.00166
  88                             gpa_1.5-2.0    0.00153
  89                            has_0_logins    0.00130
  90      attended_3+_past_networking_events    0.00129
  91                             gpa_1.0-1.5    0.00114
  92        attended_2_past_employers_events    0.00111
  93          attended_2_past_cf_prep_events    0.00106
  94       attended_2_past_networking_events    0.00087
  95           attended_1_past_hiring_events    0.00082
  96                    login_90_days_pre_cf    0.00081
  97           attended_0_past_hiring_events    0.00068
  98                     grad_3_years_pre_cf    0.00064
  99         attended_2_past_academic_events    0.00064
 100       attended_3+_past_employers_events    0.00058
 101                                 gpa_1.0    0.00053
 102                        stu_is_activated    0.00048
 103         attended_3+_past_cf_prep_events    0.00038
 104          attended_3+_past_hiring_events    0.00000
 105       attended_0_past_conference_events    0.00000
 108                       multiple_colleges    0.00000
 109       attended_2_past_conference_events    0.00000
 110        attended_3+_past_academic_events    0.00000
 113    attended_3+_past_career_fairs_events    0.00000    
 114     attended_1_past_career_fairs_events    0.00000    
 115     attended_0_past_career_fairs_events    0.00000    
 116         attended_3+_past_general_events    0.00000    
 117          attended_2_past_general_events    0.00000    
 118          attended_1_past_general_events    0.00000    
 119          attended_0_past_general_events    0.00000    
 120     attended_2_past_career_fairs_events    0.00000
```

---

## Conclusion

The Career Fair Attendance Classifier project successfully developed a predictive model to forecast student attendance at career fairs. By identifying critical determinants of attendance, such as pre-registration and engagement levels, the project provides actionable insights for enhancing future attendance rates and optimizing student engagement strategies.

## Future Directions

Potential future avenues include refining model hyperparameters further, trying out additional features, and integrating real-time data for continuous model improvements. 

Additionally, other types of models could be used to improve the shortcomings of Decision Tree and Random Forest. A Recurrent Neural Network could be tried out to capture the sequential/time-dependent relationship among data. RNNs work well with sequential data and could capture patterns in career fair attendance. By considering the sequential nature of events leading up to a career fair (appointments, career fair prep events, etc.). Taking this into account with a RNN could improve prediction accuracy. RNNs can also handle both long-term and short-term dependencies. Random Forest might struggle with this, but RNNs excel at accounting for these types of structures. The nature of student engagement data could fit well for an RNN model, but would require further investigation.

Finally, the model's insights can inform targeted action that the Oakland University Career and Life Design Center can take to encourage attendance among specific groups of students, improving the attendance of future career fairs.
