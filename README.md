# CareerFairAttendanceClassifier

## Purpose / Goal

This project is designed to predict whether or not a student will attend a career fair based on their student data. The goal is to use this information to help the Career and Life Design Center plan for predicted attendance numbers and better target students who are less likely to attend career fairs and encourage them to attend.

## Method

To predict whether or not a student will attend a career fair, a machine learning model is trained on student data, which is then used to predict whether a student will attend a career fair given what is known about them.

## Data

The data used for this project is gathered from 5 datasets:

1. Basic Student Data
   This dataset contains basic information about students including the total number of login sessions, the total number of applications submitted, and the total number of events attended.
2. Student Appointment Data
   This dataset contains the total number of appointments each student has completed and a list of all types of appointments for each student.
3. Career Fair Attendance Data
   This dataset contains every career fair each student has attended, including their graduation date, major, college, school year, and gpa at the time of the career fair.
4. Career Fair Registration Data
   This dataset contains whether or not each student registered for each career fair and whether or not they attended.
5. Career Fair Data
   This dataset contains each career fair including the industries and majors that jobs are available for.

### Data Cleaning

The data must be cleaned before it can be used for analysis. This includes handling any null values, removing any duplicate values, and merging the datasets together.

In order to merge the datasets together, certain data needs to be converted to a format that decisions can be made on. For example, the career fair majors data is in a list format while each student has one major. What matters is whether the student's major is included or not in the list of majors for each career fair. So, the career fair majors must be referenced in a way that can be compared to the student's major for the data to be useful.

### Converting Data to a Usable Format

First, certain numerical data must be converted to ranges to allow a categorical decision to be made.

- Logins/Year: `<5`, `5-10`, `10-15`, `15-20`, `20+`
- Applications/Year: `<5`, `5-10`, `10-15`, `15-20`, `20-30`, `30-50`, `50+`
- Events/Year: `1`, `2`, `3`, `4`, `5+`
- Walk-Ins/Year: `0`, `1`, `2`, `3`, `4+`
- Appointments/Year: `0`, `1`, `2`, `3`, `4+`
- Years Attended: `0`, `1`, `2`, `3`, `4+`
- Weeks Since Last Login: `<1`, `1-2`, `2-4`, `4-8`, `8-16`, `16+`
- Years Until Graduation: `0`, `1`, `2`, `3`, `4+`
- GPA: `0-1.0`, `1.0-1.5`, `1.5-2.0`, `2.0-2.5`, `2.5-3.0`, `3.0-3.5`, `3.5-4.0`

Then, certain categorical data must be converted to a format that can be used for analysis.

Majors: Each major is converted to a binary value indicating whether or not the career fair includes that major.

## Model

Currently using a decision tree model, but considering trying out a random forest model as well

## First Implementation

The first implementation of the model uses the basic student data to predict whether or not a student will attend a career fair. The model is trained on this student data without any consideration for time dependencies or information specific to that student. The model is then tested on the same data to see how well it performs.

### Data Cleaning

- Yes/No values are converted to 1/0
- Null values are filled with 0
- Strings are converted to numerical values
- Dates converted to years
- Categorical data is converted to dummy variables
- Data is split into training and testing sets (test size = 0.2)

### Hyperparamters

- Entropy is used as the criterion
- The maximum depth of the tree is 5
- The minimum number of samples required to split an internal node is 2
- The minimum number of samples required to be at a leaf node is 1
- No minimum weight fraction required to be a leaf node
- No minimum impurity decrease required to split an internal node
- No maximum leaf nodes
- No maximum features
- All classes are weighted equally
- Choosing the best split

### Results

![](https://i.imgur.com/05GEzum.png)

In 5 tests of training the model, the average metrics are as follows:

- Mean Squared Error: 0.0411
- Accuracy: 0.9589
- F1 Score: 0.8028
- Recall: 0.8095
- Precision: 0.7963

The model has a high accuracy and low mean squared error, but the F1 score is lower than desired. This is likely due to the class imbalance in the data. The model is predicting that most students will not attend a career fair, which is accurate, but it is not predicting that many students will attend a career fair, which is not accurate. This is likely due to the fact that the model is not taking into account any information specific to the student or any time dependencies in the data.

## Second Implementation

In the first implementation we used the entire data set to both train and test the model. This is not realistic because in practice we would not have access to the data for the career fair we are trying to predict attendance for. In the second implementation we will split the data up into two separate test data sets, one for evaluating the model and one for testing the model with new data. We do this to simulate a real world scenario to see if the model can apply to new data.

### Issues

Since the data is split into two separate data sets the features of the training and testing sets will be different. For example, with categorical data like majors if a major appears in the training dataset but not in the testing dataset (i.e., there was no student in winter 2024 with that same major), there will be a feature mismatch which causes an error.

To overcome this, feature alignment had to be implemented. This goes through each feature of the two datasets and adds any missing from the other so the features are the same and can be compared.

### Data Cleaning

- The data is split into two separate data sets, one for training and one for practical testing
- The training data contains all career fair attendance data up to the 2024 winter career fair
- The practical testing data contains the career fair attendance data for the 2024 winter career fair
- The training and testing data are cleaned in the same way as the first implementation, then feature aligned
- The training data is split into training and testing sets (test size = 0.2)

### Hyperparameters

- The hyperparameters are the same as the first implementation

### Results

![](https://i.imgur.com/xovOkSg.png)

As we can see, the results with the training data are very similar to the first implementation. However, the model did not fit very well for the new data. With an accuracy of 0.5604, this is only slightly better than random chance.

This could be due to the fact that the model is not taking into account any information specific to the student or any time dependencies in the data.

## Third Implementation

## Fourth Implementation

So far, we still see that the model is not fitting well for new data. In other words, we have a high variance problem where the model is overfitting the training data. One way to reduce overfitting is to reduce the number of features. Currently, the dataset ends up having a VERY large number of features. Each unique categorical value is its own feature, which adds up to over 4000 features. So, we will try to reduce the number of features by grouping certain categorical values together to be represented by a single feature.

I am also considering trying out a random forest model instead of a decision tree after some research. This would allow the use of multiple decision trees to make a prediction, which might help to reduce overfitting.

## Fifth Implementation

After trying out many options to reduce the overfitting of the decision tree model, I decided to try out a random forest model instead. This model uses multiple decision trees to make a prediction, which can help to reduce overfitting.

With this model however, I jumped right into hyperparameter tuning to ensure the model was as accurate as possible. I used a grid search to try to find the best hyperparameters for the model. The reason I did this immediately was to ensure the model did in fact perform better than the decision tree model before committing more effort into it.

The hyperparameters tuned were:

- `max_depth`: [None, 10, 100, 1000]
- `min_samples_leaf`: [1, 2, 4]
- `min_samples_split`: [2, 5, 10]
- `n_estimators`: [10, 100, 1000]

### Results

The best hyperparameters found were:

- Max Depth: None
- Min Samples Leaf: 1
- Min Samples Split: 5
- N Estimators: 1000

Using a model with these hyperparameters, the results were:
