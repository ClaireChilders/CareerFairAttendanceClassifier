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

Logins/Year: `<5`, `5-10`, `10-15`, `15-20`, `20+`
Applications/Year: `<5`, `5-10`, `10-15`, `15-20`, `20-30`, `30-50`, `50+`
Events/Year: `1`, `2`, `3`, `4`, `5+`
Walk-Ins/Year: `0`, `1`, `2`, `3`, `4+`
Appointments/Year: `0`, `1`, `2`, `3`, `4+`
Years Attended: `0`, `1`, `2`, `3`, `4+`
Weeks Since Last Login: `<1`, `1-2`, `2-4`, `4-8`, `8-16`, `16+`
Years Until Graduation: `0`, `1`, `2`, `3`, `4+`
GPA: `0-1.0`, `1.0-1.5`, `1.5-2.0`, `2.0-2.5`, `2.5-3.0`, `3.0-3.5`, `3.5-4.0`

Then, certain categorical data must be converted to a format that can be used for analysis.

Majors: Each major is converted to a binary value indicating whether or not the career fair includes that major.

## Model

Currently using a decision tree model, but considering trying out a random forest model as well
