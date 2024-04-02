import pandas as pd
from sklearn.model_selection import train_test_split
from colorama import Fore, Style

# =============================================================================
#                           Data Preprocessing
# =============================================================================


def load_data():
    print(f'{Fore.MAGENTA}\nLoading data...{Style.RESET_ALL}')

    appointment_df = pd.read_csv('data/appointment_data.csv')
    career_fair_df = pd.read_csv('data/career_fair_data.csv')
    registration_df = pd.read_csv('data/registration_data.csv')
    student_df = pd.read_csv('data/student_data.csv')
    stu_counts_1_df = pd.read_csv('data/student_counts_1.csv')
    stu_counts_2_df = pd.read_csv('data/student_counts_2.csv')

    print(f'{Fore.GREEN}  ✓{Fore.LIGHTCYAN_EX} csv files loaded'
          f'{Style.RESET_ALL}')

    # ===============================================================
    #                          Merge Data
    # ===============================================================

    # There should be a row for each career fair for every student.
    #   If the student never attended a career fair, there will be
    #   a row with all null values for that student.
    #
    # The appointment_df only contains rows for students that have
    #   appointments. There can only be one row per student however,
    #   so we will just need to merge the appointment data with the
    #   student data to ensure that we have a row for each student.
    #
    # The registration_df contains a row for each student that has
    #   registered for each career fair. There may be multiple rows
    #   for each student. We want a row for every student for every
    #   career fair, so we will need to merge the registration data
    #   with the student data.

    merged_data = pd.merge(student_df, stu_counts_1_df,
                           on='stu_id', how='left')
    merged_data = pd.merge(merged_data, stu_counts_2_df,
                           on='stu_id', how='left')
    merged_data = pd.merge(merged_data, appointment_df,
                           on='stu_id', how='left')

    print(f'{Fore.GREEN}  ✓{Fore.LIGHTCYAN_EX} Student data merged'
          f'{Style.RESET_ALL}')

    # To ensure that we have a row for each student for each career fair,
    #   we get the cross product of the student ids and career fair dates
    #   and merge that with the merged data. This ensures there is a row
    #   for each student for each career fair.
    # Then, we merge the registration data with all that to add the
    #   registration columns to the merged data.

    # Merge career fair name and date to ensure that we have a unique
    #   identifier for each career fair (some have the same name but
    #   different dates)
    registration_df['career_fair_id'] = (
        registration_df['career_fair_name'] +
        ' ' +
        registration_df['career_fair_date']
    )
    career_fair_df['career_fair_id'] = (
        career_fair_df['career_fair_name'] +
        ' ' +
        career_fair_df['career_fair_date']
    )
    career_fair_simple = career_fair_df[[
        'career_fair_id',
        'career_fair_name',
        'career_fair_date'
    ]]

    # Drop the original columns, they will be merged back later
    registration_df.drop(
        columns=[
            'career_fair_name',
            'career_fair_date'
        ],
        inplace=True
    )

    print(f'{Fore.GREEN}  ✓{Fore.LIGHTCYAN_EX} Generated unique career fair '
          f'id\'s{Style.RESET_ALL}')

    # Merge student data with career fair data

    stu_fair_combinations = pd.merge(
        student_df[['stu_id']],
        pd.DataFrame(
            registration_df['career_fair_id'].unique(),
            columns=['career_fair_id']
        ),
        how='cross'
    )

    print(f'{Fore.GREEN}  ✓{Fore.LIGHTCYAN_EX} Generated student/career fair '
          f'combinations{Style.RESET_ALL}')

    merged_data = pd.merge(
        stu_fair_combinations, merged_data,
        on=['stu_id'],
        how='left'
    )
    print(f'{Fore.GREEN}  ✓{Fore.LIGHTCYAN_EX} Student/Career Fair data merged'
          f'{Style.RESET_ALL}')

    # Merge back in the career fair data

    merged_data = pd.merge(
        merged_data, career_fair_simple,
        on='career_fair_id',
        how='left'
    )

    print(f'{Fore.GREEN}  ✓{Fore.LIGHTCYAN_EX} Career Fair data merged'
          f'{Style.RESET_ALL}')

    # Merge the registration data
    merged_data = pd.merge(
        merged_data, registration_df,
        on=['stu_id', 'career_fair_id'],
        how='left'
    )
    print(f'{Fore.GREEN}  ✓{Fore.LIGHTCYAN_EX} Career Fair Registration '
          f'data merged'
          f'{Style.RESET_ALL}')

    print(f'{Fore.LIGHTBLACK_EX}  ⓘ {Fore.BLUE} Rows: '
          f'{Fore.LIGHTBLACK_EX}{len(merged_data)}'
          f'{Style.RESET_ALL}')
    print(f'{Fore.LIGHTBLACK_EX}  ⓘ {Fore.BLUE} Students: '
          f'{Fore.LIGHTBLACK_EX}{len(merged_data["stu_id"].unique())}'
          f'{Style.RESET_ALL}')

    merged_data.drop(
        [
            'appointment_count',
            'stu_id',
            'career_fair_id',
        ], axis=1, inplace=True)

    print(f'{Fore.LIGHTBLACK_EX}  ⓘ {Fore.BLUE} Columns: '
          f'{Fore.LIGHTBLACK_EX}{merged_data.columns}'
          f'{Style.RESET_ALL}')

    print(f'{Fore.GREEN}✓{Fore.MAGENTA} Data merged{Style.RESET_ALL}')

    return merged_data, career_fair_df


def clean_data(data, career_fair_df):
    print(f'{Fore.MAGENTA}\nCleaning data...{Style.RESET_ALL}')

    yes_no_columns = [
        'is_pre_registered',
        'is_checked_in',
        'stu_is_activated',
        'stu_is_visible',
        'stu_is_archived',
        'stu_is_work_study',
        'stu_is_profile_complete'
    ]

    count_columns = [
        'stu_appointments',
        'stu_applications',
        'stu_attendances',
        'stu_work_experiences',
        'stu_experiences',
        'stu_logins'
    ]

    # ===============================================================
    #                          Null Values
    # ===============================================================

    print(f'{Fore.LIGHTBLACK_EX}  → {Fore.BLUE}Filling null values...'
          f'{Style.RESET_ALL}')

    for column in yes_no_columns:
        data[column] = data[column].fillna('No')

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Null Yes/No values filled '
          f'with \'No\'{Style.RESET_ALL}')

    for column in count_columns:
        data[column] = data[column].fillna(0)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Null count values filled '
          f'with 0{Style.RESET_ALL}')

    data['stu_colleges'] = data['stu_colleges'].fillna('No College Designated')

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} All null values filled'
          f'{Style.RESET_ALL}')

    # Convert Yes/No to 1/0

    print(f'{Fore.LIGHTBLACK_EX}  → {Fore.BLUE}Converting Yes/No values to '
          f'1/0...{Style.RESET_ALL}')

    for column in yes_no_columns:
        data[column] = data[column].apply(
            lambda x: 1 if x == 'Yes' else 0)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Yes/No converted to 1/0'
          f'{Style.RESET_ALL}')

    # ===============================================================
    #                       Integer Conversion
    # ===============================================================

    # Convert any strings in the form '1,000' to integers
    #   while keeping existing integers

    print(f'{Fore.LIGHTBLACK_EX}  → {Fore.BLUE}Converting strings to binary '
          f'values...{Style.RESET_ALL}')

    for column in count_columns:
        data[column] = data[column].apply(
            lambda x: int(x.replace(',', '')) if isinstance(x, str) else x)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Strings converted to integers'
          f'{Style.RESET_ALL}')

    # Convert integers to binary thresholds values
    #   and drop the original columns

    data['has_appointment'] = data['stu_appointments'].apply(
        lambda x: 1 if x > 0 else 0)
    data['has_5_appointments'] = data['stu_appointments'].apply(
        lambda x: 1 if x >= 5 else 0)
    data['has_10_appointments'] = data['stu_appointments'].apply(
        lambda x: 1 if x >= 10 else 0)
    data['has_application'] = data['stu_applications'].apply(
        lambda x: 1 if x > 0 else 0)
    data['has_5_applications'] = data['stu_applications'].apply(
        lambda x: 1 if x >= 5 else 0)
    data['has_10_applications'] = data['stu_applications'].apply(
        lambda x: 1 if x >= 10 else 0)
    data['has_login'] = data['stu_logins'].apply(
        lambda x: 1 if x > 0 else 0)
    data['has_10_logins'] = data['stu_logins'].apply(
        lambda x: 1 if x >= 10 else 0)
    data['has_100_logins'] = data['stu_logins'].apply(
        lambda x: 1 if x >= 100 else 0)
    data['has_attendance'] = data['stu_attendances'].apply(
        lambda x: 1 if x > 0 else 0)
    data['has_5_attendances'] = data['stu_attendances'].apply(
        lambda x: 1 if x >= 5 else 0)
    data['has_10_attendances'] = data['stu_attendances'].apply(
        lambda x: 1 if x >= 10 else 0)
    data['has_work_experience'] = data['stu_work_experiences'].apply(
        lambda x: 1 if x > 0 else 0)
    data['has_2_work_experiences'] = data['stu_work_experiences'].apply(
        lambda x: 1 if x >= 2 else 0)
    data['has_3_work_experiences'] = data['stu_work_experiences'].apply(
        lambda x: 1 if x >= 3 else 0)
    data['has_experience'] = data['stu_experiences'].apply(
        lambda x: 1 if x > 0 else 0)
    data['has_2_experiences'] = data['stu_experiences'].apply(
        lambda x: 1 if x >= 2 else 0)
    data['has_3_experiences'] = data['stu_experiences'].apply(
        lambda x: 1 if x >= 3 else 0)

    data.drop(count_columns, axis=1, inplace=True)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Integers converted to binary '
          f'values{Style.RESET_ALL}')
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} All strings converted to '
          f'binary values{Style.RESET_ALL}')

    # ===============================================================
    #                      Date Conversion
    # ===============================================================

    print(f'{Fore.LIGHTBLACK_EX}  → {Fore.BLUE}Converting dates to binary '
          f'values...{Style.RESET_ALL}')

    # Date between stu_creation_date and career_fair_date
    data['days_since_created'] = (pd.to_datetime(data['career_fair_date']) -
                                  pd.to_datetime(data['stu_creation_date'])
                                  ).dt.days
    data['created_1_year_pre_cf'] = data['days_since_created'].apply(
        lambda x: 1 if x <= 365 else 0)
    data['created_2_years_pre_cf'] = data['days_since_created'].apply(
        lambda x: 1 if x <= 730 else 0)
    data['created_3_years_pre_cf'] = data['days_since_created'].apply(
        lambda x: 1 if x <= 1095 else 0)
    data['created_4_years_pre_cf'] = data['days_since_created'].apply(
        lambda x: 1 if x <= 1825 else 0)
    data['created_5_years_pre_cf'] = data['days_since_created'].apply(
        lambda x: 1 if x > 1825 else 0)

    data.drop(['stu_creation_date', 'days_since_created'],
              axis=1, inplace=True)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Creation date converted to '
          f'binary values{Style.RESET_ALL}')

    # Date between stu_login_date and career_fair_date
    data['days_since_login'] = (pd.to_datetime(data['career_fair_date']) -
                                pd.to_datetime(data['stu_login_date'])
                                ).dt.days
    data['login_7_days_pre_cf'] = data['days_since_login'].apply(
        lambda x: 1 if x <= 7 else 0)
    data['login_30_days_pre_cf'] = data['days_since_login'].apply(
        lambda x: 1 if x <= 30 else 0)
    data['login_90_days_pre_cf'] = data['days_since_login'].apply(
        lambda x: 1 if x <= 90 else 0)
    data['login_90+_days_pre_cf'] = data['days_since_login'].apply(
        lambda x: 1 if x > 90 else 0)

    data.drop(['stu_login_date', 'days_since_login'], axis=1, inplace=True)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Login date converted to '
          f'binary values{Style.RESET_ALL}')

    # Date between stu_grad_date and career_fair_date
    data['days_until_grad'] = (pd.to_datetime(data['stu_grad_date']) -
                               pd.to_datetime(data['career_fair_date'])
                               ).dt.days

    data['grad_1_year_post_cf'] = data['days_until_grad'].apply(
        lambda x: 1 if x <= 365 else 0)
    data['grad_2_years_post_cf'] = data['days_until_grad'].apply(
        lambda x: 1 if x <= 730 else 0)
    data['grad_3_years_post_cf'] = data['days_until_grad'].apply(
        lambda x: 1 if x <= 1095 else 0)
    data['grad_4_years_post_cf'] = data['days_until_grad'].apply(
        lambda x: 1 if x > 1095 else 0)

    data.drop(['days_until_grad'], axis=1, inplace=True)

    # data.drop(['career_fair_date'], axis=1, inplace=True)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Graduation date converted to '
          f'binary values{Style.RESET_ALL}')

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} All dates converted to '
          f'binary values{Style.RESET_ALL}')

    # ===============================================================
    #                       School Year Cleaning
    # ===============================================================

    # Convert school year to a binary value. stu_school_year is a list
    #   of school years

    print(f'{Fore.LIGHTBLACK_EX}  → {Fore.BLUE}Extracting student school '
          f'years...{Style.RESET_ALL}')

    data['stu_school_year'] = data['stu_school_year'].apply(
        lambda x: x.split(',') if isinstance(x, str) else x
    )
    data['stu_school_year'] = data['stu_school_year'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    data['is_freshman'] = data['stu_school_year'].apply(
        lambda x: 1 if 'Freshman' in x else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted Freshmen '
          f'{Style.RESET_ALL}')
    data['is_sophomore'] = data['stu_school_year'].apply(
        lambda x: 1 if 'Sophomore' in x else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted Sophomores '
          f'{Style.RESET_ALL}')
    data['is_junior'] = data['stu_school_year'].apply(
        lambda x: 1 if 'Junior' in x else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted Juniors '
          f'{Style.RESET_ALL}')
    data['is_senior'] = data['stu_school_year'].apply(
        lambda x: 1 if 'Senior' in x else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted Seniors '
          f'{Style.RESET_ALL}')
    data['is_alumni'] = data['stu_school_year'].apply(
        lambda x: 1 if 'Alumni' in x else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted Alumni '
          f'{Style.RESET_ALL}')
    data['is_masters'] = data['stu_school_year'].apply(
        lambda x: 1 if 'Masters' in x else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted Masters Students '
          f'{Style.RESET_ALL}')
    data['is_doctorate'] = data['stu_school_year'].apply(
        lambda x: 1 if 'Doctorate' in x else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted Doctoral Students '
          f'{Style.RESET_ALL}')

    data.drop(['stu_school_year'], axis=1, inplace=True)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} All School years converted '
          f'to binary values{Style.RESET_ALL}')

    # ===============================================================
    #                       College Cleaning
    # ===============================================================

    print(f'{Fore.LIGHTBLACK_EX}  → {Fore.BLUE}Extracting student '
          f'colleges...{Style.RESET_ALL}')

    # Convert colleges to a list of colleges

    data['stu_colleges'] = data['stu_colleges'].apply(
        lambda x: x.split(',') if isinstance(x, str) else x
    )
    data['stu_colleges'] = data['stu_colleges'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Convert college to a binary value

    business = ['School of Business Admin.']
    health = ['School of Health Sciences']
    engineering = ['School of Egr. and Comp. Sci.',
                   'Arts & Sci and School of Egr', ]
    education = ['School of Ed. and Human Svcs.']
    arts = ['College of Arts and Sciences']
    no_college = ['No College Designated',
                  'University Programs', 'All Colleges']

    all_colleges = (business + health + engineering +
                    education + arts + no_college)

    data['is_engineering'] = data['stu_colleges'].apply(
        lambda x: 1 if any([college in engineering for college in x]) else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted engineering '
          f'binary values{Style.RESET_ALL}')
    data['is_business'] = data['stu_colleges'].apply(
        lambda x: 1 if any([college in business for college in x]) else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted business '
          f'binary values{Style.RESET_ALL}')
    data['is_health'] = data['stu_colleges'].apply(
        lambda x: 1 if any([college in health for college in x]) else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted health '
          f'binary values{Style.RESET_ALL}')
    data['is_education'] = data['stu_colleges'].apply(
        lambda x: 1 if any([college in education for college in x]) else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted education '
          f'binary values{Style.RESET_ALL}')
    data['is_arts'] = data['stu_colleges'].apply(
        lambda x: 1 if any([college in arts for college in x]) else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted arts '
          f'binary values{Style.RESET_ALL}')
    data['no_college'] = data['stu_colleges'].apply(
        lambda x: 1 if any([college in no_college for college in x]) else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted no college '
          f'binary values{Style.RESET_ALL}')
    data['multiple_colleges'] = data['stu_colleges'].apply(
        lambda x: 1 if len(x) > 1 else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted multiple colleges '
          f'binary values{Style.RESET_ALL}')
    data['other_college'] = data['stu_colleges'].apply(
        lambda x: 0 if any(college in all_colleges for college in x) else 1)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted other colleges '
          f'binary values{Style.RESET_ALL}')

    data.drop(['stu_colleges'], axis=1, inplace=True)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} All colleges converted to '
          f'binary values{Style.RESET_ALL}')

    # ===============================================================
    #                        Majors Cleaning
    # ===============================================================

    # Add a column for whether the student has a major that is
    #   included in the career fair they attended

    print(f'{Fore.LIGHTBLACK_EX}  → {Fore.BLUE}Extracting student '
          f'majors...{Style.RESET_ALL}')

    # First ensure that stu_majors only contains lists

    data['stu_majors'] = data['stu_majors'].apply(
        lambda x: x.split(',') if isinstance(x, str) else x
    )
    data['stu_majors'] = data['stu_majors'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Check if any of the student's majors are included in the career fair's
    data['cf_has_major'] = data['stu_majors'].apply(
        lambda x: 1 if any(
            [major in career_fair_df['majors'].values for major in x]) else 0
    )

    data.drop(['stu_majors'], axis=1, inplace=True)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} All majors converted to '
          f'binary values{Style.RESET_ALL}')

    # ===============================================================
    #                       Appointments Cleaning
    # ===============================================================

    # Convert appointment list to whether or not the student has had
    #   certain appointment types and drop the original appointment_types
    #   column since we only need the binary values.
    #
    # Appointment Types included: Walk-Ins, Resume Reviews, Career Fair
    #   Preparation, Career Exploration, Internship/Job Search, and Other.

    print(f'{Fore.LIGHTBLACK_EX}  → {Fore.BLUE}Extracting appointment '
          f'types...{Style.RESET_ALL}')

    # First ensure that appointment_types only contains lists of
    #   lowercase strings

    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: x.lower().split(',') if isinstance(x, str) else x
    )
    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Walk-Ins
    data['has_walk_in'] = data['appointment_types'].apply(
        lambda x: 1 if any('walk-in' in i for i in x) else 0
    )
    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: [i for i in x if 'walk-in' not in i]
    )

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Walk-Ins extracted'
          f'{Style.RESET_ALL}')

    # Resume Reviews
    data['has_resume_review'] = data['appointment_types'].apply(
        lambda x: 1 if any('resume' in i for i in x) else 0
    )
    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: [i for i in x if 'resume' not in i]
    )

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Resume Reviews extracted'
          f'{Style.RESET_ALL}')

    # Career Fair Prep
    data['has_cf_prep'] = data['appointment_types'].apply(
        lambda x: 1 if any('career fair' in i for i in x) else 0
    )
    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: [i for i in x if 'career fair' not in i]
    )

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Career Fair Prep extracted'
          f'{Style.RESET_ALL}')

    # Career Exploration
    data['has_career_exploration'] = data['appointment_types'].apply(
        lambda x: 1 if any('career exploration' in i for i in x) else 0
    )
    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: [i for i in x if 'career exploration' not in i]
    )

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Career Exploration extracted'
          f'{Style.RESET_ALL}')

    # Internship/Job Search
    data['has_job_search'] = data['appointment_types'].apply(
        lambda x: 1 if any('internship' in i or 'job' in i for i in x) else 0
    )
    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: [i for i in x if 'internship' not in i and 'job' not in i]
    )

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Internship/Job Search '
          f'extracted{Style.RESET_ALL}')

    # Other
    data['has_other'] = data['appointment_types'].apply(
        lambda x: 1 if len(x) > 0 else 0
    )

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Other extracted'
          f'{Style.RESET_ALL}')

    # Drop appointment_types since we've extracted the binary values
    data.drop(['appointment_types'], axis=1, inplace=True)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Appointments cleaned'
          f'{Style.RESET_ALL}')

    print(f'{Fore.GREEN}✓{Fore.MAGENTA} Data cleaned{Style.RESET_ALL}')

    print(f'{Fore.LIGHTBLACK_EX}  ⓘ {Fore.BLUE} Data: '
          f'{Fore.LIGHTBLACK_EX}{len(data)}{Style.RESET_ALL}')
    print(f'{Fore.LIGHTBLACK_EX}  ⓘ {Fore.BLUE} Features: '
          f'{Fore.LIGHTBLACK_EX}{len(data.columns) - 1}{Style.RESET_ALL}')

    return data


def extract_features_target(data):
    print(f'{Fore.MAGENTA}\n  Extracting features and target...'
          f'{Style.RESET_ALL}')

    features = data.drop(
        [
            'is_checked_in',
        ],
        axis=1
    )

    target = data['is_checked_in']

    print(f'{Fore.GREEN}  ✓{Fore.LIGHTCYAN_EX} '
          f'Features and target extracted{Style.RESET_ALL}')

    return features, target


def split_data(features, target, test_size):
    print(f'{Fore.MAGENTA}\n  Splitting data...{Style.RESET_ALL}')
    print(f'{Fore.BLUE}    Test size: {Fore.CYAN}{test_size}{Style.RESET_ALL}')

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=0)

    print(f'{Fore.GREEN}  ✓{Fore.LIGHTCYAN_EX} Data split{Style.RESET_ALL}')

    return x_train, x_test, y_train, y_test


def align_features(train_data: pd.DataFrame, test_data: pd.DataFrame):
    all_features = set(train_data.columns) | set(test_data.columns)
    print(f'{Fore.MAGENTA}\n  Aligning {len(all_features)} features '
          f'between train and test data...{Style.RESET_ALL}')

    train_missing: set = all_features - set(train_data.columns)
    test_missing: set = all_features - set(test_data.columns)
    count = len(train_missing) + len(test_missing)

    if len(train_missing) > 0:
        print(f'  {Fore.LIGHTYELLOW_EX}{len(train_missing)}{Fore.RED} '
              f'Features missing from training data{Style.RESET_ALL}')
        train_data = train_data.reindex(columns=all_features, fill_value=0)
    if len(test_missing) > 0:
        print(f'  {Fore.LIGHTYELLOW_EX}{len(test_missing)}{Fore.RED} '
              f'Features missing from testing data{Style.RESET_ALL}')
        test_data = test_data.reindex(columns=all_features, fill_value=0)

    # Ensure the column are in the same order
    test_data = test_data[train_data.columns]

    print(f'{Fore.GREEN}  ✓{Fore.CYAN} {count}{Fore.LIGHTCYAN_EX} Features '
          f'aligned{Style.RESET_ALL}')

    return train_data, test_data


def split_practical_data(data: pd.DataFrame, test_career_fair_name: str):
    print(f'{Fore.BLUE}  Splitting test and training data by '
          f'{Fore.CYAN}{test_career_fair_name}{Style.RESET_ALL}')

    data.drop(
        columns=['career_fair_date', 'stu_grad_date'],
        axis=1,
        inplace=True
    )

    training_data = data[data['career_fair_name'] != test_career_fair_name]
    testing_data = data[data['career_fair_name'] == test_career_fair_name]

    training_data = training_data.drop(columns=['career_fair_name'], axis=1)
    testing_data = testing_data.drop(columns=['career_fair_name'], axis=1)

    training_data, testing_data = align_features(training_data, testing_data)

    print(f'{Fore.GREEN}  ✓{Fore.LIGHTCYAN_EX} Data successfully split into '
          f'testing and training data by '
          f'{Fore.CYAN}{test_career_fair_name}{Style.RESET_ALL}')

    return training_data, testing_data


def get_practical_test(
    data: pd.DataFrame,
    test_career_fair_name: str,
    test_size: float
):
    print(f'{Fore.MAGENTA}\nPreparing practical test data...{Style.RESET_ALL}')

    training_data, testing_data = split_practical_data(
        data, test_career_fair_name)

    features, target = extract_features_target(training_data)
    x_practical_test, y_practical_test = extract_features_target(testing_data)

    features, x_practical_test = align_features(features, x_practical_test)

    x_train, x_test, y_train, y_test = split_data(
        features, target, test_size)

    print(f'{Fore.GREEN}✓{Fore.MAGENTA} Practical test data prepared'
          f'{Style.RESET_ALL}')
    print(f'{Fore.LIGHTBLACK_EX}  ⓘ {Fore.BLUE} Training data: '
          f'{Fore.LIGHTBLACK_EX}{len(x_train)}{Style.RESET_ALL}')
    print(f'{Fore.LIGHTBLACK_EX}  ⓘ {Fore.BLUE} Testing data: '
          f'{Fore.LIGHTBLACK_EX}{len(x_test)}{Style.RESET_ALL}')
    print(f'{Fore.LIGHTBLACK_EX}  ⓘ {Fore.BLUE} Practical test data: '
          f'{Fore.LIGHTBLACK_EX}{len(x_practical_test)}{Style.RESET_ALL}')
    print(f'{Fore.LIGHTBLACK_EX}  ⓘ {Fore.BLUE} Features: '
          f'{Fore.LIGHTBLACK_EX}{len(features.columns)}{Style.RESET_ALL}')

    return x_train, x_test, y_train, y_test, x_practical_test, y_practical_test
