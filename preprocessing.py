import pandas as pd
from sklearn.model_selection import train_test_split
from colorama import Fore, Style

# =============================================================================
#                           Data Preprocessing
# =============================================================================


def load_data():
    print(Fore.MAGENTA + "\nLoading data..." + Style.RESET_ALL)

    appointment_df = pd.read_csv('data/appointment_data.csv')
    attendance_df = pd.read_csv('data/attendance_data.csv')
    career_fair_df = pd.read_csv('data/career_fair_data.csv')
    registration_df = pd.read_csv('data/registration_data.csv')
    student_df = pd.read_csv('data/student_data.csv')

    print(Fore.GREEN + "Data loaded" + Style.RESET_ALL)

    merged_data = pd.merge(student_df, registration_df,
                           on='stu_id', how='left')
    merged_data = pd.merge(merged_data, appointment_df,
                           on='stu_id', how='left')
    merged_data = pd.merge(
        merged_data,
        attendance_df,
        on=[
            'stu_id', 'career_fair_name', 'career_fair_date'
        ],
        how='left'
    )

    print(Fore.GREEN + "Data merged" + Style.RESET_ALL)

    return merged_data, career_fair_df


def clean_data(data, career_fair_df):
    print(Fore.MAGENTA + "\nCleaning data..." + Style.RESET_ALL)

    # Convert Yes/No to 1/0

    data['is_checked_in'] = data['is_checked_in'].apply(
        lambda x: 1 if x == 'Yes' else 0)
    data['is_pre_registered'] = data['is_pre_registered'].apply(
        lambda x: 1 if x == 'Yes' else 0)
    data['stu_is_activated'] = data['stu_is_activated'].apply(
        lambda x: 1 if x == 'Yes' else 0)
    data['stu_is_visible'] = data['stu_is_visible'].apply(
        lambda x: 1 if x == 'Yes' else 0)

    print(Fore.BLUE + "  Yes/No converted to 1/0" + Style.RESET_ALL)

    # ===============================================================
    #                       Integer Conversion
    # ===============================================================

    # Convert any strings in the form '1,000' to integers
    # while keeping existing integers
    data['appointment_count'] = data['appointment_count'].apply(
        lambda x: int(x.replace(',', '')) if isinstance(x, str) else x)
    data['stu_applications'] = data['stu_applications'].apply(
        lambda x: int(x.replace(',', '')) if isinstance(x, str) else x)
    data['stu_logins'] = data['stu_logins'].apply(
        lambda x: int(x.replace(',', '')) if isinstance(x, str) else x)
    data['stu_attendances'] = data['stu_attendances'].apply(
        lambda x: int(x.replace(',', '')) if isinstance(x, str) else x)

    print(Fore.BLUE + "  Strings converted to integers" + Style.RESET_ALL)

    # Convert integers to binary thresholds values
    #   and drop the original columns
    data['has_appointment'] = data['appointment_count'].apply(
        lambda x: 1 if x > 0 else 0)
    data['has_5_appointments'] = data['appointment_count'].apply(
        lambda x: 1 if x >= 5 else 0)
    data['has_10_appointments'] = data['appointment_count'].apply(
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

    data.drop([
        'appointment_count',
        'stu_applications',
        'stu_logins',
        'stu_attendances'
    ], axis=1, inplace=True)

    print(f"{Fore.BLUE}  Integers converted to binary values{Style.RESET_ALL}")

    # ===============================================================
    #                      Date Conversion
    # ===============================================================

    # Date between stu_created and career_fair_date
    data['days_since_created'] = (pd.to_datetime(data['career_fair_date']) -
                                  pd.to_datetime(data['stu_created'])).dt.days
    data['created_1_year_pre_cf'] = data['days_since_created'] <= 365
    data['created_2_years_pre_cf'] = data['days_since_created'] <= 730
    data['created_3_years_pre_cf'] = data['days_since_created'] <= 1095
    data['created_4_years_pre_cf'] = data['days_since_created'] > 1095

    data.drop(['stu_created', 'days_since_created'], axis=1, inplace=True)

    # Date between stu_login and career_fair_date
    data['days_since_login'] = (pd.to_datetime(data['career_fair_date']) -
                                pd.to_datetime(data['stu_login'])).dt.days
    data['login_7_days_pre_cf'] = data['days_since_login'] <= 7
    data['login_30_days_pre_cf'] = data['days_since_login'] <= 30
    data['login_90_days_pre_cf'] = data['days_since_login'] <= 90
    data['login_90+_days_pre_cf'] = data['days_since_login'] > 90

    data.drop(['stu_login', 'days_since_login'], axis=1, inplace=True)

    # Date between stu_grad_date and career_fair_date
    data['days_until_grad'] = (pd.to_datetime(data['stu_grad_date']) -
                               pd.to_datetime(data['career_fair_date'])
                               ).dt.days

    data['grad_1_year_post_cf'] = data['days_until_grad'] <= 365
    data['grad_2_years_post_cf'] = data['days_until_grad'] <= 730
    data['grad_3_years_post_cf'] = data['days_until_grad'] <= 1095
    data['grad_4_years_post_cf'] = data['days_until_grad'] > 1095

    data.drop(['stu_grad_date', 'days_until_grad'], axis=1, inplace=True)

    print(Fore.BLUE + "  Dates converted to binary values" + Style.RESET_ALL)

    # ===============================================================
    #                       School Year Cleaning
    # ===============================================================

    # Convert school year to a binary value

    data['is_freshman'] = data['stu_school_year'].apply(
        lambda x: 1 if x == 'Freshman' else 0)
    data['is_sophomore'] = data['stu_school_year'].apply(
        lambda x: 1 if x == 'Sophomore' else 0)
    data['is_junior'] = data['stu_school_year'].apply(
        lambda x: 1 if x == 'Junior' else 0)
    data['is_senior'] = data['stu_school_year'].apply(
        lambda x: 1 if x == 'Senior' else 0)
    data['is_alumni'] = data['stu_school_year'].apply(
        lambda x: 1 if x == 'Alumni' else 0)
    data['is_masters'] = data['stu_school_year'].apply(
        lambda x: 1 if x == 'Masters' else 0)
    data['is_doctorate'] = data['stu_school_year'].apply(
        lambda x: 1 if x == 'Doctorate' else 0)

    data.drop(['stu_school_year'], axis=1, inplace=True)

    print(f"{Fore.BLUE}  School year converted to binary values"
          f"{Style.RESET_ALL}")

    # ===============================================================
    #                       College Cleaning
    # ===============================================================

    # convert colleges to a list of colleges

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
    data['is_business'] = data['stu_colleges'].apply(
        lambda x: 1 if any([college in business for college in x]) else 0)
    data['is_health'] = data['stu_colleges'].apply(
        lambda x: 1 if any([college in health for college in x]) else 0)
    data['is_education'] = data['stu_colleges'].apply(
        lambda x: 1 if any([college in education for college in x]) else 0)
    data['is_arts'] = data['stu_colleges'].apply(
        lambda x: 1 if any([college in arts for college in x]) else 0)
    data['no_college'] = data['stu_colleges'].apply(
        lambda x: 1 if any([college in no_college for college in x]) else 0)
    data['multiple_colleges'] = data['stu_colleges'].apply(
        lambda x: 1 if len(x) > 1 else 0)
    data['other_college'] = data['stu_colleges'].apply(
        lambda x: 0 if any(college in all_colleges for college in x) else 1)

    data.drop(['stu_colleges'], axis=1, inplace=True)

    print(f"{Fore.BLUE}  Colleges converted to binary values{Style.RESET_ALL}")

    # ===============================================================
    #                        Majors Cleaning
    # ===============================================================

    # Add a column for whether the student has a major that is
    #   included in the career fair they attended

    print(Fore.BLUE + "  Extracting student majors..." + Style.RESET_ALL)

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

    print(Fore.GREEN + "  Majors converted to binary values" + Style.RESET_ALL)

    # ===============================================================
    #                       Appointments Cleaning
    # ===============================================================

    # Convert appointment list to whether or not the student has had
    #   certain appointment types and drop the original appointment_types
    #   column since we only need the binary values.
    #
    # Appointment Types included: Walk-Ins, Resume Reviews, Career Fair
    #   Preparation, Career Exploration, Internship/Job Search, and Other.

    print(Fore.BLUE + "  Extracting appointment types..." + Style.RESET_ALL)

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

    print(Fore.BLUE + "    Walk-Ins extracted" + Style.RESET_ALL)

    # Resume Reviews
    data['has_resume_review'] = data['appointment_types'].apply(
        lambda x: 1 if any('resume' in i for i in x) else 0
    )
    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: [i for i in x if 'resume' not in i]
    )

    print(Fore.BLUE + "    Resume Reviews extracted" + Style.RESET_ALL)

    # Career Fair Prep
    data['has_cf_prep'] = data['appointment_types'].apply(
        lambda x: 1 if any('career fair' in i for i in x) else 0
    )
    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: [i for i in x if 'career fair' not in i]
    )

    print(Fore.BLUE + "    Career Fair Prep extracted" + Style.RESET_ALL)

    # Career Exploration
    data['has_career_exploration'] = data['appointment_types'].apply(
        lambda x: 1 if any('career exploration' in i for i in x) else 0
    )
    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: [i for i in x if 'career exploration' not in i]
    )

    print(Fore.BLUE + "    Career Exploration extracted" + Style.RESET_ALL)

    # Internship/Job Search
    data['has_job_search'] = data['appointment_types'].apply(
        lambda x: 1 if any('internship' in i or 'job' in i for i in x) else 0
    )
    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: [i for i in x if 'internship' not in i and 'job' not in i]
    )

    print(Fore.BLUE + "    Internship/Job Search extracted" + Style.RESET_ALL)

    # Other
    data['has_other'] = data['appointment_types'].apply(
        lambda x: 1 if len(x) > 0 else 0
    )

    print(Fore.BLUE + "    Other extracted" + Style.RESET_ALL)

    # Drop appointment_types since we've extracted the binary values
    data.drop(['appointment_types'], axis=1, inplace=True)

    print(Fore.GREEN + "  Appointments cleaned" + Style.RESET_ALL)

    # ===============================================================
    #                          Null Values
    # ===============================================================

    # Fill null values with 0
    data.fillna(0, inplace=True)

    print(Fore.GREEN + "Data cleaned" + Style.RESET_ALL)

    return data


def extract_features_target(data):
    print(f'{Fore.MAGENTA}\nExtracting features and target...'
          f'{Style.RESET_ALL}')

    features = data.drop(
        ['stu_id', 'is_checked_in', 'career_fair_name', 'career_fair_date'],
        axis=1
    )

    target = data['is_checked_in']

    print(f'{Fore.GREEN}Features and target extracted{Style.RESET_ALL}')

    return features, target


def split_data(features, target, test_size):
    print(Fore.MAGENTA + "\nSplitting data..." + Style.RESET_ALL)
    print(f"{Fore.BLUE}  Test size: {Fore.CYAN}{test_size}" + Style.RESET_ALL)

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=0)

    print(f'{Fore.GREEN}Data split{Style.RESET_ALL}')

    return x_train, x_test, y_train, y_test


def align_features(train_data: pd.DataFrame, test_data: pd.DataFrame):
    all_features = set(train_data.columns) | set(test_data.columns)
    print(f'{Fore.MAGENTA}\nAligning {len(all_features)} features '
          f'between train and test data{Style.RESET_ALL}')

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

    print(f'{Fore.CYAN}{count}{Fore.GREEN} Features aligned{Style.RESET_ALL}')

    return train_data, test_data


def split_practical_data(data: pd.DataFrame, test_career_fair_name: str):
    print(f'{Fore.BLUE}Splitting test and training data by '
          f'{Fore.CYAN}{test_career_fair_name}{Style.RESET_ALL}')

    training_data = data[
        (data['career_fair_name'] != test_career_fair_name)
    ]
    testing_data = data[
        (data['career_fair_name'] == test_career_fair_name)
    ]

    training_data, testing_data = align_features(training_data, testing_data)

    print(f'{Fore.GREEN}Data successfully split into testing and training '
          f'data by {Fore.CYAN}{test_career_fair_name}{Style.RESET_ALL}')

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

    print(f'{Fore.GREEN}Practical test data prepared{Style.RESET_ALL}')

    return x_train, x_test, y_train, y_test, x_practical_test, y_practical_test
