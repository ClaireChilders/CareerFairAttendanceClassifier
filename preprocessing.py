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

    # Convert dates to year only
    # stu_created, stu_login, stu_grad_date

    data['stu_created'] = pd.to_datetime(
        data['stu_created']).dt.year
    data['stu_login'] = pd.to_datetime(
        data['stu_login']).dt.year
    data['stu_grad_date'] = pd.to_datetime(
        data['stu_grad_date']).dt.year

    print(Fore.BLUE + "  Dates converted to year only" + Style.RESET_ALL)

    # ===============================================================
    #                       Appointments Cleaning
    # ===============================================================

    # Convert appointment list to a count of different categories of
    #   appointments including: Walk-Ins, Resume Reviews, Career Fair
    #   Preparation, Career Exploration, Internship/Job Search, and
    #   Other. Also, drop the original appointment_types column since
    #   we only need the counts.

    print(Fore.BLUE + "  Extracting appointment types..." + Style.RESET_ALL)

    # First ensure that appointment_types only contains lists

    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: x.split(',') if isinstance(x, str) else x
    )
    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Walk-Ins
    walk_ins = data['appointment_types'].apply(
        lambda x: [i for i in x if 'walk-in' in i]
    )
    not_walk_ins = data['appointment_types'].apply(
        lambda x: [i for i in x if 'walk-in' not in i]
    )
    data['walk_in_count'] = len(walk_ins)
    data['appointment_types'] = not_walk_ins

    print(Fore.BLUE + "    Walk-Ins extracted" + Style.RESET_ALL)

    # Resume Reviews
    resume_reviews = data['appointment_types'].apply(
        lambda x: [i for i in x if 'resume' in i]
    )
    not_resume_reviews = data['appointment_types'].apply(
        lambda x: [i for i in x if 'resume' not in i]
    )
    data['resume_review_count'] = len(resume_reviews)
    data['appointment_types'] = not_resume_reviews

    print(Fore.BLUE + "    Resume Reviews extracted" + Style.RESET_ALL)

    # Career Fair Prep
    cf_prep = data['appointment_types'].apply(
        lambda x: [i for i in x if 'career fair' in i]
    )
    not_cf_prep = data['appointment_types'].apply(
        lambda x: [i for i in x if 'career fair' not in i]
    )
    data['cf_prep_count'] = len(cf_prep)
    data['appointment_types'] = not_cf_prep

    print(Fore.BLUE + "    Career Fair Prep extracted" + Style.RESET_ALL)

    # Career Exploration
    career_exploration = data['appointment_types'].apply(
        lambda x: [i for i in x if 'career exploration' in i]
    )
    not_career_exploration = data['appointment_types'].apply(
        lambda x: [i for i in x if 'career exploration' not in i]
    )
    data['career_exploration_count'] = len(career_exploration)
    data['appointment_types'] = not_career_exploration

    print(Fore.BLUE + "    Career Exploration extracted" + Style.RESET_ALL)

    # Internship/Job Search
    job_search = data['appointment_types'].apply(
        lambda x: [i for i in x if 'internship' in i or 'job' in i]
    )
    not_job_search = data['appointment_types'].apply(
        lambda x: [i for i in x if 'internship' not in i and 'job' not in i]
    )
    data['job_search'] = len(job_search)
    data['appointment_types'] = not_job_search

    print(Fore.BLUE + "    Internship/Job Search extracted" + Style.RESET_ALL)

    # Other
    data['other_count'] = data['appointment_types'].apply(
        lambda x: len(x)
    )

    print(Fore.BLUE + "    Other extracted" + Style.RESET_ALL)

    # Drop appointment_types since we've extracted the counts
    data.drop(['appointment_types'], axis=1, inplace=True)

    print(Fore.BLUE + "    Appointment types dropped" + Style.RESET_ALL)
    print(Fore.GREEN + "  Appointments cleaned" + Style.RESET_ALL)

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

    data['has_major'] = data['stu_majors'].apply(
        lambda x: any(
            [major in career_fair_df['majors'].values for major in x])
    )

    # Convert majors back to strings for dummies
    data['stu_majors'] = data['stu_majors'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else x
    )

    print(Fore.GREEN + "  Majors cleaned" + Style.RESET_ALL)

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

    features = pd.get_dummies(
        data.drop(
            ['stu_id', 'is_checked_in'], axis=1
        ), columns=[
            'appointment_types',
            'career_fair_name',
            'career_fair_date',
            'stu_school_year',
            'stu_majors',
            'stu_colleges'
        ]
    )

    print(f'  {Fore.BLUE}  Categorical features converted to dummies'
          f'{Style.RESET_ALL}')

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
