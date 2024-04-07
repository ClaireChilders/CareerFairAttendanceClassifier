import os
import pandas as pd
from sklearn.model_selection import train_test_split
from colorama import Fore, Style

# =============================================================================
#                           Data Preprocessing
# =============================================================================

cleaned_data_file_name = 'cleaned_data.csv'
data_directory = 'data'


def load_data() -> pd.DataFrame:
    """
    Loads and merges data from multiple CSV files to create a cleaned dataset.

    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    print(f'{Fore.MAGENTA}\nLoading data...{Style.RESET_ALL}')

    if data_directory not in os.listdir():
        os.makedirs(data_directory)

    if cleaned_data_file_name in os.listdir(data_directory):
        print(f'{Fore.LIGHTBLACK_EX}  → {Fore.BLUE}Loading cleaned '
              f'data...{Style.RESET_ALL}')
        cleaned_data = pd.read_csv(
            f'{data_directory}\\{cleaned_data_file_name}')
        print(f'{Fore.GREEN}  ✓{Fore.LIGHTCYAN_EX} Cleaned data loaded'
              f'{Style.RESET_ALL}')
        return cleaned_data

    appointment_df = pd.read_csv('data/appointment_data.csv')
    career_fair_df = pd.read_csv('data/career_fair_data.csv')
    registration_df = pd.read_csv('data/registration_data.csv')
    student_df = pd.read_csv('data/student_data.csv')
    stu_counts_1_df = pd.read_csv('data/student_counts_1.csv')
    stu_counts_2_df = pd.read_csv('data/student_counts_2.csv')
    stu_fair_attendance_df = pd.read_csv('data/student_fair_attendance.csv')
    stu_event_attendance_df = pd.read_csv('data/student_event_attendance.csv')

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
            # 'stu_id',
            'career_fair_id',
        ], axis=1, inplace=True)

    print(f'{Fore.LIGHTBLACK_EX}  ⓘ {Fore.BLUE} Columns: '
          f'{Fore.LIGHTBLACK_EX}{merged_data.columns}'
          f'{Style.RESET_ALL}')

    print(f'{Fore.GREEN}✓{Fore.MAGENTA} Data merged{Style.RESET_ALL}')

    cleaned_data = clean_data(merged_data, career_fair_df,
                              stu_fair_attendance_df, stu_event_attendance_df)

    print(f'{Fore.MAGENTA}\nSaving cleaned data...{Style.RESET_ALL}')

    cleaned_data.to_csv(
        f'{data_directory}\\{cleaned_data_file_name}',
        index=False
    )

    print(f'{Fore.GREEN}✓{Fore.MAGENTA} Cleaned data saved to '
          f'{Fore.LIGHTBLACK_EX}{data_directory}\\{cleaned_data_file_name}'
          f'{Style.RESET_ALL}')

    return cleaned_data


def clean_data(
    data: pd.DataFrame,
    career_fair_df: pd.DataFrame,
    stu_fair_attendance_df: pd.DataFrame,
    stu_event_attendance_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Cleans the data by filling null values, converting Yes/No values to 1/0,
      converting strings, dates, school years, colleges, majors, and
      appointments to binary values.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to be
          cleaned.
        career_fair_df (pd.DataFrame): The DataFrame containing career fair
          information.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
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

    print(f'{Fore.LIGHTBLACK_EX}  → {Fore.BLUE}Converting numerical values '
          f'to binary values...{Style.RESET_ALL}')

    for column in count_columns:
        data[column] = data[column].apply(
            lambda x: int(x.replace(',', '')) if isinstance(x, str) else x)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Strings converted to integers'
          f'{Style.RESET_ALL}')

    # Convert integers to binary thresholds values
    #   and drop the original columns

    data['has_0_appointments'] = data['stu_appointments'].apply(
        lambda x: 1 if x == 0 else 0)
    data['has_1-5_appointments'] = data['stu_appointments'].apply(
        lambda x: 1 if x >= 1 and x < 5 else 0)
    data['has_5-10_appointments'] = data['stu_appointments'].apply(
        lambda x: 1 if x >= 5 and x < 10 else 0)
    data['has_10+_appointments'] = data['stu_appointments'].apply(
        lambda x: 1 if x >= 10 else 0)
    data['has_0_applications'] = data['stu_applications'].apply(
        lambda x: 1 if x == 0 else 0)
    data['has_1-5_applications'] = data['stu_applications'].apply(
        lambda x: 1 if x >= 1 and x < 5 else 0)
    data['has_5-10_applications'] = data['stu_applications'].apply(
        lambda x: 1 if x >= 5 and x < 10 else 0)
    data['has_10+_applications'] = data['stu_applications'].apply(
        lambda x: 1 if x >= 10 else 0)
    data['has_0_logins'] = data['stu_logins'].apply(
        lambda x: 1 if x == 0 else 0)
    data['has_1-10_logins'] = data['stu_logins'].apply(
        lambda x: 1 if x >= 1 and x < 10 else 0)
    data['has_10-100_logins'] = data['stu_logins'].apply(
        lambda x: 1 if x >= 10 and x < 100 else 0)
    data['has_100+_logins'] = data['stu_logins'].apply(
        lambda x: 1 if x >= 100 else 0)
    data['has_0_attendances'] = data['stu_attendances'].apply(
        lambda x: 1 if x == 0 else 0)
    data['has_1-2_attendance'] = data['stu_attendances'].apply(
        lambda x: 1 if x >= 1 and x <= 2 else 0)
    data['has_3-5_attendances'] = data['stu_attendances'].apply(
        lambda x: 1 if x <= 5 and x >= 3 else 0)
    data['has_5-10_attendances'] = data['stu_attendances'].apply(
        lambda x: 1 if x <= 10 and x > 5 else 0)
    data['has_10+_attendances'] = data['stu_attendances'].apply(
        lambda x: 1 if x > 10 else 0)
    data['has_0_work_experiences'] = data['stu_work_experiences'].apply(
        lambda x: 1 if x == 0 else 0)
    data['has_1_work_experience'] = data['stu_work_experiences'].apply(
        lambda x: 1 if x == 1 else 0)
    data['has_2_work_experiences'] = data['stu_work_experiences'].apply(
        lambda x: 1 if x == 2 else 0)
    data['has_3+_work_experiences'] = data['stu_work_experiences'].apply(
        lambda x: 1 if x >= 3 else 0)
    data['has_learning_experience'] = data['stu_experiences'].apply(
        lambda x: 1 if x > 0 else 0)

    data.drop(count_columns, axis=1, inplace=True)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Integers converted to binary '
          f'values{Style.RESET_ALL}')
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} All strings converted to '
          f'binary values{Style.RESET_ALL}')

    # ===============================================================
    #                     Fair Attendance Cleaning
    # ===============================================================
    print(f'{Fore.LIGHTBLACK_EX}  → {Fore.BLUE}Extracting fair attendance '
          f'values...{Style.RESET_ALL}')

    # stu_fair_attendance_df contains a row for each career fair for each
    #   student that attended that career fair. We want to count the number
    #   of fairs before the fair date for each student and merge that with
    #   the student data.
    #
    # For every row in the data, we want to lookup and count the number of
    #   fairs that the student attended before the career fair date for
    #   that row.
    #
    # Process:
    #   1. Get the cross product of the fair attendances and career fair dates
    #       to get a row for each attendance for each career fair.
    #   2. Filter out the attendances that are after the career fair date
    #       to get the attendances that are before the career fair date.
    #   3. Separate into main career fair and other career fair attendances.
    #      (main career fairs are the ones we are predicting)
    #   4. Group by student id and career fair date and count the number of
    #       attendances before the career fair date.
    #   5. Merge the counts with the data and fill null values with 0.
    #   6. Convert the counts to binary values.

    # Step 0. - Data initialization
    stu_fair_attendance_df['career_fair_date'] = pd.to_datetime(
        stu_fair_attendance_df['career_fair_date'])

    simple_cf_df = career_fair_df[['career_fair_name', 'career_fair_date']]
    main_fair_names = simple_cf_df['career_fair_name'].unique()
    simple_cf_df.loc[:, 'career_fair_date'] = pd.to_datetime(
        simple_cf_df['career_fair_date'])

    stu_fair_attendance_df.rename(
        columns={
            'career_fair_name': 'attended_career_fair_name',
            'career_fair_date': 'attended_career_fair_date'
        },
        inplace=True
    )

    # Step 1.
    cross_attendances = pd.merge(
        stu_fair_attendance_df, simple_cf_df,
        how='cross')
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Cross product of fair '
          f'attendance and career fair dates created{Style.RESET_ALL}')

    # Step 2.
    previous_attendances = cross_attendances[
        cross_attendances['career_fair_date'] >
        cross_attendances['attended_career_fair_date']
    ]
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Attendances before career '
          f'fair date extracted{Style.RESET_ALL}')

    # Step 3.
    previous_main_attendances = previous_attendances[
        previous_attendances['attended_career_fair_name'].isin(main_fair_names)
    ]
    previous_other_attendances = previous_attendances[
        ~previous_attendances['attended_career_fair_name'].isin(
            main_fair_names)
    ]
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Main and other fair '
          f'attendances separated{Style.RESET_ALL}')

    # Step 4.
    previous_main_attendances = previous_main_attendances.groupby(
        ['stu_id', 'career_fair_date']
    ).agg(
        {'attended_career_fair_date': 'count'}
    ).reset_index()
    previous_other_attendances = previous_other_attendances.groupby(
        ['stu_id', 'career_fair_date']
    ).agg(
        {'attended_career_fair_date': 'count'}
    ).reset_index()

    previous_main_attendances.rename(
        columns={
            'attended_career_fair_date': 'attended_main_fair_before',
        },
        inplace=True
    )
    previous_other_attendances.rename(
        columns={
            'attended_career_fair_date': 'attended_other_fair_before',
        },
        inplace=True
    )

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Fair attendance grouped by '
          f'student id and career fair date{Style.RESET_ALL}')

    # Step 5.
    # First we have to ensure career_fair_date is in the same format
    data['career_fair_date'] = pd.to_datetime(data['career_fair_date'])
    # Then we can merge the counts with the data
    data = pd.merge(
        data, previous_main_attendances,
        on=['stu_id', 'career_fair_date'],
        how='left')
    data = pd.merge(
        data, previous_other_attendances,
        on=['stu_id', 'career_fair_date'],
        how='left')

    data['attended_other_fair_before'] = (
        data['attended_other_fair_before'].fillna(0))
    data['attended_main_fair_before'] = (
        data['attended_main_fair_before'].fillna(0))

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Fair attendance merged with '
          f'data{Style.RESET_ALL}')

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Fair attendance extracted'
          f'{Style.RESET_ALL}')

    # Step 6.
    data['attended_0_main_fairs_before'] = (
        data['attended_main_fair_before'].apply(
            lambda x: 1 if x == 0 else 0))
    data['attended_1_main_fair_before'] = (
        data['attended_main_fair_before'].apply(
            lambda x: 1 if x == 1 else 0))
    data['attended_2_main_fairs_before'] = (
        data['attended_main_fair_before'].apply(
            lambda x: 1 if x == 2 else 0))
    data['attended_3+_main_fairs_before'] = (
        data['attended_main_fair_before'].apply(
            lambda x: 1 if x >= 3 else 0))

    data['attended_0_other_fairs_before'] = (
        data['attended_other_fair_before'].apply(
            lambda x: 1 if x == 0 else 0))
    data['attended_1_other_fairs_before'] = (
        data['attended_other_fair_before'].apply(
            lambda x: 1 if x == 1 and x <= 2 else 0))
    data['attended_2_other_fairs_before'] = (
        data['attended_other_fair_before'].apply(
            lambda x: 1 if x == 2 else 0))
    data['attended_3+_other_fairs_before'] = (
        data['attended_other_fair_before'].apply(
            lambda x: 1 if x >= 3 else 0))

    data.drop(['attended_main_fair_before', 'attended_other_fair_before'],
              axis=1, inplace=True)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Fair attendance converted to '
          f'binary values{Style.RESET_ALL}')

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Fair attendance extracted'
          f'{Style.RESET_ALL}')

    # ===============================================================
    #                     Event Attendance Cleaning
    # ===============================================================
    print(f'{Fore.LIGHTBLACK_EX}  → {Fore.BLUE}Extracting event attendance '
          f'values...{Style.RESET_ALL}')

    # stu_event_attendance_df contains a row for each event for each
    #   student that attended that event. We want to count the number
    #   of events before the career fair date for each student and merge
    #   that with the student data.
    #
    # Unlike the career fair attendance where we counted past attendance,
    #   we also want to consider the category of the event. Each event may
    #   have multiple categories.
    #
    # Possible Event Categories:
    #   - Academic
    #   - Career fairs
    #   - Conference
    #   - Employers
    #   - General
    #   - Guidance
    #   - Hiring
    #   - Networking
    #
    # We also want to consider specifically career fair prep sessions
    #   because they are likely to be more relevant to the career fair.
    #
    # Process:
    #   1. Get the cross product of the event attendances and career fair dates
    #   2. Filter out the attendances that are after the career fair date
    #   3. Separate categories into different dataframes
    #   4. Separate career fair prep sessions into a separate dataframe
    #   5. For each dataframe, group by student id and career fair date and
    #      count the number of attendances.
    #   6. For the career fair prep session dataframe, add a boolean for
    #      whether that attendance/event is for that upcoming career fair
    #      (this can be determined as whether the event is within a month)
    #   7. Merge the counts with the data and fill null values with 0.
    #   8. Convert the counts to binary values.

    print(f'{Fore.LIGHTBLACK_EX}    → {Fore.BLUE}Loading event attendance '
          f'data...{Style.RESET_ALL}')

    stu_event_attendance_df['event_date'] = pd.to_datetime(
        stu_event_attendance_df['event_date'])

    # Convert string list to list type
    stu_event_attendance_df['event_categories'] = (
        stu_event_attendance_df['event_categories'].apply(
            lambda x: (
                x.lower().strip().split(',') if isinstance(x, str) else []
            )
        )
    )

    all_event_categories = (
        stu_event_attendance_df['event_categories'].explode().unique())
    event_categories = set(
        str(category).strip().lower().replace(' ', '_')
        for category in all_event_categories
    )
    print(f'{Fore.GREEN}      ✓{Fore.LIGHTCYAN_EX} Event categories loaded')
    print(f'{Fore.LIGHTBLACK_EX}      ⓘ {Fore.BLUE} Event Categories: '
          f'{Fore.LIGHTBLACK_EX}{event_categories}{Style.RESET_ALL}')

    print(f'{Fore.GREEN}      ✓{Fore.LIGHTCYAN_EX} Event attendance data '
          f'loaded{Style.RESET_ALL}')

    # Step 1.
    cross_event_attendances = pd.merge(
        stu_event_attendance_df, simple_cf_df,
        how='cross')
    print(f'{Fore.GREEN}      ✓{Fore.LIGHTCYAN_EX} Cross product of event '
          f'attendance and career fair dates created{Style.RESET_ALL}')

    # Step 2.
    previous_event_attendances = cross_event_attendances[
        cross_event_attendances['career_fair_date'] >
        cross_event_attendances['event_date']
    ]
    print(f'{Fore.GREEN}      ✓{Fore.LIGHTCYAN_EX} Attendances before career '
          f'fair date extracted{Style.RESET_ALL}')

    # Step 3.
    print(f'{Fore.LIGHTBLACK_EX}    → {Fore.BLUE}Separating event attendance '
          f'by category...{Style.RESET_ALL}')
    category_dfs = {}
    for category in event_categories:
        category_df = previous_event_attendances[
            previous_event_attendances['event_categories'].apply(
                lambda x: category in x)
        ]
        category_dfs[category] = category_df
        print(f'{Fore.GREEN}      ✓ {Fore.LIGHTMAGENTA_EX}{category}'
              f'{Fore.LIGHTCYAN_EX} event attendance separated'
              f'{Style.RESET_ALL}')
    print(f'{Fore.GREEN}      ✓{Fore.LIGHTCYAN_EX} Event attendance separated '
          f'by category{Style.RESET_ALL}')

    # Step 4.
    print(f'{Fore.LIGHTBLACK_EX}    → {Fore.BLUE}Separating career fair prep '
          f'sessions...{Style.RESET_ALL}')

    career_fair_prep_df = previous_event_attendances[
        previous_event_attendances['event_name'].apply(
            lambda x: 'career fair' in str(x).lower())
    ]

    category_dfs['cf_prep'] = career_fair_prep_df.copy(deep=True)
    event_categories.add('cf_prep')
    # Step 5.
    print(f'{Fore.LIGHTBLACK_EX}    → {Fore.BLUE}Grouping event attendance by '
          f'student id and career fair date...{Style.RESET_ALL}')
    for category, category_df in category_dfs.items():
        category_df = category_df.groupby(
            ['stu_id', 'career_fair_date']
        ).agg(
            {'event_date': 'count'}
        ).reset_index()
        category_df.rename(
            columns={'event_date': f'attended_{category}_before'},
            inplace=True
        )
        category_dfs[category] = category_df
        print(f'{Fore.GREEN}      ✓ {Fore.LIGHTMAGENTA_EX}{category}'
              f'{Fore.LIGHTCYAN_EX} event attendance grouped by student id'
              f'and career fair date{Style.RESET_ALL}')
    print(f'{Fore.GREEN}      ✓{Fore.LIGHTCYAN_EX} Event attendance grouped '
          f'by student id and career fair date{Style.RESET_ALL}')

    # Step 6.
    print(f'{Fore.LIGHTBLACK_EX}    → {Fore.BLUE}Adding boolean for career '
          f'fair prep sessions...{Style.RESET_ALL}')

    prep_event_dates = (pd.to_datetime(career_fair_prep_df['career_fair_date']) -
         pd.to_datetime(career_fair_prep_df['event_date'])).dt.days
    
    attended_prep_event = career_fair_prep_df[prep_event_dates <= 60]

    # drop duplicates because we only want to count each prep session once per 
    #   career fair/student
    attended_prep_event = attended_prep_event.drop_duplicates(
        subset=['stu_id', 'career_fair_date'])

    print(f'{Fore.GREEN}      ✓{Fore.LIGHTCYAN_EX} Selected relevant career '
          f'fair prep sessions (within 60 days of career fair date)'
          f'{Style.RESET_ALL}')

    data['attended_career_fair_prep'] = (
        (data['stu_id'].isin(attended_prep_event['stu_id'])) &
        (data['career_fair_date'].isin(attended_prep_event['career_fair_date']))
    ).astype(int)

    print(f'{Fore.GREEN}      ✓{Fore.LIGHTCYAN_EX} Career fair prep sessions '
          f'boolean added{Style.RESET_ALL}')

    # Step 7.
    print(f'{Fore.LIGHTBLACK_EX}    → {Fore.BLUE}Merging event attendance '
          f'with data...{Style.RESET_ALL}')
    for category, category_df in category_dfs.items():
        data = pd.merge(
            data, category_df,
            on=['stu_id', 'career_fair_date'],
            how='left')
        data[f'attended_{category}_before'] = (
            data[f'attended_{category}_before'].fillna(0))
        print(f'{Fore.GREEN}      ✓ {Fore.LIGHTMAGENTA_EX}{category}'
              f'{Fore.LIGHTCYAN_EX} event attendance merged with data'
              f'{Style.RESET_ALL}')
    print(f'{Fore.GREEN}      ✓{Fore.LIGHTCYAN_EX} Event attendance merged '
          f'with data{Style.RESET_ALL}')

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Event attendance extracted'
          f'{Style.RESET_ALL}')

    # Step 8.
    print(f'{Fore.LIGHTBLACK_EX}    → {Fore.BLUE}Converting event attendance '
          f'to binary values...{Style.RESET_ALL}')
    for category in event_categories:
        data[f'attended_0_past_{category}_events'] = (
            data[f'attended_{category}_before'].apply(
                lambda x: 1 if x == 0 else 0))
        data[f'attended_1_past_{category}_events'] = (
            data[f'attended_{category}_before'].apply(
                lambda x: 1 if x == 1 else 0))
        data[f'attended_2_past_{category}_events'] = (
            data[f'attended_{category}_before'].apply(
                lambda x: 1 if x == 2 else 0))
        data[f'attended_3+_past_{category}_events'] = (
            data[f'attended_{category}_before'].apply(
                lambda x: 1 if x >= 3 else 0))

        data.drop([f'attended_{category}_before'],
                  axis=1, inplace=True)
        print(f'{Fore.GREEN}      ✓ {Fore.LIGHTMAGENTA_EX}{category}'
              f'{Fore.LIGHTCYAN_EX} event attendance converted to binary '
              f'values{Style.RESET_ALL}')

    print(f'{Fore.GREEN}      ✓{Fore.LIGHTCYAN_EX} Event attendance converted '
          f'to binary values{Style.RESET_ALL}')

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Event attendance '
          f'extracted{Style.RESET_ALL}')

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
        lambda x: 1 if x <= 730 and x > 365 else 0)
    data['created_3_years_pre_cf'] = data['days_since_created'].apply(
        lambda x: 1 if x <= 1095 and x > 730 else 0)
    data['created_4_years_pre_cf'] = data['days_since_created'].apply(
        lambda x: 1 if x <= 1825 and x > 1095 else 0)
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
        lambda x: 1 if x <= 30 and x > 7 else 0)
    data['login_90_days_pre_cf'] = data['days_since_login'].apply(
        lambda x: 1 if x <= 90 and x > 30 else 0)
    data['login_90+_days_pre_cf'] = data['days_since_login'].apply(
        lambda x: 1 if x > 90 else 0)

    data.drop(['stu_login_date', 'days_since_login'], axis=1, inplace=True)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Login date converted to '
          f'binary values{Style.RESET_ALL}')

    # Date between stu_grad_date and career_fair_date
    data['days_until_grad'] = (
        pd.to_datetime(data['stu_grad_date']) -
        pd.to_datetime(data['career_fair_date'])
    ).dt.days

    data['grad_4+_years_pre_cf'] = data['days_until_grad'].apply(
        lambda x: 1 if x <= -1095 else 0)
    data['grad_3_years_pre_cf'] = data['days_until_grad'].apply(
        lambda x: 1 if x <= -730 and x > -1095 else 0)
    data['grad_2_years_pre_cf'] = data['days_until_grad'].apply(
        lambda x: 1 if x <= -365 and x > -730 else 0)
    data['grad_1_year_pre_cf'] = data['days_until_grad'].apply(
        lambda x: 1 if x <= 0 and x > -365 else 0)

    data['grad_1_year_post_cf'] = data['days_until_grad'].apply(
        lambda x: 1 if x <= 365 and x > 0 else 0)
    data['grad_2_years_post_cf'] = data['days_until_grad'].apply(
        lambda x: 1 if x <= 730 and x > 365 else 0)
    data['grad_3_years_post_cf'] = data['days_until_grad'].apply(
        lambda x: 1 if x <= 1095 and x > 730 else 0)
    data['grad_4+_years_post_cf'] = data['days_until_grad'].apply(
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
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted '
          f'{Fore.LIGHTMAGENTA_EX}Freshmen {Style.RESET_ALL}')
    data['is_sophomore'] = data['stu_school_year'].apply(
        lambda x: 1 if 'Sophomore' in x else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted '
          f'{Fore.LIGHTMAGENTA_EX}Sophomores {Style.RESET_ALL}')
    data['is_junior'] = data['stu_school_year'].apply(
        lambda x: 1 if 'Junior' in x else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted '
          f'{Fore.LIGHTMAGENTA_EX}Juniors {Style.RESET_ALL}')
    data['is_senior'] = data['stu_school_year'].apply(
        lambda x: 1 if 'Senior' in x else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted '
          f'{Fore.LIGHTMAGENTA_EX}Seniors {Style.RESET_ALL}')
    data['is_alumni'] = data['stu_school_year'].apply(
        lambda x: 1 if 'Alumni' in x else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted '
          f'{Fore.LIGHTMAGENTA_EX}Alumni {Style.RESET_ALL}')
    data['is_masters'] = data['stu_school_year'].apply(
        lambda x: 1 if 'Masters' in x else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted '
          f'{Fore.LIGHTMAGENTA_EX}Masters Students {Style.RESET_ALL}')
    data['is_doctorate'] = data['stu_school_year'].apply(
        lambda x: 1 if 'Doctorate' in x else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted '
          f'{Fore.LIGHTMAGENTA_EX}Doctoral Students {Style.RESET_ALL}')

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
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted '
          f'{Fore.LIGHTMAGENTA_EX}engineering '
          f'{Fore.LIGHTCYAN_EX}binary values{Style.RESET_ALL}')
    data['is_business'] = data['stu_colleges'].apply(
        lambda x: 1 if any([college in business for college in x]) else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted '
          f'{Fore.LIGHTMAGENTA_EX}business '
          f'{Fore.LIGHTCYAN_EX}binary values{Style.RESET_ALL}')
    data['is_health'] = data['stu_colleges'].apply(
        lambda x: 1 if any([college in health for college in x]) else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted '
          f'{Fore.LIGHTMAGENTA_EX}health '
          f'{Fore.LIGHTCYAN_EX}binary values{Style.RESET_ALL}')
    data['is_education'] = data['stu_colleges'].apply(
        lambda x: 1 if any([college in education for college in x]) else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted '
          f'{Fore.LIGHTMAGENTA_EX}education '
          f'{Fore.LIGHTCYAN_EX}binary values{Style.RESET_ALL}')
    data['is_arts'] = data['stu_colleges'].apply(
        lambda x: 1 if any([college in arts for college in x]) else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted '
          f'{Fore.LIGHTMAGENTA_EX}arts '
          f'{Fore.LIGHTCYAN_EX}binary values{Style.RESET_ALL}')
    data['no_college'] = data['stu_colleges'].apply(
        lambda x: 1 if any([college in no_college for college in x]) else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted '
          f'{Fore.LIGHTMAGENTA_EX}no college '
          f'{Fore.LIGHTCYAN_EX}binary values{Style.RESET_ALL}')
    data['multiple_colleges'] = data['stu_colleges'].apply(
        lambda x: 1 if len(x) > 1 else 0)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted '
          f'{Fore.LIGHTMAGENTA_EX}multiple colleges '
          f'{Fore.LIGHTCYAN_EX}binary values{Style.RESET_ALL}')
    data['other_college'] = data['stu_colleges'].apply(
        lambda x: 0 if any(college in all_colleges for college in x) else 1)
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Extracted '
          f'{Fore.LIGHTMAGENTA_EX}other colleges '
          f'{Fore.LIGHTCYAN_EX}binary values{Style.RESET_ALL}')

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
            [major in career_fair_df['career_fair_majors'].values for major in x]) else 0
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
    data['has_walk_in_appointment'] = data['appointment_types'].apply(
        lambda x: 1 if any('walk-in' in i for i in x) else 0
    )
    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: [i for i in x if 'walk-in' not in i]
    )

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Walk-Ins extracted'
          f'{Style.RESET_ALL}')

    # Resume Reviews
    data['has_resume_review_appointment'] = data['appointment_types'].apply(
        lambda x: 1 if any('resume' in i for i in x) else 0
    )
    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: [i for i in x if 'resume' not in i]
    )

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Resume Reviews extracted'
          f'{Style.RESET_ALL}')

    # Career Fair Prep
    data['has_cf_prep_appointment'] = data['appointment_types'].apply(
        lambda x: 1 if any('career fair' in i for i in x) else 0
    )
    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: [i for i in x if 'career fair' not in i]
    )

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Career Fair Prep extracted'
          f'{Style.RESET_ALL}')

    # Career Exploration
    data['has_career_exploration_appointment'] = (
        data['appointment_types'].apply(
            lambda x: 1 if any('career exploration' in i for i in x) else 0
        )
    )
    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: [i for i in x if 'career exploration' not in i]
    )

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Career Exploration extracted'
          f'{Style.RESET_ALL}')

    # Internship/Job Search
    data['has_job_search_appointment'] = data['appointment_types'].apply(
        lambda x: 1 if any('internship' in i or 'job' in i for i in x) else 0
    )
    data['appointment_types'] = data['appointment_types'].apply(
        lambda x: [i for i in x if 'internship' not in i and 'job' not in i]
    )

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Internship/Job Search '
          f'extracted{Style.RESET_ALL}')

    # Other
    data['has_other_appointment'] = data['appointment_types'].apply(
        lambda x: 1 if len(x) > 0 else 0
    )

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Other extracted'
          f'{Style.RESET_ALL}')

    # Drop appointment_types since we've extracted the binary values
    data.drop(['appointment_types'], axis=1, inplace=True)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Appointments cleaned'
          f'{Style.RESET_ALL}')

    # ===============================================================
    #                          GPA Cleaning
    # ===============================================================

    print(f'{Fore.LIGHTBLACK_EX}  → {Fore.BLUE}Cleaning GPA '
          f'values...{Style.RESET_ALL}')

    data['stu_gpa'] = data['stu_gpa'].apply(
        lambda x: float(x) if isinstance(x, str) else x
    )

    data['no_gpa'] = data['stu_gpa'].apply(
        lambda x: 1 if pd.isna(x) else 0
    )
    data['gpa_1.0'] = data['stu_gpa'].apply(
        lambda x: 1 if x < 1.0 else 0
    )
    data['gpa_1.0-1.5'] = data['stu_gpa'].apply(
        lambda x: 1 if x >= 1.0 and x < 1.5 else 0
    )
    data['gpa_1.5-2.0'] = data['stu_gpa'].apply(
        lambda x: 1 if x >= 1.5 and x < 2.0 else 0
    )
    data['gpa_2.0-2.5'] = data['stu_gpa'].apply(
        lambda x: 1 if x >= 2.0 and x < 2.5 else 0
    )
    data['gpa_2.5-3.0'] = data['stu_gpa'].apply(
        lambda x: 1 if x >= 2.5 and x < 3.0 else 0
    )
    data['gpa_3.0-3.5'] = data['stu_gpa'].apply(
        lambda x: 1 if x >= 3.0 and x < 3.5 else 0
    )
    data['gpa_3.5-4.0'] = data['stu_gpa'].apply(
        lambda x: 1 if x >= 3.5 else 0
    )

    data.drop(['stu_gpa'], axis=1, inplace=True)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} GPA cleaned{Style.RESET_ALL}')

    print(f'{Fore.GREEN}✓{Fore.MAGENTA} Data cleaned{Style.RESET_ALL}')

    data.drop(['stu_id'], axis=1, inplace=True)

    print(f'{Fore.LIGHTBLACK_EX}  ⓘ {Fore.BLUE} Data: '
          f'{Fore.LIGHTBLACK_EX}{len(data)}{Style.RESET_ALL}')
    print(f'{Fore.LIGHTBLACK_EX}  ⓘ {Fore.BLUE} Features: '
          f'{Fore.LIGHTBLACK_EX}{len(data.columns) - 1}{Style.RESET_ALL}')

    return data


def extract_features_target(data):
    """
    Extracts features and target from the given data.

    Parameters:
        data (DataFrame): The input data containing features and target.

    Returns:
        features (DataFrame): The extracted features from the data.
        target (Series): The extracted target from the data.
    """
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
    """
    Split the data into training and testing sets.

    Parameters:
    features (array-like): The input features.
    target (array-like): The target variable.
    test_size (float): The proportion of the dataset to include in the test
      split.

    Returns:
    x_train (array-like): The training features.
    x_test (array-like): The testing features.
    y_train (array-like): The training target variable.
    y_test (array-like): The testing target variable.
    """
    print(f'{Fore.MAGENTA}\n  Splitting data...{Style.RESET_ALL}')
    print(f'{Fore.BLUE}    Test size: {Fore.CYAN}{test_size}{Style.RESET_ALL}')

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=0)

    print(f'{Fore.GREEN}  ✓{Fore.LIGHTCYAN_EX} Data split{Style.RESET_ALL}')

    return x_train, x_test, y_train, y_test


def align_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns the features between the train and test dataframes.

    Args:
        train_data (pd.DataFrame): The training data.
        test_data (pd.DataFrame): The testing data.

    Returns:
        pd.DataFrame, pd.DataFrame: The aligned training and testing
          dataframes.
    """
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
    """
    Split the given data into training and testing data based on the specified
      test career fair name.

    Parameters:
        data (pd.DataFrame): The input data to be split.
        test_career_fair_name (str): The name of the career fair to be used as
          the test data.

    Returns:
        training_data (pd.DataFrame): The training data after splitting.
        testing_data (pd.DataFrame): The testing data after splitting.
    """
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
    """
    Prepare practical test data for the Career Fair Attendance Classifier.

    Args:
        data (pd.DataFrame): The cleaned input data containing the attendance
          records.
        test_career_fair_name (str): The name of the career fair to be used
          for practical testing.
        test_size (float): The proportion of the data from 0 to 1 to be used
          for testing.

    Returns:
        tuple: A tuple containing the training and testing data for the
          classifier.
            - x_train (pd.DataFrame): The features of the training data.
            - x_test (pd.DataFrame): The features of the testing data.
            - y_train (pd.Series): The target labels of the training data.
            - y_test (pd.Series): The target labels of the testing data.
            - x_practical_test (pd.DataFrame): The features of the practical
              test data.
            - y_practical_test (pd.Series): The target labels of the practical
              test data.
    """
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


def train_test_validate(
    data: pd.DataFrame,
    test_size: float,
    validation_size: float
):
    """
    Split the given data into training and testing data.

    Parameters:
        data (pd.DataFrame): The input data to be split.
        test_size (float): The proportion of the dataset to include in the test
          split.
        validation_size (float): The proportion of the dataset to include in
            the validation split.

    Returns:
        x_train (pd.DataFrame): The training features.
        x_test (pd.DataFrame): The testing features.
        x_val (pd.DataFrame): The validation features.
        y_train (pd.Series): The training target variable.
        y_test (pd.Series): The testing target variable.
        y_val (pd.Series): The validation target variable.
    """
    print(f'{Fore.MAGENTA}\n  Splitting data...{Style.RESET_ALL}')
    print(f'{Fore.LIGHTBLACK_EX}    ⓘ {Fore.BLUE} Test size: '
          f'{Fore.CYAN}{test_size}{Style.RESET_ALL}')
    print(f'{Fore.LIGHTBLACK_EX}    ⓘ {Fore.BLUE} Validation size: '
          f'{Fore.CYAN}{validation_size}{Style.RESET_ALL}')

    features, target = extract_features_target(data)

    print(f'{Fore.LIGHTBLACK_EX}    → {Fore.BLUE}Splitting train, test, '
          f'validation data sets...{Style.RESET_ALL}')

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=0)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Test data split'
          f'{Style.RESET_ALL}')

    # Example:
    #   test_size = 0.2, validation_size = 0.1
    #
    #   The train size will be 0.7 in this case
    #
    #   First, we split train and test from the data with test_size. Now the
    #   training data is 0.8 the original data. Then we split the training
    #   data by 0.125 (0.1 / 0.8) to get the validation data. Tihs gives us
    #   validation data being 0.1 th size of the original data, the
    #   training data being 0.7 the original data, and the testing data being
    #   0.2 the original data.

    print(f'{Fore.LIGHTBLACK_EX}    → {Fore.BLUE}Splitting validation data...'
          f'{Style.RESET_ALL}')

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train,
        test_size=validation_size/(1-test_size),
        random_state=0)

    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Training data split'
          f'{Style.RESET_ALL}')
    print(f'{Fore.GREEN}    ✓{Fore.LIGHTCYAN_EX} Validation data split'
          f'{Style.RESET_ALL}')

    print(f'{Fore.GREEN}  ✓{Fore.LIGHTCYAN_EX} Data split{Style.RESET_ALL}')

    return x_train, x_test, x_val, y_train, y_test, y_val


def print_metrics(
    mse=None,
    accuracy=None,
    f1=None,
    recall=None,
    precision=None
):
    if mse is not None:
        if mse > 0.25:
            print(f'{Fore.RED}    Mean Squared Error: '
                  f'{Fore.CYAN}{mse:.4f}' + Style.RESET_ALL)
        else:
            print(f'{Fore.GREEN}    Mean Squared Error: '
                  f'{Fore.CYAN}{mse:.4f}' + Style.RESET_ALL)

    if accuracy is not None:
        if accuracy < 0.75:
            print(f'{Fore.RED}    Accuracy: '
                  f'{Fore.CYAN}{accuracy:.4f}' + Style.RESET_ALL)
        else:
            print(f'{Fore.GREEN}    Accuracy: '
                  f'{Fore.CYAN}{accuracy:.4f}' + Style.RESET_ALL)

    if f1 is not None:
        if f1 < 0.5:
            print(f'{Fore.RED}    F1 Score: '
                  f'{Fore.CYAN}{f1:.4f}' + Style.RESET_ALL)
        else:
            print(f'{Fore.GREEN}    F1 Score: '
                  f'{Fore.CYAN}{f1:.4f}' + Style.RESET_ALL)

    if recall is not None:
        if recall < 0.6:
            print(f'{Fore.RED}    Recall: '
                  f'{Fore.CYAN}{recall:.4f}' + Style.RESET_ALL)
        else:
            print(f'{Fore.GREEN}    Recall: '
                  f'{Fore.CYAN}{recall:.4f}' + Style.RESET_ALL)
    if precision is not None:
        if precision < 0.6:
            print(f'{Fore.RED}    Precision: '
                  f'{Fore.CYAN}{precision:.4f}' + Style.RESET_ALL)
        else:
            print(f'{Fore.GREEN}    Precision: '
                  f'{Fore.CYAN}{precision:.4f}' + Style.RESET_ALL)
