def bucketize_last_hire_date(last_hire_date) -> str :
    '''
    INPUT - last hire data answer
    OUTPUT - a discrete value assigned to each answer
    '''
    buckets = {
        'Less than a year ago': 1,
        '1-2 years ago': 2,
        '3-4 years ago': 3,
        'More than 4 years ago': 4
    }

    return buckets.get(last_hire_date, -1)


def transform_coding_experience(experience) -> float :
    '''
    INPUT - experience answer in years
    OUTPUT - years in a numeric field
    '''

    conversion = {
        'Less than 1 year': 0.5,
        'More than 50 years': 51
    }

    return conversion.get(experience) if experience in conversion else float(experience)