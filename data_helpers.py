import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import List


def set_dependent_variable(last_hire_date) -> int:
    """
    INPUT
    last_hire_date - last hire data variable answer
    OUTPUT
    class label - 0 or 1
    """

    return 0 if last_hire_date == 'Less than a year ago' else 1


def transform_coding_experience(experience) -> float:
    """
    INPUT
    experience - experience variable answer in years
    OUTPUT
    years - years in a numeric field
    """

    conversion = {
        'Less than 1 year': 0.5,
        'More than 50 years': 51
    }

    return conversion.get(experience) if experience in conversion else float(experience)


def filter_employed_developers(df: pd.DataFrame) -> pd.DataFrame:
    """
    INPUT
    df - a pandas DataFrame
    OUTPUT
    filtered df - the input DataFrame filtered by developers who are currently working
    """

    return df[
        (df.LastHireDate != 'NA - I am an independent contractor or self employed') &
        (df.LastHireDate != "I've never had a job") &
        (df.Employment.isin(['Employed full-time', 'Employed part-time']))
        ]


def fillna_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    INPUT
    df - a pandas DataFrame with only numeric features
    OUTPUT
    transformed df - the DataFrame with nan values filled by its feature mean
    """

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(df)

    return imputer.transform(df)


def fillna_with_none(df: pd.DataFrame) -> pd.DataFrame:
    """
    INPUT
    df - a pandas DataFrame with only categorical features
    OUTPUT
    transformed df - the DataFrame with nan filled by None string
    """

    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='None')
    imputer.fit(df)

    return imputer.transform(df)


def encode_categorical_features(df: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
    """
    INPUT
    df - a pandas DataFrame with all features
    categorical_features - list of categorical feature namees
    OUTPUT
    transformed df - the original DataFrame with all categorical features encoded
    """

    for var in categorical_features:
        # for each cat add dummy var, drop original column
        df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var], prefix=var, prefix_sep='_', drop_first=True)],
                       axis=1)

    return df


def transform_feature_on_popularity(feature: str, feature_popularity: dict) -> str:
    """
    INPUT
    feature - any categorical feauture (e.g., Developer, full-stack; Developer, back-end)
    feature_popularity - a dictionary mapping the feature to popularity
    OUTPUT
    transformed feature - the more popular feature value (e.g., Developer, full-stack)
    """

    feature_types = feature.split(';')
    max_pop = -1
    popular_feature_type = 'None'

    for r in feature_types:
        pop = feature_popularity.get(r, -1)

        if pop > max_pop:
            popular_feature_type = r
            max_pop = pop

    return popular_feature_type


def add_number_of_different_roles(dev_type: str) -> int:
    """
    INPUT
    dev_type - dev_type answer (separeted by ';')
    OUTPUT
    numeric value - number of different dev types
    """

    try:
        return len(dev_type.split(';'))
    except:
        return 0


def feature_popularity(feature: str, df: pd.DataFrame) -> dict:
    """
    INPUT
    feature - feature to calculate the popularity
    df - a pandas DataFrame
    OUTPUT
    dict - a dict with feature -> popularity data
    """
    temp = df.copy()
    temp.loc[:, 'feature'] = temp[feature].apply(lambda x: str(x).split(';'))

    return temp.explode(feature)[feature].value_counts().to_dict()


def find_optimal_classifier(X, y, cutoffs, n_estimators=100, test_size=.30, random_state=42, plot=True):
    """
    Adapted from https://github.com/jjrunner/stackoverflow code
    INPUT
    X - pandas DataFrame, X matrix
    y - pandas DataFrame, y matrix
    cutoffs - list of ints, cutoff for number of non-zeros in dummy categorical vars
    n_estimators - number of estimators for the random forest
    test_size - float between 0 an 1 (default 0.3), determines the proportion of data as test data
    random_state - int, defaulft 42, controls random state for train_test_split
    plot - boolean, default 0.3, True to plot result

    OUTPUT
    accuracy_scores_train - list of floats of accuracy scores on the train data
    accuracy_scores_test - list of floats of accuracy scores on the test data
    predictor - model object from sklearn
    X_test - reduced X test matrix
    y_test - reduced t test matrix
    """

    acc_scores_train, acc_scores_test, num_feats, results = [], [], [], dict()
    for cutoff in cutoffs:
        # reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])

        # split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size=test_size, random_state=random_state)

        # fit classifier
        classifier = RandomForestClassifier(n_estimators=n_estimators, criterion='entropy', random_state=42)
        classifier.fit(X_train, y_train.values.ravel())

        y_test_preds = classifier.predict(X_test)
        y_train_preds = classifier.predict(X_train)

        # append the accuracy score
        acc_scores_train.append(accuracy_score(y_train, y_train_preds))
        acc_scores_test.append(accuracy_score(y_test, y_test_preds))
        results[str(cutoff)] = accuracy_score(y_test, y_test_preds)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Accuracy score by Number of Features')

        ax1.plot(num_feats, acc_scores_train, label='Train', alpha=.5)
        ax1.set(xlabel='Number of Features', ylabel='Accuracy score')
        ax1.legend(loc=1)

        ax2.plot(num_feats, acc_scores_test, label='Test', alpha=.5)
        ax2.set(xlabel='Number of Features', ylabel='Accuracy score')
        ax2.legend(loc=1)
        plt.show()

    best_cutoff = max(results, key=results.get)

    # reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size=test_size, random_state=random_state)

    # fit the model
    classifier = RandomForestClassifier(n_estimators=n_estimators, criterion='entropy', random_state=42)
    classifier.fit(X_train, y_train.values.ravel())

    return acc_scores_train, acc_scores_test, classifier, X_test, y_test
