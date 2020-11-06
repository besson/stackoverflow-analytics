# Predicting how long developers will stay in their current jobs

In this analysis, we will investigate what factors can "predict" how many years developers will stay in their current job.


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap

from data_helpers import transform_coding_experience
from data_helpers import filter_employed_developers
from data_helpers import fillna_with_mean
from data_helpers import fillna_with_none
from data_helpers import encode_categorical_features
from data_helpers import transform_feature_on_popularity
from data_helpers import feature_popularity
from data_helpers import find_optimal_classifier
from data_helpers import set_dependent_variable
from data_helpers import add_number_of_different_roles

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
sns.set(color_codes=True)
```

## Loading data


```python
df = pd.read_csv('data/stackoverflow-2019/survey_results_public.csv')
df.shape
```




    (88883, 85)



## Part III: What other factors lead developers stay longer in their jobs ?

To simplify the problem, we will reduce it to a binary classifier. Then, we have this binary dependent variable:

* 0 if developers are **Less than a year ago** in their current jobs
* 1, otherwise

This way, our model predict whether developers will stay more than **1 year** in the current job or not.

### Data preparation


```python
df = filter_employed_developers(df)
df.shape
```




    (68063, 85)




```python
df = df[~df.LastHireDate.isna()]
df.loc[:,'MoreThanOneYear'] = df['LastHireDate'].apply(set_dependent_variable)
df['MoreThanOneYear'].value_counts()
```




    1    43064
    0    22378
    Name: MoreThanOneYear, dtype: int64



### Feature Engineering


```python
X = df.copy()

## Pre-selected features
X = X[['Hobbyist', 'OpenSourcer', 'Country', 'Student', 'EdLevel', 'EduOther', 'UndergradMajor', 'OrgSize', 'DevType',
        'YearsCode', 'YearsCodePro', 'CareerSat', 'JobSat', 'MgrIdiot', 'MgrMoney',
        'MgrWant',  'JobSeek', 'CompTotal', 'CompFreq', 'ConvertedComp', 'WorkWeekHrs', 'WorkRemote', 'WorkLoc', 'ImpSyn',
        'PurchaseWhat', 'LanguageWorkedWith', 'LanguageDesireNextYear', 'DatabaseWorkedWith', 'DatabaseDesireNextYear', 
        'OpSys', 'DevEnviron', 'Gender', 'Age', 'Dependents'
       ]]

## What we want to predict
y = df[['MoreThanOneYear']]

X.shape
```




    (65442, 34)




```python
%%time

# Preparing years coding
for feature in ['YearsCode', 'YearsCodePro']:
    X[feature] = X[feature].fillna(value=-1)
    X.loc[:,feature] = X[feature].apply(transform_coding_experience)
    
# Filling numeric columns with the mean
num_features = X.select_dtypes(include=['float', 'int']).columns
X[num_features] = fillna_with_mean(X[num_features])

# Adding new features
X.loc[:,'differentRoles'] = X['DevType'].apply(add_number_of_different_roles)
        
# Encoding categorical variables
cat_features = X.select_dtypes(include=['object']).columns

# Categorical features transformed based on popularity
transformed_features = ['DevType', 'LanguageWorkedWith', 'DatabaseWorkedWith', 
                        'LanguageDesireNextYear', 'DatabaseDesireNextYear', 'DevEnviron']

for feature in transformed_features:
    X[feature] = fillna_with_none(X[[feature]])

    # Preparing dev type (Keep only the most popular one)
    f_popularity = feature_popularity(feature, X)
    X[feature] = X.apply(lambda r: transform_feature_on_popularity(r[feature], f_popularity), axis=1)
    
X = encode_categorical_features(X, cat_features)
X.shape
```

    CPU times: user 9.09 s, sys: 1.62 s, total: 10.7 s
    Wall time: 10.8 s





    (65442, 861)




```python
y.shape
```




    (65442, 1)



### Finding the optimal model


```python
%%time 

# Defining threshold to cut features off
cutoffs = [10000, 5000, 3500, 2500, 1000, 100, 50, 30, 25]

# This function will run many experiments to find optmial model and plot the results
accuracy_train_scores, accuracy_test_scores, classifier, X_test, y_test = find_optimal_classifier(X, y, cutoffs, 
                                                                    n_estimators = 500, test_size = .30, 
                                                                    random_state = 42, plot = True)
```


    
![png](years_current_job_prediction_files/years_current_job_prediction_15_0.png)
    


    CPU times: user 12min 31s, sys: 6.04 s, total: 12min 37s
    Wall time: 12min 42s



```python
## Number of features for the optimal model
classifier.n_features_
```




    135



As we can observe in the plot above, the optimal number of features is **135** and the model accuracy in the test set is **~73%**.

### Analyzing feature importance

In this analysis, more important than accuracy is to understand the importance of each feature to predict whether developers will stay more than 1 year in their jobs.


```python
Xf = X_test.copy()
```


```python
Xf['pred'] = classifier.predict(X_test)
Xf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YearsCode</th>
      <th>YearsCodePro</th>
      <th>CompTotal</th>
      <th>ConvertedComp</th>
      <th>WorkWeekHrs</th>
      <th>Age</th>
      <th>differentRoles</th>
      <th>Hobbyist_Yes</th>
      <th>OpenSourcer_Less than once per year</th>
      <th>OpenSourcer_Never</th>
      <th>...</th>
      <th>DevEnviron_Atom</th>
      <th>DevEnviron_IntelliJ</th>
      <th>DevEnviron_Notepad++</th>
      <th>DevEnviron_Sublime Text</th>
      <th>DevEnviron_Vim</th>
      <th>DevEnviron_Visual Studio</th>
      <th>DevEnviron_Visual Studio Code</th>
      <th>Gender_Woman</th>
      <th>Dependents_Yes</th>
      <th>pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25100</th>
      <td>1.0</td>
      <td>0.5</td>
      <td>22000.0</td>
      <td>25206.0</td>
      <td>40.0</td>
      <td>27.0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>67254</th>
      <td>6.0</td>
      <td>4.0</td>
      <td>75000.0</td>
      <td>75000.0</td>
      <td>40.0</td>
      <td>23.0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>69193</th>
      <td>12.0</td>
      <td>8.0</td>
      <td>70000.0</td>
      <td>70000.0</td>
      <td>56.0</td>
      <td>37.0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>59573</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>6000.0</td>
      <td>17352.0</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29780</th>
      <td>36.0</td>
      <td>30.0</td>
      <td>57000.0</td>
      <td>65308.0</td>
      <td>40.0</td>
      <td>49.0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 136 columns</p>
</div>




```python
shap.initjs() 
explainer = shap.TreeExplainer(classifier)
```

Picking randomly a sample instance classified as **1** which means the developer is more than 1 year in the current job:


```python
shap.initjs() 
choosen_instance = Xf[Xf.pred == 1].sample(1, random_state=42).drop('pred', axis=1)
shap_values = explainer.shap_values(choosen_instance)
shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)
```


![png](years_current_job_prediction_files/shap_value1.png)


We have shapley values for the features. Feature values in pink increases the prediction. Size of the bar is the magnitude of the feature's effect. Feature values in blue cause to decrease the prediction. In this sample, we can observe the following features contributed more to have a positive classification (class = 1):

* `JobSeek: I am not interested in new job opportunities = 0`
* `YearsCode = 14`
* `YearsCodePro = 5`
* `differentRoles = 7`

In this case, the candidate is more than 1 year in the current job when these conditions happen:
* Not looking for a new job
* Have good experience (at least 5 coding as a professional)
* Assume different job roles 

Similarly, we pick randomly a sample instance classified as **0** which means the developer is less than 1 year in the current job:


```python
shap.initjs() 
choosen_instance = Xf[Xf.pred == 0].sample(1, random_state=42).drop('pred', axis=1)
shap_values = explainer.shap_values(choosen_instance)
shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)
```

![png](years_current_job_prediction_files/shap_value2.png)

In this case, these features contributed more to have a negative classification (class = 0):

* `YearsCodePro = -1`
* `YearsCode = 6`
* `Student_Yes, full-time = 1`
* `Age = 25`

In this case, the candidate is less than 1 year in the current job when these conditions happen:
* Low professional experience
* It is a full time student
* Young age

#### Analyzing features impact on a higher scale


```python
%%time
sample = X_test.sample(100, random_state=42)

shap_values = explainer.shap_values(sample) 
shap.summary_plot(shap_values, sample)
```


    
![png](years_current_job_prediction_files/years_current_job_prediction_30_0.png)
    


    CPU times: user 17min 18s, sys: 5.81 s, total: 17min 24s
    Wall time: 17min 25s


By analyzing a sample of 100 randomly picked instances, we can confirm what we had already observed. The current developer coding experience, the fact the developer is not looking for a new job and the person's age are the major factors to determine whether developer will stay longer than 1 year in the current job or not.