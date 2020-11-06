# stackoverflow-analytics

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
The code should run with no issues using Python versions 3.*. When using `pip`, please install the dependencies via
requirements.txt file:

```pip install -r requirements.txt```

Then, enable Javascript rendering in jupyter by running:

```jupyter nbextension enable --py widgetsnbextension```

## Project Motivation<a name="motivation"></a>

For this project, I have used [Stack Overflow data from 2019](https://insights.stackoverflow.com/survey) to better understand 
what factors lead developers to stay longer in their current jobs. I have broken the analysis down in the 
following questions: 

1. Does the job role lead developers to stay longer in the job ?
2. Does wishing to become a manager leads developers to stay longer in a job ?
3. What other factors lead developers to stay longer in their jobs ?

All results are presented in my Medium post: [What makes you stay many years in a job ?](https://besson.medium.com/what-makes-you-stay-many-years-in-a-job-2c2cc34db17d)

## File Descriptions <a name="files"></a>

* **exploratory_analysis.ipynb**: notebook to tackle questions 1 and 2.
* **years_current_job_prediction.ipynb**: notebook to train a model and tackle question 3. 
* **reports/**: notebook output in markdown format. *Some Javascript visualizations are not visible in the github ipynb rendering*
* **data_helper.py**: python functions used in the notebooks

## Results<a name="results"></a>

The main findings of the code can be found at the Medium post available [here](https://besson.medium.com/what-makes-you-stay-many-years-in-a-job-2c2cc34db17d).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Stack Overflow for the data.  You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/mchirico/stack-overflow-developer-survey-results-2019).  Otherwise, feel free to use the code here as you would like! 
