<h1> 
  Prediction: Baseball Player Salaries via Random Forest
</h1>

## Business Problem

In this section, we plan to predict baseball players' salaries.

## Dataset Story

This dataset was originally taken from the StatLib library which is maintained at Carnegie Mellon University. This is part of the data that was used in the 1988 ASA Graphics Section Poster Session. The salary data were originally from Sports Illustrated, April 20, 1987. The 1986 and career statistics were obtained from The 1987 Baseball Encyclopedia Update published by Collier Books, Macmillan Publishing Company, New York.

Format
* A data frame with 322 observations of major league players on the following 20 variables.
* AtBat Number of times at bat in 1986
* Hits Number of hits in 1986
* HmRun Number of home runs in 1986
* Runs Number of runs in 1986
* RBI Number of runs batted in in 1986
* Walks Number of walks in 1986
* Years Number of years in the major leagues
* CAtBat Number of times at bat during his career
* CHits Number of hits during his career
* CHmRun Number of home runs during his career
* CRuns Number of runs during his career
* CRBI Number of runs batted in during his career
* CWalks Number of walks during his career
* League A factor with levels A and N indicating player’s league at the end of 1986
* Division A factor with levels E and W indicating the player’s division at the end of 1986
* PutOuts Number of putouts in 1986
* Assists Number of assists in 1986
* Errors Number of errors in 1986
* Salary 1987 annual salary on opening day in thousands of dollars
NewLeague A factor with levels A and N indicating player’s league at the beginning of 1987

## Necessary Libraries

Required libraries, and some settings for this section are:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

import warnings
warnings.filterwarnings("ignore")
```

## Importing the Dataset

First, we import the dataset `Hitters.csv` into the pandas DataFrame.

## General Information About the Dataset

### Checking the Data Frame

As we want to check the data to have a general opinion about it, we create and use a function called `check_df(dataframe, head=5, tail=5)` that prints referred functions:


    dataframe.head(head)
    
    dataframe.tail(tail)
    
    dataframe.shape
    
    dataframe.dtypes
    
    dataframe.size
    
    dataframe.isnull().sum()
    
    dataframe.describe([0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1]).T

### Defining the Columns

After checking the data frame, we need to define and separate columns as **categorical** and **numerical**. We define a function called `grab_col_names` for separation that benefits from multiple list comprehensions as follows:

    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ['category', 'object', 'bool']]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ['uint8', 'int64', 'int32', 'float64']]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and str(df[col].dtypes) in ['object', 'category']]
    cat_cols = cat_cols + num_but_cat
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ['uint8', 'int64', 'float64']]
    num_cols = [col for col in num_cols if col not in cat_cols]

`cat_th` and `car_th` are the threshold parameters to decide the column type.

**Categorical Columns:**

* League
* Division
* NewLeague

**Numerical Columns:**

* Atbat
* Hits
* HmRun
* Runs
* RBI
* Walks
* Years
* CAtBat
* CHits
* CHmRun
* CRuns
* CRBI
* CWalks
* PutOuts
* Assists
* Errors
* Salary

### Summarization and Visualization of the Categorical and Numerical Columns

To summarize and visualize the referred column we create two other functions called `cat_summary` and `num_summary`.

For example, categorical column **League**:

############### League ###############

League  | Ratio |
------|------|
A     |     175 54.3478 |
N     |     147 45.6522 |

![download](https://github.com/Trigenaris/Prediction_of_Baseball_Player_Salaries_via_Machine_Learning_Algorithms/assets/122381599/8f81947d-65f9-4c31-8cd0-f97d76b75352)

Another example, numerical column **Salary**:

############### Salary ###############

Process | Result |
-------|-----------|
count  |  263.0000 |
mean   |  535.9259 |
std    |  451.1187 |
min    |   67.5000 |
1%     |   70.0000 |
5%     |   86.6000 |
10%    |  100.0000 |
20%    |  155.0000 |
30%    |  221.0000 |
40%    |  300.0000 |
50%    |  425.0000 |
60%    |  538.0000 |
70%    |  700.0000 |
80%    |  809.0000 |
90%    | 1048.6666 |
95%    | 1346.0000 |
99%    | 2032.8865 |
max    | 2460.0000 |

Name: Salary, dtype: float64

![download](https://github.com/Trigenaris/Prediction_of_Baseball_Player_Salaries_via_Machine_Learning_Algorithms/assets/122381599/5bec19db-cd0d-4d5c-9464-06c04a956d49)

With the help of a for loop we apply these functions to all columns in the data frame.

We create another plot function called `plot_num_summary(dataframe)` to see whole summary of numerical columns due to high quantity of them:

![download](https://github.com/Trigenaris/Prediction_of_Baseball_Player_Salaries_via_Machine_Learning_Algorithms/assets/122381599/9e5e3b03-9670-4abc-81b1-e526062db078)

## Target Analysis


