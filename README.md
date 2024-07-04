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

Required libraries and some settings for this section are:

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

As we want to check the data to have a general opinion about it, we create and use a function called `check_df(dataframe, head=5, tail=5)` that prints the referred functions:


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

We create another plot function called `plot_num_summary(dataframe)` to see the whole summary of numerical columns due to the high quantity of them:

![download](https://github.com/Trigenaris/Prediction_of_Baseball_Player_Salaries_via_Machine_Learning_Algorithms/assets/122381599/9e5e3b03-9670-4abc-81b1-e526062db078)

## Target Analysis

We create another function called `target_summary_with_cat(dataframe, target, categorical_col)` to examine the target by categorical features.

For instance *League Feature*

################ Salary --> League #################

League     |    Target Mean
------------|---------------|
A    |      541.9995
N    |      529.1175

## Correlation Analysis

To analyze correlations between numerical columns we create a function called `correlated_cols(dataframe)`:

![download](https://github.com/Trigenaris/Prediction_of_Baseball_Player_Salaries_via_Machine_Learning_Algorithms/assets/122381599/b3dd574f-0d4d-4c17-bdfa-f88cbca20938)

## Missing Value Analysis

We check the data to designate the missing values in it, `dataframe.isnull().sum()`:

Feature | Missing Value |
--------|-------------|
AtBat      |   0
Hits       |   0
HmRun      |   0
Runs       |   0
RBI        |   0
Walks      |   0
Years      |   0
CAtBat     |   0
CHits      |   0
CHmRun     |   0
CRuns      |   0
CRBI       |   0
CWalks     |   0
League     |   0
Division   |   0
PutOuts    |   0
Assists    |   0
Errors     |   0
***Salary***     |  ***59***
NewLeague  |   0

dtype: int64

#

Missing Values Table:

target    |    n_miss  | ratio |
----------|------------|-------|
Salary    |  59 | 18.3200 |

We fill the missing values with the mean value of the target which is `425.0000`

## Encoding

To convert categorical features into boolean we create a function called `one_hot_encoding(dataframe, drop_first=True)`:

League_N | Division_W | NewLeague_N |
---------|------------|-------------|
False	| False | False
True | True | True
False |	True | False
True | False | True
True | False | True

## Random Forest: Machine Learning Algorithm

At last, we create our model and see the results:

******************** RF Model Results ********************

MSE Train:  110.665

MSE Test:  292.432

RMSE Train:  70.707

RMSE Test:  194.495

R2 Train:  0.924

R2 Test:  0.555

Cross Validate MSE Score: 87996.550

Cross Validate RMSE Score: 291.386

![download](https://github.com/Trigenaris/Prediction_of_Baseball_Player_Salaries_via_Machine_Learning_Algorithms/assets/122381599/9e235ff1-38de-4321-a188-b5c2151772b0)

## Loading a Base Model and Prediction

Via **joblib** we can save and/or load our model:

    def load_model(pklfile):
      model_disc = joblib.load(pklfile)
      return model_disc
      
    model_disc = load_model("rf_model.pkl")

________

Now we can make predictions with our model:
    
    X = df.drop("Salary", axis=1)
    x = X.sample(1).values.tolist()

    model_disc.predict(pd.DataFrame(X))[0]
    331.68

________

    sample2 = [250, 70, 15, 40, 100, 30, 8, 1800, 500, 80, 220, 290, 140, 700, 90, 8, False, True, True]
    
    model_disc.predict(pd.DataFrame(sample2).T)[0]
    620.0307300000001

## Model Tuning

To have better predictions we tune our model and the results are:

Fitting 5 folds for each of 45 candidates, totalling 225 fits

******************** RF Model Results ********************

MSE Train:  195.900

MSE Test:  210.187

RMSE Train:  140.532

RMSE Test:  145.399

R2 Train:  0.761

R2 Test:  0.770

Cross Validate MSE Score: 84230.838

Cross Validate RMSE Score: 285.428

![download](https://github.com/Trigenaris/Prediction_of_Baseball_Player_Salaries_via_Machine_Learning_Algorithms/assets/122381599/52d46d2f-35b3-4801-9757-dae5d62eb1d4)

## Loading a Tuned Model and Prediction

    def load_model(pklfile):
      model_disc = joblib.load(pklfile)
      return model_disc
      
    model_disc = load_model("rf_model_tuned.pkl")

______

    X = df.drop("Salary", axis=1)
    x = X.sample(1).values.tolist()
    
    model_disc.predict(pd.DataFrame(X))[0]
    186.33951137202502

______

    sample2 = [250, 70, 15, 40, 100, 30, 8, 1800, 500, 80, 220, 290, 140, 700, 90, 8, False, True, True]
    
    model_disc.predict(pd.DataFrame(sample2).T)[0]
    560.5904032788515


