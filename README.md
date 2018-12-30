# credible-clients

This repository contains a small pre-task for potential ML team
members for [UBC Launch Pad](https://www.ubclaunchpad.com).


## Overview

The dataset bundled in this repository contains information about credit card
bill payments, courtesy of the [UCI Machine Learning Repository][UCI]. Your task
is to train a model on this data to predict whether or not a customer will default
on their next bill payment.

Most of the work should be done in [**`model.py`**](model.py). It contains a
barebones model class; your job is to implement the `fit` and `predict` methods,
in whatever way you want (feel free to import any libraries you wish). You can
look at [**`main.py`**](main.py) to see how these methods will be called. Don't
worry about getting "good" results (this dataset is _very tough_ to predict on)
— treat this as an exploratory task!

To run this code, you'll need Python and three libraries: [NumPy], [SciPy],
and [`scikit-learn`]. After invoking **`python main.py`** from your shell of
choice, you should see the model accuracy printed: approximately 50% if you
haven't changed anything, since the provided model predicts completely randomly.

## Instructions

Here are the things you should do:

1. Fork this repo, so we can see your code!
2. Install the required libraries using `pip install -r requirements.txt` (if needed).
3. Ensure you see the model's accuracy/precision/recall scores printed when running `python main.py`.
4. Replace the placeholder code in [`model.py`](model.py) with your own model.
5. Fill in the "write-up" section below in your forked copy of the README.

_Good luck, and have fun with this_! :rocket:


## Write-up

So, to begin the project I first copied the code over to a python notebook to visualize the dataset, as well as understand what the prewritten code split the database. For that, I used numpy, pandas, seaborn and matplotlib. I also decided to add a linear regression model just to check if the program ran without any errors before proceeding.  

Since this was a classification problem, I decided to choose the following models to explore:
* Gradient Boosting Classifier from SciKit Ensemble
* Random Forest Classifier from SciKit Ensemble
* AdaBoost Classifier from SciKit Ensemble
* Catboost Classifier from Yandex (Proprietary Gradient Boosting Classifier, used as reference)  

I chose these models over estimators because ensembles combine individual estimators anyways, and I felt that these would provide better results. The reason for choosing so many models was simply because I had no idea which one of these would work best, so I decided to try all of them!

Inspecting the data revealed no reason for cleaning up, or any missing values. I also decided to keep all of the data columns for training.

I ran all of the aforementioned models, and plotted their 'Feature Importance' to see which data was weighted higher, where I ran into my first issue. My Gradient Boosting Classifier was ranking all columns except 'Pay_0' as equal. I used the Catboost classifier's feature importance and compared the two graphs. This problem was solved after changing the model's learning rates and estimators, though I still don't fully understand why that occurred in the first place. Regardless, I moved on to plot the rest of the curves.

I saw varying trends across each model, but  'Sex', 'Marital Status', and Payments '4,5,6' in different orders were consistently weighted low. I think this makes sense because 'Sex' and 'Marital Status' would have little correlation with defaulting, and as will the later payments because the likelihood of defaulting would be higher if the client is default on previous payments.

In the end, I did some parameter tuning for each model to find the maximum scores possible. These scores are listed below:

| Model | Accuracy | Precision | Recall |
| :---: | -------- | --------- | ------ |
|Gradient Boosting| 82.067|66.023|35.710|
|Random Forest|81.773|65.366|33.989|
|AdaBoost|81.653|65.437|32.968|
|Catboost|82.067|66.133|35.526|

For the purpose of this submission, I decided to ignore Catboost since the library was not provided as a prerequisite and it does take about ~10 minutes to get results while the other models completed under a minute or two. From the remaining three, I chose to keep Gradient Boosting due to its slightly higher accuracy, precision rate and recall.

I think that these values can be improved by further testing and tuning of the models using methods like scikit's GridSearchCV. I have never used it before, and my attempt using it brought the metrics down, hence I decided not to mess with it until I understood it better. I also have not explored scoring using F1 or similar metrics for a similar reason, though I feel like it boils down to the goal of the algorithm - whether the user wishes to minimize FP or FN, and the model's performance can be evaluated subjectively based on context.


## Data Format

`X_train` and `X_test` contain data of the following form:

| Column(s) | Data |
| :-------: | ---- |
| 0         | Amount of credit given, in dollars |
| 1         | Gender (_1 = male, 2 = female_) |
| 2         | Education (_1 = graduate school; 2 = university; 3 = high school; 4 = others_) |
| 3         | Marital status (_1 = married; 2 = single; 3 = others_) |
| 4         | Age, in years |
| 5–10      | History of past payments over 6 months (_-1 = on-time; 1 = one month late; …_) |
| 11–16     | Amount of previous bill over 6 months, in dollars |
| 17–22     | Amount of previous payment over 6 months, in dollars |

`y_train` and `y_test` contain a `1` if the customer defaulted on their next
payment, and a `0` otherwise.


[UCI]: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
[NumPy]: http://www.numpy.org
[SciPy]: https://www.scipy.org
[`scikit-learn`]: http://scikit-learn.org
