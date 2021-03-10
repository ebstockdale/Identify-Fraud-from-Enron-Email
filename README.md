

# Introduction

The Enron scandal was one of the largest corporate fraud cases of all time. Enron went from one of the largest Fortune 500 companies to bankruptcy seemingly overnight. Massive fraud was uncovered and years later, a significant amount of confidential information entered the public domain in the form of the now famous Enron corpus.

Using machine learning techniques, the aim of this project will be to create a “person of interest” (POI) algorithm capable of identifying guilty parties.

## Environment
Python 2.7

## Data Exploration

The goal of the project is to build a predictive model using machine learning algorithms to identify a "person of interest" (POI) using financial and email data included in the Enron corpus. A POI is someone who was indicted for fraud, settled with the government, or testified in exchange for immunity.

## Data Features

The Enron dataset contains 146 Enron employees to investigate. Each sample in this dictionary containing 21 features and 18 people from this dataset are labeled as POI.

-Financial Features: salary, deferral_payments, total_payments, loan_advances, bonus, restricted_stock_deferred, deferred_income, total_stock_value, expenses, exercised_stock_options, other, long_term_incentive, restricted_stock, director_fees.
-Email Features: to_messages, email_address, from_poi_to_this_person, from_messages, from_this_person_to_poi, shared_receipt_with_poi.
-POI Label: poi.

## Outliers

In my analysis, two outliers were discovered.

-Total: Using a scatter plot matrix visualization, I found that Total represents the sum of all salaries and was skewing the data quite a bit.

![Figure 1](https://github.com/ebstockdale/Identify-Fraud-from-Enron-Email/blob/main/salary%20with%20Total.png)

After removing this outlier, things look much better.

![Figure 2](https://github.com/ebstockdale/Identify-Fraud-from-Enron-Email/blob/main/salary%20without%20Total.png)


-The Travel Agency in the Park: This also appears to be another outlier and could possibly be a data entry error.

Additional features such as director_fees,  restricted_stock_deferred and loan_advances contained many NaN values and were thus removed. 


## New Feature Creation: 

The ratio of number of messages from a person to a poi was seen as a value indicator. 

New features
ratio_from_person_to_poi
ratio_from_poi_to_person

Removed features
from_messages
to_messages
from_poi_to_this_person

## Feature Scaling:

As some features do not use the same ranges, the min_max_scaler from scikit-learn was used to normalize the range of features in the dataset. 

## Feature Selection

SelectKBest was used to select the top 10 best features shown below. 

poi
salary
total_payments
exercised_stock_options
bonus
restricted_stock
shared_receipt_with_poi
total_stock_value
deferred_income
long_term_incentive

## Splitting the Data

Data was split into training and testing datasets.

 

## Evaluation of classifier algorithms

Classifiers chosen for evaluation were Random Forest, Decision Tree, Ada Boost and Gausian NB. Random Forest and Ada Boost were selected for parameter tuning. 





#### Random Forest

parametrer tune 1

parameter tune 2

parameter tune 3 (best)

#### Decision Tree

#### Ada Boost

parameter tune 1

parameter tune 2


parameter tune 3 (best)

#### Gassian NB

## Parameter Tuning

In machine learning, each parameter is set to a default value which can be changed or tuned.
Every real-world data set is different and needs to be worked on differently. If the same model with a strict parameter set is applied on every data, a good result cannot be expected uniformly in all cases. For that reason, these parameters must be adjusted in such a manner that the best predictions can be achieved by the model for every individual data set.
This technique of adjusting the elements which control the behavior of a given model is called parameter tuning.

Parameter tuning was attempted on several algorithms. I noted the changed of each parameter tune  and arrived at the best result for Ada boost and Random Forest classifiers.

## Summary of Results

Algorithms	

Decision Tree	Accuracy 0.84347 Precision 0.41967 Recall 0.45450
Gassian NB	Accuracy 0.83613 Precision 0.36639 Recall 0.31400

Random Forest 	Accuracy 0.86007 Precision 0.43629 Recall 0.16950
Random Forest 	Accuracy 0.85868 Precision 0.42772 Recall 0.17900
Random Forest 	Accuracy 0.84993 Precision 0.39190 Recall 0.22750

Ada Boost 	Accuracy 0.85773 Precision 0.44029 Recall 0.24700
Ada Boost 	Accuracy 0.85527 Precision 0.42276 Recall 0.23400
Ada Boost 	Accuracy 0.85687 Precision 0.44885 Recall 0.32250

## Conclusion

My two main evaluation metrics for this project were precision and recall. Using those metrics, I found that Decision Tree with precision of 0.41967 and recall of 0.45450 to be the best choice. 

## References
- Enron scandal, Wikipedia - https://en.wikipedia.org/wiki/Enron_scandal
- Feature normalization - http://stats.stackexchange.com/questions/77350/perform-feature-normalization-before-or-within-model-validation
- Test set vs validation set?, Cross Validated - http://stats.stackexchange.com/questions/19048/what-is-the-difference-between-test-set-and-validation-set
- Sci-kit and Regression Summary, Stack Overflow - http://stackoverflow.com/questions/26319259/sci-kit-and-regression-summary 
