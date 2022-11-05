# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a logistic regression model using the default hyperparameters in scikit-learn version 
1.1.2

Default parameters were used to train this model with no hyper-parameter tuning. 

## Intended Use
This model should be used to predict if the salary of a person would be above or below 50K.

## Data Training & Evaluation
The data was obtained from publicly accessible data available from the Census Bureau.
The target class/label to be predicted are:
* Salary Equal to or Below 50K (<=50K)
* Salary Greater than or Above 50K (>50K)

The original dataset contained 32,561 rows with the following columns:
 * 0   age              32561 non-null  int64 
 * 1    workclass       32561 non-null  object
 * 2    fnlgt           32561 non-null  int64 
 * 3    education       32561 non-null  object
 * 4    education-num   32561 non-null  int64 
 * 5    marital-status  32561 non-null  object
 * 6    occupation      32561 non-null  object
 * 7    relationship    32561 non-null  object
 * 8    race            32561 non-null  object
 * 9    sex             32561 non-null  object
 * 10   capital-gain    32561 non-null  int64 
 * 11   capital-loss    32561 non-null  int64 
 * 12   hours-per-week  32561 non-null  int64 
 * 13   native-country  32561 non-null  object
 * 14   salary          32561 non-null  object

## Metrics
Metrics for this model include:
* Precision: 0.693844
* Recall: 0.255046
* f-beta: 0.372987

## Ethical Considerations
The accuracy performance of the model on some of the Native Countries is lower because we have less data for those countries. 
E.g., Hong, Cambodia, Holand-Netherlands. Be sure to be careful when predicting for people in these countries. 


## Caveats and Recommendations
Be sure to continue to build training data that includes Native Countries that we don't have a lot of data for, to build up accuracy in predictions for people in those countries.  
