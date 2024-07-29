# Model Card

## Model Details
Martin Frankenstein created the model. It is a classification model using Random Forest Classifier from sklearn.ensemble.

## Intended Use
This model should be used to predict a salary of less than 50k and greater than or equal to 50k.

## Training Data
The training data was obtained from https://archive.ics.uci.edu/dataset/20/census+income.

Data is from the 1994 Census database extracted by Barry Becker. Training size is 20% of this dataset.

Target is salary from the following features:
* age
* workclass
* fnlgt
* education
* education-num
* marital-status
* occupation
* relationship
* race
* sex
* capital-gain
* capital-loss
* hours-per-week
* native-country

## Evaluation Data
20% of the original dataset is used to evauluate the model.

## Metrics
The model was evaluated using Precision, Recall, and F1 score.
* Precision: 0.7222
* Recall: 0.6194
* F1 0.6669

## Ethical Considerations
The use of this model in decision making should be considered carefully as this is a classification model that considers groupings of individuals and may not represent individuals fairly.

## Caveats and Recommendations
This model was created as a requirement for a course project. Further training should be done to limit potential data bias and retraining with updated is recommended to make the model relevant to modern real-life scenarios.