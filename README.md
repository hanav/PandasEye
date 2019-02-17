# PandasEye

PandasEye consists of a set of tools developed for action prediction and affect recognition. 

## Getting started

## Built with
* Python 2.7
* Pandas 
* Numpy
* Scipy
* Scikit-learn
* Imblearn

## Prediction pipeline

#### Sequencing for action prediction
Scripts are in [parseFeatures](_scripts_preprocessing/parseFeatures).

| Parameters    | Description   | Example  |
| ------------- |-------------|-----|
| prefix        | right-aligned | 2 (2 fixations prior to the action) |
| suffix        | centered      | 2 (1 fixation after the action) |

### Example run

python main_new_featureExtraction_WTP.py


### Feature engineering for action prediction papers

### Feature engineering for affect recognition

### Data preprocessing

* 
# - feature scaling
# - imputing missing values
# - splitting to a training and testing set (stratified shuffling)
# - splitting to a training and testing set (person specified)
# - feature selection (percentile)
# - downsampling the majority class
# - upsampling the minority class


### Machine learning experiments

### Outcome measures
kFold crossvalidation and leaveOnePersonOut. - for each fold. Two last lines contains mean 
and standard deviation of following outcome measures. 
Each measure is calculated for training and testing folds.

| Measure | Description|
|-------------| --------------|
| [train/test]_ACC | Accuracy |
| [train/test]_AUC| Area under the ROC curve|
| [train/test]_kappa | Cohen's kappa| 
| [train/test]_positive_acc| Accuracy of positive class|
| [train/test]_negative_acc|Accuracy of negative class|
| [train/test]_f1 | F1-measure|
| [train/test]_precision| |
| [train/test]_recall| |
|no_rows| Number of rows|
|no_features| Number of features|
|no_testingSamples| Number of samples in the testing fold|
|no_positiveTest| Number of samples in the positive class (testing fold)|
|no_negativeTest| Number of samples in the negative class (testing fold)|

## Example run

python PandasEye/PredictionPipeline/3_ml_classify_loocv_longterm_RF.py

-i ExampleData/features_longterm_last5min_stats_omitShort_ValenceArousal_npNan.csv

-o Results/

-m "HOT arousal: last 5 minutes - RandomGridSearch"

-d "HOT_LAST_SMOTE"

## References
1. Bednarik, R., Vrzakova, H., Hradis, M.: What you want to do next: A novel approach for intent prediction in gaze-based interaction. In proceedings of ETRA'12, pp. 83-90.
2. Fast and Comprehensive Extension to Intention Prediction from Gaze. In Interacting with Smart Objects, Intelligent User Interfaces(IUI '13). ACM, 2013
3. Vrzakova, H. and Bednarik, R. Quiet Eye Affects Action Detection from Gaze more than Context Length In Proceedings of User Modeling, Adaptation and Personalization (UMAP). Springer, 2015
4. Vrzakova, H., Begel, A., Mehtatalo, L., and Bednarik, R.: Affect Recognition in Code Review:An in-situ biometric study of reviewer’s affect. In revision in Journal of Systems and Software, 2019

## Acknowledgments
Dedicated to my feline friend Ofelia who rolled over the keyboard many times and erased my work.
