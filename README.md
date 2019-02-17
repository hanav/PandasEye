# PandasEye

PandasEye consists of a set of tools developed for action prediction and affect recognition. 

## Built with
* Python 2.7
* Pandas 
* Numpy
* Scipy
* Scikit-learn
* Imblearn

## FeatureExtraction
Scripts contain methods employed in sequencing eye-tracking data in 8Puzzle (P1,P2,P3).
ExampleData provides anonymized raw eye-tracking and event data from the GazeAugmented condition.
Parameters prefix and suffix defines the start and the end of the sequence with respect to an action (mouse click).

| Inputs | Description |
| ---- | -----|
| AOI_codes.csv | Coordinates of AOIs  in 8Puzzles. |
| *_events.csv | Timestamps of mouse clicks. |
| *_PX.txt | Participant's combined eye-tracking data.|
| prefix | Start of the sequence (a number of fixations).|
| suffix | End of the sequence (a number of fixations).|

|Output| Description|
| --- | --- |
| Results | Folder automatically created in the FeatureExtraction. |
| features_x_y.csv| Output feature set. (x = prefix, y = suffix). |

### Example run
python main_new_featureExtraction.py


## PredictionPipeline
Scripts comprises three steps (preprocessing, feature engineering, and machine
learning experiments) in the prediction pipeline, employed in P5. Since raw data
are proprietary, only the resulting and anonymized feature set is provided to run 
an example machine learning process. 

### 3_ml_classify_loocv_longterm_RF.py
| Inputs | Description |
| ---- | -----|
| -i ExampleData/*.csv | Example feature set with labels. |
| -o  Results| Directory to save the prediction outputs. |
| -m Message | Title of the experiment. |
| - d Tag | ID of the experiment, propagated to the output files.|

| Output | Description |
| ---- | -----|
| Tag_Arousal_YYYYMMDD_HHMM| Folder with performence outcomes of arousal recognition.|
| Tag_Valence_YYYYMMDD_HHMM| Folder with performence outcomes of arousal recognition.|
| ALL_* | Results achieved using the entire feature set.|
| GAZE_* | Results achieved using eye-tracking features.|
| GSR_* | Results received using GSR features.|
|TM_* | Results received using TouchMouse features.|
|*\_stratifiedkfold\_| Performance received during kFold crossvalidation.|
|*\_loocv\_| Performance received during leave-one-person-out crossvalidation. |
| \_results| Output file with train and test performance. |
| \_importance | Feature importance estimated by Random Forest.|

### Performance metrics
Following metrics are computed in each iteration of kFold crossvalidation and leaveOnePersonOut. Two last lines contains mean 
and standard deviation. Each measure is calculated for training and testing folds.

| Measure | Description|
|-------------| --------------|
| [train/test]_ACC | Accuracy |
| [train/test]_AUC| Area under the ROC curve|
| [train/test]_kappa | Cohen's kappa| 
| [train/test]_positive_acc| Accuracy of positive class|
| [train/test]_negative_acc|Accuracy of negative class|
| [train/test]_f1 | F1-measure|
| [train/test]_precision| Precision|
| [train/test]_recall| Recall|
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
