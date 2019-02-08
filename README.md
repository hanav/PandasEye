# PandasEye

PandasEye consists of a set of tools developed for action prediction and affect recognition. 

## Getting started

## Prerequisities

## Built with
* Python 2.7
* Pandas 
* Numpy
* Scikit-learn

## Prediction pipeline

### Data preprocessing

#### Sequencing for action prediction
Scripts are in [parseFeatures](_scripts_preprocessing/parseFeatures).

| Parameters    | Description   | Example  |
| ------------- |:-------------:| -----:|
| prefix        | right-aligned | 2 (2 fixations prior to the action) |
| suffix        | centered      | 2 (1 fixation after the action) |


### Feature engineering for action prediction papers

### Feature engineering for affect recognition

### Machine learning experiments

## References
1. Bednarik, R., Vrzakova, H., Hradis, M.: What you want to do next: A novel approach for intent prediction in gaze-based interaction. In proceedings of ETRA'12, pp. 83-90.
2. Fast and Comprehensive Extension to Intention Prediction from Gaze. In Interacting with Smart Objects, Intelligent User Interfaces(IUI '13). ACM, 2013
3. Vrzakova, H. and Bednarik, R. Quiet Eye Affects Action Detection from Gaze more than Context Length In Proceedings of User Modeling, Adaptation and Personalization (UMAP). Springer, 2015
4. Vrzakova, H., Begel, A., Mehtatalo, L., and Bednarik, R.: Affect Recognition in Code Review:An in-situ biometric study of reviewer’s affect. In revision in Journal of Systems and Software, 2019

## Acknowledgments
* Dedicated to my feline friend Ofelia. 
