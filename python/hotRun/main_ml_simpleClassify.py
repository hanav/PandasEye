import os.path
import pandas as pd
from mlPipeline import Pipeline

featureFilePath = '/Users/icce/Dropbox/_thesis_framework/_scripts_hoy/r_icmi/results_features_eye.csv'
featuresFile = 'results_features_eye.csv'
featureDf = pd.read_csv(featureFilePath, sep=',')

results = pd.DataFrame([])

pipe = Pipeline(featureDf)
labelColumn = 16
pipe.LoadFeaturesJournal(labelColumn)
pipe.ImputeNormalize()
# pipe.SelectFeatures() - todo: opravdu moc nefunguje, produkuje to divny pole s nulama

# pipe.UpsampleMinority() -todo: doinstalovat SMOTE, unresolved references

pipe.GridSearchFull(("Nested:" + featuresFile + "Feature selection + Smote"))


#pipe.CrossvalidationUsers(featuresFile+"Logo.Smote.35.gamma.auto.class.weights")

print("Split into training and testing set")
print("Train")
print("Crossvalidate on train")
print("Performance")


results = pipe.outputResults

resultsOut =  '/Users/icce/Dropbox/_thesis_framework/_scripts_hoy/r_icmi/results_features_eye_predictions.csv'
with open(resultsOut, 'a') as f:
    results.to_csv(f, header=False, index=False)

print("***********************************\n All good\n")
exit(0)
