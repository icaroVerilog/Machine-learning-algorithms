import pandas as PD
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

dataset = PD.read_csv("heart_failure_clinical_records_dataset.csv")

trainDataframe, validationDataframe = train_test_split(dataframe, test_size = 0.25, random_state = 1)