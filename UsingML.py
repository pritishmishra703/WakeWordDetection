####### IMPORTS #############
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from PCM.PCM import plot_confusion_matrix
import pickle

##### Loading saved csv ##############
df = pd.read_pickle("final_audio_data_csv/audio_data.csv")

####### Making our data training-ready
X = df["feature"].values
X = np.concatenate(X, axis=0).reshape(len(X), 40)

y = np.array(df["class_label"].tolist())

####### train test split ############
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##### Training ############

logit_reg = LogisticRegression(max_iter=10000)
logit_reg.fit(X_train, y_train)
score = logit_reg.score(X_test, y_test)
print("Model Score: \n")
print(score)

#### Evaluating our model ###########
print("Model Classification Report: \n")

y_pred = logit_reg.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
plot_confusion_matrix(cm, classes=["Does not have WW", "Has WW"])

#### Save the model
pickle.dump(logit_reg, open('saved_model/WWD_ML.txt', 'wb'))

'''
To load the model again run this:

>>> model = pickle.load(open('saved_model/WWD_ML.txt', 'rb'))
>>> model.predict(<-- your matrix -->) # to predict
'''
