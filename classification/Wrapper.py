import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import model
import preprocess

# train model...
input_dataset = '/home/mvanessa/pastprojects/finalcode/Augmented_Feat.csv'
# input_dataset = '/Users/michellevanessa/Desktop/automatic-text-scoring-master/Final Code and Data/Augmented_Feat.csv'

df = preprocess.cleaning_dataset(input_dataset)
# df = df.iloc[:2500, :]

X = df[['Ref Answer', 'Answer']]
Y = pd.DataFrame(df['ans_grade'])
Y_np = Y.to_numpy()
y = np.zeros((Y_np.shape[0], 3))
for index, row in Y.iterrows():
    val = 1
    if row['ans_grade'] <= 1:
        val = 0
    elif row['ans_grade'] >= 4:
        val = 2
    y[index, val] = 1

## Min Max Scaling of the features used for feature engineering
x = pd.DataFrame(df['Length Answer'])
scaler_x = MinMaxScaler()
scaler_x.fit(x)
x = scaler_x.transform(x)
X['Length Answer'] = x

x = pd.DataFrame(df['Len Ref By Ans'])
scaler_x2 = MinMaxScaler()
scaler_x2.fit(x)
x = scaler_x2.transform(x)
X['Len Ref By Ans'] = x

x = pd.DataFrame(df['Words Answer'])
scaler_x3 = MinMaxScaler()
scaler_x3.fit(x)
x = scaler_x3.transform(x)
X['Words Answer'] = x

x = pd.DataFrame(df['Length Ref Answer'])
scaler_x4 = MinMaxScaler()
scaler_x4.fit(x)
x = scaler_x4.transform(x)
X['Length Ref Answer'] = x

x = pd.DataFrame(df['Unique Words Answer'])
scaler_x5 = MinMaxScaler()
scaler_x5.fit(x)
x = scaler_x5.transform(x)
X['Unique Words Answer'] = x

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=101)

df = X_train
#
df_test = X_test

# Train the model
test, train_model, tokenizer = model.train_dataset_model(df, y_train)

# Obtain test results by training on the test dataset dataframe
test_results = model.test_dataset_model(df_test, train_model, tokenizer)

## Processing of the test result to obtain a uniform format and then inverse transform
y_true = y_test.tolist()
# y_true = scaler_y.inverse_transform(y_true)
y_t = []
for x in y_true:
    if x[0] == 1:
        y_t.append(0)
    elif x[1] == 1:
        y_t.append(1)
    elif x[2] == 1:
        y_t.append(2)
    else:
        y_t.append(None)
y_true = y_t

t = []
for x in test_results:
    t.append(np.where(x == np.amax(x))[0][0])

test_results = t

# ===== DONE =====
## Evaluation metrics
print('test results')
print(test_results)
print('y_true')
print(y_true)

acc = accuracy_score(y_true, test_results)
print("Accuracy", acc)
print(confusion_matrix(y_true, test_results))
print(classification_report(y_true, test_results, digits=4))
