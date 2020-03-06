import NLP
from sklearn.model_selection import train_test_split
import model
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import pandas as pd
import numpy as np
#train model...
input_dataset = '/home/mvanessa/pastprojects/finalcode/Augmented_Feat.csv'
# input_dataset = '/Users/michellevanessa/Desktop/automatic-text-scoring-master/Final Code and Data/Augmented_Feat.csv'

df = NLP.cleaning_dataset(input_dataset)
# df = df.iloc[:2500, :]

X = df[['Ref Answer', 'Answer']]
y = pd.DataFrame(df['ans_grade'])

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


scaler_y = MinMaxScaler()
scaler_y.fit(y)
y = scaler_y.transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=101)

df = X_train
df['ans_grade'] = y_train

df_test = X_test
df_test['ans_grade'] = y_test

# Train the model
test, train_model, tokenizer = model.train_dataset_model(df)

#Obtain test results by training on the test dataset dataframe
test_results = model.test_dataset_model(df_test,train_model, tokenizer)

## Processing of the test result to obtain a uniform format and then inverse transform
y_true = y_test.tolist()
y_true = scaler_y.inverse_transform(y_true)
y_t = []
for i in range(len(y_true)):
    for x in y_true[i]:
        y_t.append(x)
y_true = y_t

t = []
for i in range(len(test_results)):
    temp = []
    temp.append(test_results[i])
    t.append(temp)
test_results = t
test_results = pd.DataFrame(test_results)
test_results = scaler_y.inverse_transform(test_results)
t = []
for i in range(len(test_results)):
    for x in test_results[i]:
        t.append(x)
test_results = t


# ===== DONE =====
## Evaluation metrics
print('test results')
print(test_results)
print('y_true')
print(y_true)

pearson, pval = pearsonr(test_results, y_true)
rms = sqrt(mean_squared_error(test_results, y_true))
mae = mean_absolute_error(test_results, y_true)
print("Pearson" , round(pearson, 4))
print("RMS", round(rms, 4))
print("MAE", round(mae, 4))
