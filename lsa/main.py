from math import sqrt

import LSA
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

input_dataset = '/home/mvanessa/pastprojects/finalcode/Augmented_Feat.csv'
# input_dataset = '/Users/michellevanessa/Desktop/automatic-text-scoring-master/Final Code and Data/Augmented_Feat.csv'

df = pd.read_csv(input_dataset, encoding='utf-8')

ref = list(df['Ref Answer'].to_numpy())
ref = LSA.append(ref)

ans = list(df['Answer'].to_numpy())
ans = LSA.append(ans)

predict = []
for x in range(len(ref)):
    print((x + 1) / (len(ref) + 1) * 100, '%')
    predict.append(LSA.get_lsa_score(ref[x], ans[x]) * 5)

y = df['ans_grade']
result = pd.concat([y, pd.Series(predict)], axis=1, sort=False)
result = result.dropna()

pearson, _ = pearsonr(result['ans_grade'], result[0])
rmse = sqrt(mean_squared_error(result['ans_grade'], result[0]))
mae = mean_absolute_error(result['ans_grade'], result[0])

print()
print('Pearson:', round(pearson, 4))
print('RMSE:', round(rmse, 4))
print('MAE', round(mae, 4))
