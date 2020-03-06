import preprocess
import model

# input_dataset = '/home/mvanessa/pastprojects/finalcode/Augmented_Feat.csv'
input_dataset = '/Users/michellevanessa/Desktop/automatic-text-scoring-master/Final Code and Data/Augmented_Feat.csv'

df = preprocess.cleaning_dataset(input_dataset)
# df = df.iloc[:2500, :]
df = df.iloc[:10, :]

X, y, scaler_y = preprocess.scale(df)

df, df_test, y_test = preprocess.split(X, y, 0.1)

test, train_model, tokenizer = model.train(df)

test_results = model.predict(df_test, train_model, tokenizer)

test_results, y_true = model.processresult(test_results, y_test, scaler_y)

pearson, rms, mae = model.evaluate(test_results, y_true)

print("Pearson", round(pearson, 4))
print("RMS", round(rms, 4))
print("MAE", round(mae, 4))
