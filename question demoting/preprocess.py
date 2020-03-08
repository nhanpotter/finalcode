import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

stop = stopwords.words('english')


def scale(df):
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

    return X, y, scaler_y


def split(X, y, split):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=101)

    return X_train, X_test, y_train, y_test


def cleaning_dataset(input_file):
    df_train = pd.read_csv(input_file, encoding='unicode escape')

    # Pre-Processing...
    # convert all answers to string format...
    df_train['Ref Answer'] = df_train['Ref Answer'].astype(str)
    df_train['Answer'] = df_train['Answer'].astype(str)

    # convert all answers to lower case...
    df_train['Ref Answer'] = df_train['Ref Answer'].apply(lambda answer1: answer1.lower())
    df_train['Answer'] = df_train['Answer'].apply(lambda answer2: answer2.lower())

    # Remove of Stop Words from answers...
    df_train['Ref Answer'] = df_train['Ref Answer'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df_train['Answer'] = df_train['Answer'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    return df_train
