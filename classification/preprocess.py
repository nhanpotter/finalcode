import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

stop = stopwords.words('english')


def scale(df):
    X = df[['Ref Answer', 'Answer']]
    Y = pd.DataFrame(df['ans_grade'])

    Y_np = Y.to_numpy()
    # y = np.zeros((Y_np.shape[0], 3)) # 3 classes
    y = np.zeros((Y_np.shape[0], 5)) # 5 classes
    for index, row in Y.iterrows():
        # 3 classes
        # val = 1
        # if row['ans_grade'] <= 1:
        #     val = 0
        # elif row['ans_grade'] >= 4:
        #     val = 2

        # 5 classes
        val = row['ans_grade']
        val = int(round(val)) - 1

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

    return X, y


def split(X, y, split):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=101)

    return X_train, X_test, y_train, y_test


def cleaning_dataset(input_file):
    df_train = pd.read_csv(input_file, encoding='unicode escape')  # TODO: Try change to utf
    # df_train = df_train.iloc[:10, :]

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

    return df_train.iloc[:10,:]


def get_questions(file):
    df = pd.read_csv(file, encoding='utf-8')
    df['Question'] = df['Question'].astype(str)
    df['Question'] = df['Question'].apply(lambda question: question.lower())
    df['Question'] = df['Question'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop))
    df['Question'] = df['Question'].apply(lambda question: question.split())
    return df


def uniquecount(sentence):
    specialchar = [',', '.', '(', ')']
    for i in specialchar:
        sentence = sentence.replace(i, '')
    return len(set(sentence.split()))


def question_demoting(df, file):
    questions = get_questions(file)
    for index, row in df.iterrows():
        qn = questions.loc[questions['Q_ID'] == row['Q_ID']]
        ans = row['Answer']
        ref = row['Ref Answer']
        demoted_ans = row['Answer'].split(' ')
        demoted_ref = row['Ref Answer'].split(' ')
        qn = qn['Question'].values.tolist()[0]
        for x in qn:
            if x in demoted_ans:
                demoted_ans.remove(x)
                if len(demoted_ans) == 0:
                    demoted_ans.append('null')
            if x in demoted_ref:
                demoted_ref.remove(x)
        demoted_ans = ' '.join(demoted_ans)
        demoted_ref = ' '.join(demoted_ref)
        if len(row['Answer']) != len(demoted_ans):
            df.at[index, 'Answer'] = demoted_ans
            length = len(demoted_ans)
            df.at[index, 'Length Answer'] = length
            df.at[index, 'Len Ref By Ans'] = row['Length Ref Answer'] / length
            words = len(demoted_ans.split())
            df.at[index, 'Words Answer'] = words
            unique = uniquecount(demoted_ans)
            df.at[index, 'Unique Words Answer'] = unique
        if len(row['Ref Answer']) != len(demoted_ref):
            df.at[index, 'Ref Answer'] = demoted_ref
            length = len(demoted_ref)
            df.at[index, 'Length Ref Answer'] = length
            df.at[index, 'Len Ref By Ans'] = length / row['Length Answer']
    return df
