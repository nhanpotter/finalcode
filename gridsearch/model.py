from keras.models import load_model
import pandas as pd
import Siamene_LSTM_network
import pre_processing
from operator import itemgetter

def initializer(df):
    df['Ref Answer'] = df['Ref Answer'].astype(str)
    df['Answer'] = df['Answer'].astype(str)

    answer1 = df['Ref Answer'].tolist()
    answer2 = df['Answer'].tolist()
    scores = df['ans_grade'].tolist()

    answers_pair = [(x1, x2) for x1, x2 in zip(answer1, answer2)]
    print("----------created answers pairs-----------")

    feat = pd.DataFrame(df[['Length Answer', 'Length Ref Answer', 'Len Ref By Ans', 'Words Answer', 'Unique Words Answer']])

    tokenizer, embedding_matrix = pre_processing.word_embed_meta_data(answer1 + answer2, 300)
    embedding_meta_data = {'tokenizer': tokenizer,'embedding_matrix': embedding_matrix}
    print("----------created word embedding meta data-----------")

    return df, answers_pair, feat, scores, embedding_meta_data, tokenizer


def train_dataset_model(batch_size, lr, no_dense, no_lstm, drop_lstm, drop_dense, reg_lstm, reg_dense, answers_pair, feat, scores, embedding_meta_data, tokenizer):
    siamese = Siamene_LSTM_network.SiameneLSTM(batch_size, lr, no_dense, no_lstm, drop_lstm, drop_dense, reg_lstm, reg_dense)
    preds, model_path = siamese.train_model(answers_pair, feat, scores, embedding_meta_data, model_save_directory='./')

    model = load_model(model_path)
    print("----------model trained-----------")
    return preds, model, tokenizer

def test_dataset_model(df_test,model, tokenizer):
    
    df_test['Ref Answer'] = df_test['Ref Answer'].astype(str)
    df_test['Answer'] = df_test['Answer'].astype(str)
    
    answer1_test = df_test['Ref Answer'].tolist()
    answer2_test = df_test['Answer'].tolist()
    
    ## creating answers pairs
    answers_test_pair = [(x1, x2) for x1, x2 in zip(answer1_test, answer2_test)]
    print("----------created test dataset-----------")
    
    ## features input
    feat = pd.DataFrame(df_test[['Length Answer', 'Length Ref Answer', 'Len Ref By Ans', 'Words Answer', 'Unique Words Answer']])
    
    test_data_x1, test_data_x2, feat, leaks_test = pre_processing.create_test_data(tokenizer,answers_test_pair, feat, 5)
    
    
    #predict the results
    preds = list(model.predict([test_data_x1, test_data_x2, feat, leaks_test], verbose=1).ravel())
    print("----------predicted test results-----------")
    
    #mapping results with input test data...
    
    return preds
