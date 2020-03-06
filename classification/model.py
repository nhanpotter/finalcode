from keras.models import load_model
import pandas as pd
import lstm
import embedding
from operator import itemgetter

#initialized required parameters for LSTM network...
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 5
VALIDATION_SPLIT = 0.20
RATE_DROP_LSTM = 0.20
RATE_DROP_DENSE = 0.50
NUMBER_LSTM = 500
NUMBER_DENSE_UNITS = 50
ACTIVATION_FUNCTION = 'sigmoid'

def train_dataset_model(df, y_train):
    df['Ref Answer'] = df['Ref Answer'].astype(str)
    df['Answer'] = df['Answer'].astype(str)
    
    answer1 = df['Ref Answer'].tolist()
    answer2 = df['Answer'].tolist()
    scores = y_train.tolist()
    
    ## creating answers pairs
    answers_pair = [(x1, x2) for x1, x2 in zip(answer1, answer2)]
    print("----------created answers pairs-----------")
    
    ## add features for feature engineering
    feat = pd.DataFrame(df[['Length Answer', 'Length Ref Answer', 'Len Ref By Ans', 'Words Answer', 'Unique Words Answer']])
    
    # creating word embedding meta data for word embedding 
    tokenizer, embedding_matrix = embedding.word_embed_meta_data(answer1 + answer2, EMBEDDING_DIM)
    embedding_meta_data = {'tokenizer': tokenizer,'embedding_matrix': embedding_matrix}
    print("----------created word embedding meta data-----------")
    
    #SiameneBiLSTM is a class for  Long short Term Memory networks
    siamese = lstm.SiameneLSTM(EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, NUMBER_LSTM, NUMBER_DENSE_UNITS, RATE_DROP_LSTM, RATE_DROP_DENSE, ACTIVATION_FUNCTION, VALIDATION_SPLIT)
    preds, model_path = siamese.train_model(answers_pair, feat, scores, embedding_meta_data, model_save_directory='./')
    #preds, model_path = siamese.train_model(answers_pair, scores, embedding_meta_data, model_save_directory='./')
    
    #load the train data in model...
    model = load_model(model_path)
    print("----------model trained-----------")
    return preds, model, tokenizer

def test_dataset_model(df_test, model, tokenizer):
    
    df_test['Ref Answer'] = df_test['Ref Answer'].astype(str)
    df_test['Answer'] = df_test['Answer'].astype(str)
    
    answer1_test = df_test['Ref Answer'].tolist()
    answer2_test = df_test['Answer'].tolist()
    
    ## creating answers pairs
    answers_test_pair = [(x1, x2) for x1, x2 in zip(answer1_test, answer2_test)]
    print("----------created test dataset-----------")
    
    ## features input
    feat = pd.DataFrame(df_test[['Length Answer', 'Length Ref Answer', 'Len Ref By Ans', 'Words Answer', 'Unique Words Answer']])
    
    test_data_x1, test_data_x2, feat, leaks_test = embedding.create_test_data(tokenizer, answers_test_pair, feat, MAX_SEQUENCE_LENGTH)
    
    
    #predict the results
    preds = list(model.predict([test_data_x1, test_data_x2, feat, leaks_test], verbose=1))
    print("----------predicted test results-----------")
    
    #mapping results with input test data...
    
    return preds
