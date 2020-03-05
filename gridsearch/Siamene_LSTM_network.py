import numpy as np
import pre_processing
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional, Dot, GRU, Conv1D, MaxPooling1D, Flatten, Subtract
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate, dot, subtract
from keras.optimizers import Adam
from keras import backend
from keras.models import load_model,Model
from keras.callbacks import TensorBoard,EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
import time,os

class SiameneLSTM:
    def __init__(self, batch_size, lr, no_dense, no_lstm, drop_lstm, drop_dense, reg_lstm, reg_dense):
        self.EMBEDDING_DIM = 300
        self.MAX_SEQ_LEN = 5
        self.VAL_SPLIT = 0.2
        self.batch_size = batch_size
        self.lr = lr
        self.no_dense = no_dense
        self.no_lstm = no_lstm
        self.drop_lstm = drop_lstm
        self.drop_dense = drop_dense
        self.reg_lstm = reg_lstm
        self.reg_dense = reg_dense
        
        
    
    def train_model(self, sentences_pair, feat, scores, embedding_meta_data, model_save_directory='./'):
        tokenizer, embedding_matrix = embedding_meta_data['tokenizer'], embedding_meta_data['embedding_matrix']

        train_data_x1, train_data_x2, train_scores, leaks_train, feat_train,\
        val_data_x1, val_data_x2, val_scores, leaks_val, feat_val = pre_processing.create_train_dev_set(tokenizer, sentences_pair,feat, scores, self.MAX_SEQ_LEN, self.VAL_SPLIT)
        
        if train_data_x1 is None:
            print("-----Failure: Unable to train model-----")
            return None

        nb_words = len(tokenizer.word_index) + 1

        # Creating word embedding layer
        embedding_layer = Embedding(nb_words, self.EMBEDDING_DIM, weights=[embedding_matrix],
                                    input_length=self.MAX_SEQ_LEN, trainable=False)


        # Creating LSTM Encoders
        lstm_layer1 = Bidirectional(LSTM(self.no_lstm, kernel_initializer='random_uniform', bias_initializer='zeros', activation='sigmoid', kernel_regularizer=l2(self.reg_lstm), recurrent_regularizer=l2(self.reg_lstm), bias_regularizer=l2(self.reg_lstm)))
        lstm_layer2 = Bidirectional(LSTM(self.no_lstm, kernel_initializer='random_uniform', bias_initializer='zeros', activation='sigmoid', kernel_regularizer=l2(self.reg_lstm), recurrent_regularizer=l2(self.reg_lstm), bias_regularizer=l2(self.reg_lstm)))
        
        # Setting LSTM Encoder layer for Second Sentence
        sequence_2_input = Input(shape=(self.MAX_SEQ_LEN,), dtype='int32') # Input 1
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        x2 = lstm_layer2(embedded_sequences_2)
        x2 = Dropout(self.drop_lstm)(x2)
        x2 = Dense(50, activation='sigmoid')(x2)
        
        # Setting LSTM Encoder layer for First Sentence
        sequence_1_input = Input(shape=(self.MAX_SEQ_LEN,), dtype='int32') # Input 2
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        x1 = Subtract()([embedded_sequences_1, embedded_sequences_2]) #dist = v1 - v2
        x1 = lstm_layer1(x1)
        x1 = Dropout(self.drop_lstm)(x1)
        x1 = Dense(50, activation='sigmoid')(x1)
        
        # Create feature engineering input
        feat_input = Input(shape=(5,)) # Input 3
        feat_dense = Dense(self.no_dense, activation='sigmoid')(feat_input)
        feat_dense = Dense(self.no_dense, activation='sigmoid')(feat_dense)
        feat_dense = Dense(self.no_dense, activation='sigmoid', kernel_regularizer=l2(self.reg_dense), bias_regularizer=l2(self.reg_dense))(feat_dense)
        feat_dense = Dense(self.no_dense, activation='sigmoid', kernel_regularizer=l2(self.reg_dense), bias_regularizer=l2(self.reg_dense))(feat_dense)
        
        # Creating leaks input
        leaks_input = Input(shape=(leaks_train.shape[1],))  # Input 4
        leaks_dense_layer = Dense(self.no_dense, activation='sigmoid', kernel_regularizer=l2(self.reg_dense), bias_regularizer=l2(self.reg_dense))
        leaks_dense = leaks_dense_layer(leaks_input)
        leaks_dense = Dense(50, activation='sigmoid')(leaks_dense)
    
        # Merging two LSTM encodes vectors from sentences to
        # pass it to dense layer applying dropout and batch normalisation
        merged = concatenate([x1, x2, feat_dense, leaks_dense])
        merged = Dense(self.no_dense, activation='sigmoid')(merged)
        merged = Dense(self.no_dense, activation='sigmoid', kernel_regularizer=l2(self.reg_dense), bias_regularizer=l2(self.reg_dense))(merged)
        merged = Dense(self.no_dense, activation='sigmoid', kernel_regularizer=l2(self.reg_dense), bias_regularizer=l2(self.reg_dense))(merged)
        merged = Dense(25, activation='sigmoid')(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(self.drop_dense)(merged)
        preds = Dense(1, activation='linear')(merged)

        model = Model(inputs=[sequence_2_input, sequence_1_input, feat_input, leaks_input], outputs=preds)
        opt = keras.optimizers.Adagrad(lr=self.lr)
        model.compile(loss='mae', optimizer=opt , metrics=['mse', 'mae', 'acc'])

        STAMP = 'lstm_%d_%d_%.2f_%.2f' % (self.no_lstm, self.no_dense, self.drop_lstm, self.drop_dense)

        checkpoint_dir = model_save_directory + 'checkpoints/' + str(int(time.time())) + '/'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        bst_model_path = checkpoint_dir + STAMP + '.h5'

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

        tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

        model.fit([train_data_x1, train_data_x2, feat_train, leaks_train], train_scores,
               validation_data=([val_data_x1, val_data_x2, feat_val, leaks_val], val_scores),
              epochs=35, batch_size=self.batch_size, shuffle=True,
          callbacks=[model_checkpoint, tensorboard])
         
        preds = list(model.predict([train_data_x1, train_data_x2, feat_train, leaks_train], verbose=1).ravel()) #Only for cross check purposes,
                                                                                                                #not used for actual testing
        return preds, bst_model_path
