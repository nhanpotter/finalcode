import os
import time

import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional, Subtract
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

import embedding


class SiameneLSTM:
    def __init__(self, embedding_dim, max_sequence_length, number_lstm, number_dense, rate_drop_lstm, \
                 rate_drop_dense, hidden_activation, validation_split_ratio):
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.number_lstm_units = number_lstm
        self.rate_drop_lstm = rate_drop_lstm
        self.number_dense_units = number_dense
        self.activation_function = hidden_activation
        self.rate_drop_dense = rate_drop_dense
        self.validation_split_ratio = validation_split_ratio

    def train_model(self, sentences_pair, feat, scores, embedding_meta_data, model_save_directory='./'):
        tokenizer, embedding_matrix = embedding_meta_data['tokenizer'], embedding_meta_data['embedding_matrix']

        train_data_x1, train_data_x2, train_scores, leaks_train, feat_train, \
        val_data_x1, val_data_x2, val_scores, leaks_val, feat_val = embedding.create_train_dev_set(tokenizer,
                                                                                                   sentences_pair, feat,
                                                                                                   scores,
                                                                                                   self.max_sequence_length,
                                                                                                   self.validation_split_ratio)

        if train_data_x1 is None:
            print("-----Failure: Unable to train model-----")
            return None

        nb_words = len(tokenizer.word_index) + 1

        # Creating word embedding layer
        embedding_layer = Embedding(nb_words, self.embedding_dim, weights=[embedding_matrix],
                                    input_length=self.max_sequence_length, trainable=False)

        # Creating LSTM Encoders
        lstm_layer1 = Bidirectional(
            LSTM(150, kernel_initializer='random_uniform', bias_initializer='zeros', activation='sigmoid',
                 kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
        lstm_layer2 = Bidirectional(
            LSTM(150, kernel_initializer='random_uniform', bias_initializer='zeros', activation='sigmoid',
                 kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.001)))

        # Setting LSTM Encoder layer for Second Sentence
        sequence_2_input = Input(shape=(self.max_sequence_length,), dtype='int32')  # Input 1
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        x2 = lstm_layer2(embedded_sequences_2)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.4)(x2)
        x2 = Dense(50, activation='sigmoid')(x2)

        # Setting LSTM Encoder layer for First Sentence
        sequence_1_input = Input(shape=(self.max_sequence_length,), dtype='int32')  # Input 2
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        x1 = Subtract()([embedded_sequences_1, embedded_sequences_2])  # dist = v1 - v2
        x1 = lstm_layer1(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.4)(x1)
        x1 = Dense(50, activation='sigmoid')(x1)

        # Create feature engineering input
        feat_input = Input(shape=(5,))  # Input 3
        feat_dense = Dense(125, activation='sigmoid')(feat_input)
        feat_dense = Dense(125, activation='sigmoid')(feat_dense)
        feat_dense = Dense(125, activation='sigmoid', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(feat_dense)
        feat_dense = Dense(125, activation='sigmoid', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(feat_dense)

        # Creating leaks input
        leaks_input = Input(shape=(leaks_train.shape[1],))  # Input 4
        leaks_dense_layer = Dense(self.number_dense_units, activation=self.activation_function)
        leaks_dense = leaks_dense_layer(leaks_input)
        leaks_dense = Dense(50, activation='sigmoid')(leaks_dense)

        # Merging two LSTM encodes vectors from sentences to
        # pass it to dense layer applying dropout and batch normalisation
        merged = concatenate([x1, x2, feat_dense, leaks_dense])
        merged = Dense(125, activation='sigmoid')(merged)
        merged = Dense(125, activation='sigmoid', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(merged)
        merged = Dense(125, activation='sigmoid', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(merged)
        merged = Dense(25, activation='sigmoid')(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(0.5)(merged)
        preds = Dense(1, activation='linear')(merged)

        model = Model(inputs=[sequence_2_input, sequence_1_input, feat_input, leaks_input], outputs=preds)
        opt = keras.optimizers.Adagrad(lr=0.01)
        model.compile(loss='mae', optimizer=opt, metrics=['mse', 'mae', 'acc'])

        STAMP = 'lstm_%d_%d_%.2f_%.2f' % (
            self.number_lstm_units, self.number_dense_units, self.rate_drop_lstm, self.rate_drop_dense)

        checkpoint_dir = model_save_directory + 'checkpoints/' + str(int(time.time())) + '/'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        bst_model_path = checkpoint_dir + STAMP + '.h5'

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

        tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

        model.fit([train_data_x1, train_data_x2, feat_train, leaks_train], train_scores,
                  validation_data=([val_data_x1, val_data_x2, feat_val, leaks_val], val_scores),
                  epochs=25, batch_size=128, shuffle=True,
                  callbacks=[model_checkpoint, tensorboard])

        return bst_model_path
