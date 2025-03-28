import pandas as pd
import re
from tensorflow import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Bidirectional, Dense, Embedding, Layer, Reshape
from keras import backend as K
from gensim.models import FastText

# Tokenization and preprocessing 
def tokenization(dataTeks):
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=1439, split=" ")
    tokenizer.fit_on_texts(dataTeks.values)
    word_index = tokenizer.word_index
    X = tokenizer.texts_to_sequences(dataTeks.values)
    X = keras.preprocessing.sequence.pad_sequences(X, maxlen=25)
    return X, word_index

def textPreprocessing(dataTeks):
    dataTeks = dataTeks.apply(lambda x: x.lower())
    dataTeks = dataTeks.apply(lambda x: re.sub('\\\\"', ' ', x))
    dataTeks = dataTeks.apply(lambda x: re.sub('[^a-zA-Z0-9\s]', ' ', x))
    dataTeks = dataTeks.apply(lambda x: re.sub('\s+', ' ', x))
    dataTeks = dataTeks.apply(lambda x: x.strip())
    data, word_index = tokenization(dataTeks)
    return data, word_index

# Prediction (unchanged)
def prediction(pred):
    threshold = 0.5
    hasil_predict = [""] * (len(pred) + 1)
    for i in range(0, len(pred)):
        if pred[i] >= threshold:
            hasil_predict[i] = "Cyberbullying"
        else:
            hasil_predict[i] = "Not Cyberbullying"      
    return hasil_predict

# FastText embedding 
def fasttext(word_index):
    loaded_ft = FastText.load("CBDetection/Model/ft_model_100_andriansyah_defaultconfig.bin")
    embedding_matrix = np.zeros((len(word_index)+1, 100))
    for word, i in word_index.items():
        if word in loaded_ft.wv:
            embedding_vector = loaded_ft[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

# Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        alpha = K.softmax(e, axis=1)
        context = inputs * alpha
        context = K.sum(context, axis=1)
        return context

# Squash function for CapsNet
def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

# Primary Capsule Layer
class PrimaryCaps(Layer):
    def __init__(self, n_capsules, dim_capsule, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.n_capsules = n_capsules
        self.dim_capsule = dim_capsule

    def build(self, input_shape):
        self.dense = Dense(self.n_capsules * self.dim_capsule, activation='linear')
        super(PrimaryCaps, self).build(input_shape)

    def call(self, inputs):
        x = self.dense(inputs)
        x = Reshape((self.n_capsules, self.dim_capsule))(x)
        return squash(x)

# Simplified Capsule Layer with Routing
class CapsuleLayer(Layer):
    def __init__(self, n_capsules, dim_capsule, num_routing=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.n_capsules = n_capsules
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing

    def build(self, input_shape):
        self.input_n_capsules = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(name='capsule_weight',
                                 shape=(self.input_n_capsules, self.n_capsules, 
                                        self.input_dim_capsule, self.dim_capsule),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs):
        inputs_expanded = K.expand_dims(inputs, 2)  # (batch, input_caps, 1, dim)
        inputs_tiled = K.tile(inputs_expanded, [1, 1, self.n_capsules, 1])  # (batch, input_caps, n_caps, dim)
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W), inputs_tiled)  # Prediction vectors

        # Dynamic Routing
        b = K.zeros(shape=(K.shape(inputs)[0], self.input_n_capsules, self.n_capsules))
        for i in range(self.num_routing):
            c = K.softmax(b, axis=2)  # Coupling coefficients
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # Weighted sum
            if i < self.num_routing - 1:
                b += K.batch_dot(outputs, inputs_hat, [2, 3])  # Update coefficients
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_capsules, self.dim_capsule)

# Global graph
global graph
graph = tf.compat.v1.get_default_graph()

# Hyperparameters
epoch = 10
batch_size = 8
unit = 25
dropout = 0.05
regularization = 0.001
activation = 'sigmoid'
optimizer = 'Adadelta'

# Cyberbullying detection function
def cyberbullying_detection(dataTeks):
    dataTeks = pd.DataFrame(dataTeks)
    data, word_index = textPreprocessing(dataTeks[0])
    embedding_matrix = fasttext(word_index)

    with graph.as_default():
        model = Sequential()
        
        # Embedding Layer
        model.add(Embedding(len(word_index)+1, 100, input_length=data.shape[1], 
                            weights=[embedding_matrix], trainable=False))
        
        # Bidirectional GRU
        model.add(Bidirectional(GRU(unit, return_sequences=True, dropout=dropout, 
                                    recurrent_dropout=dropout, 
                                    kernel_regularizer=keras.regularizers.l2(regularization)), 
                                name='bigru_1'))
        
        # Attention Mechanism
        model.add(AttentionLayer(name='attention_layer'))
        
        # Primary Capsules
        model.add(PrimaryCaps(n_capsules=8, dim_capsule=16, name='primary_caps'))
        
        # Capsule Layer with Routing
        model.add(CapsuleLayer(n_capsules=2, dim_capsule=16, num_routing=3, name='capsule_layer'))
        
        # Flatten capsule outputs and predict
        model.add(Layer(lambda x: K.sqrt(K.sum(K.square(x), axis=-1)), name='capsule_length'))  # Length of vectors
        model.add(Dense(1, activation=activation, name='dense_output'))

        # Load weights (partial match)
        try:
            model.load_weights('CBDetection/Model/model_cyberbullying_detection_100embeddingsize.h5', 
                             by_name=True)
            print("Loaded pre-trained weights (partial match).")
        except:
            print("Could not load weights fully. Proceeding with random initialization.")

        # Compile
        model.compile(loss='binary_crossentropy', optimizer=optimizer)

        # Prediction
        pred = model.predict(data)

    hasil_predict = prediction(pred)

    result = dict()
    ctr = 0
    for hasil in hasil_predict:
        result[ctr] = hasil
        ctr += 1

    return result
