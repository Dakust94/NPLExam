from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import regularizers

##cwe

def custom_we():

    try:
        model = Sequential()
        model.add(embedding)
        model.add(layers.Flatten())
        model.add(layers.Dense(
            10,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)
         ))
        model.add(layers.Dense(len(categories), activation='softmax'))
        model.compile(optimizer='adam',
                      loss=SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
         model.summary()

        model.fit(X_train, twenty_train.target, epochs=10)
        predicted = model.predict(X_test).argmax(axis=1)
        np.mean(predicted == twenty_test.target)
 except:
        print("custom_we ne fonctionne pas")


"""custom_we create is a customized word embeddings to integrate the number of words we want to store, the dimensions your word
embeddings should have, and calculate the number of words in each text by using keras tensorflow"""