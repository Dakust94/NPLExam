from gensim.models import KeyedVectors
import numpy as np
from tensorflow.keras.preprocessing.text import text_to_word_sequence

##Word embeddings
def wordembedding():
  
    try:
        model = KeyedVectors.load_word2vec_format(
            'wiki.en.vec',
            binary=False, encoding='utf8'
         )

        def embed_text(text: str):
          vector_list = [
            model.wv[w].reshape(-1, 1) for w in text_to_word_sequence(text)
            if w in model.wv
          ]
          if len(vector_list) > 0:
            return np.mean(
                np.concatenate(vector_list, axis=1),
                axis=1
            ).reshape(1, 300)
          else:
           return np.zeros(shape=(1, 300))

        assert embed_text('training run').shape == (1, 300)
        
        train_transformed = np.concatenate(
            [embed_text(t) for t in twenty_train.data]
        )
        rf = RandomForestClassifier().fit(train_transformed, twenty_train.target)
        
        test_transformed = np.concatenate(
            [embed_text(t) for t in twenty_test.data]
        )
        predicted = rf.predict(test_transformed)
        np.mean(predicted == twenty_test.target)
    
    except:
        print("word embeddings ne fonctionne pas")


"""word embeddings is a strategy to vectorize some datas (a text of several wordsworks usually at least reasonably well for short text)
 and train it by using gensim. 
"""
