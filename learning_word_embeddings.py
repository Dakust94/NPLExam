from gensim.utils import tokenize
from gensim.test.utils import datapath
from gensim.models import FastText
from tensorflow.keras.layers import Embedding

##fasttext
def fasttext():

    try:
        class FileIter(object):
          def __init(self, filepath: str):
            self.path = datapath(filepath)

          def __iter__(self):
            with utils.open(self.path, 'r', encoding='utf-8') as fin:
              for line in fin:
                yield list(tokenize(line))

        model = FastText(size=4, window=3, min_count=1)
        model.build_vocab(
          sentences=FileIter(
            'crime-and-punishment.txt'
        ))
        model.train(sentences=common_texts, total_examples=len(common_texts), epochs=10)

        model.wv['axe']

        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)

        word_index = {i: w for i, w in enumerate(model.wv.vocab.keys())}

        embedding_layer = Embedding(
             len(word_index) + 1,
             300,
             weights=[list(model.wv.vectors)],
             input_length=500,
             trainable=False
        )
    except:
        print("fasttext ne fonctionne pas")