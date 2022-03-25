import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

class GroupZeroTokenizer:

    def getTokens(self, complaint):
        fileName = 'resources/onevsresttokenization.pkl'
        maxLength = 1000
        print(complaint);
        file = open(fileName, 'rb')
        tokenizer = pickle.load(file)
        file.close()
        token = tokenizer.texts_to_sequences([complaint])
        print (complaint)
        token = pad_sequences(token, maxlen=maxLength)
        print('Sequence')
        print(token.shape)
        print(token[0])
        return token