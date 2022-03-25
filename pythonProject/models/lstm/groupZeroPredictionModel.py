from tensorflow.keras.models import load_model
import numpy as np


class GroupZeroPredictionModel:
    def __init__(self):
        super().__init__()

    def predictlabel(self, embeddings):
        result = self.predict(embeddings)
        print(result)
        if result == 0:
            return 'ALL-OTHERS'

        return 'GRP_0'

    def predict(self, embeddings):
        self.loadmodel()
        print('predict function')
        print(embeddings.shape)
       # X=np.reshape(embeddings, (1, 1000))
        X=embeddings

        predicted = self.model.predict(X)
        print('predicted')
        print(predicted)
        Value = np.argmax(predicted[0])
        return Value

    def loadmodel(self):
        modelpath='resources/onevsrest_model.h5'
        modelweightpath = 'resources/onevsrest_model_weights.h5'

        self.model = load_model(modelpath)
        self.model.load_weights(modelweightpath)
