from tensorflow.keras.models import load_model
import numpy as np
import pickle


class AlGroupsPredictionModel:
    def __init__(self):
        super().__init__()

    def loadmodel(self):
        modelpath='resources/all_grps_one_model.h5'
        modelweightpath = 'resources/all_grps_one_model_weights.h5'

        self.model = load_model(modelpath)
        self.model.load_weights(modelweightpath)

    def predict(self, embeddings):
        self.loadmodel()
        print('predict function')
        print(embeddings.shape)
        X=embeddings

        predicted = self.model.predict(X)
        print('predicted')
        print(predicted)
        Value = np.argmax(predicted[0])
        return Value

    def predictlabel(self, embeddings):
        result = self.predict(embeddings)
        print(result)
        filename = 'resources/encode_74_groups.pkl'
        file = open(filename, 'rb')
        le_loaded = pickle.load(file)
        print(le_loaded.classes_)
        file.close()
        label = le_loaded.inverse_transform([result])
        print(label)
        return label
