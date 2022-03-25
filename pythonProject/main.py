import yaml
from fastapi import FastAPI
from yaml import SafeLoader

from dtos.customerComplaint import CustomerComplaint
from tokenizers.groupZeroTokenizer import GroupZeroTokenizer
from tokenizers.subGroupTokenizer import SubGroupTokenizer
from models.lstm.groupZeroPredictionModel import GroupZeroPredictionModel
from models.lstm.subGroupsPredictionModel import SubGroupsPredictionModel
from tokenizers.allGroupsTokenizer import AllGroupTokenizer
from models.lstm.allGroupsPredictionModel import AlGroupsPredictionModel
from preprocessing.dataPreprocessor import DataPreprocessor
from translators.englishTranslator import EnglishTranslator

app = FastAPI()


@app.post("/complaint/", response_model= CustomerComplaint)
async def predictcomplaint(complaint: CustomerComplaint):
    translator = EnglishTranslator()
    englishtext = translator.translate(complaint.description)

    preprocessor = DataPreprocessor()
    description = preprocessor.cleaning_data(englishtext)
    category = evaluatecomplaint(description)
    complaint.description = description
    complaint.predictedcategory=category[0]

    return complaint


@app.get("/")
async def root():
    #preprocessor = DataPreprocessor()
    #s = preprocessor.cleaning_data('received from: monitoring_tool@company.comjob Job_2588 failed in job_scheduler at: 10/31/2016 01:30:00')
    translator = EnglishTranslator()
    s = translator.translate('a kÃ¼ndigung for fgxprnub hlanwgqj, 11161827, effective 31.05.2017 has been approved. ')
    print(s)
    return {"message": s}


def evaluatecomplaint(complaint):
    tokenizer = GroupZeroTokenizer()
    token = tokenizer.getTokens(complaint);
    print(token.shape)
    print(token)

    groupZeroPredictionModel= GroupZeroPredictionModel()
    predictedLabel = groupZeroPredictionModel.predictlabel(token)

    print(f'Predicted Value : {predictedLabel}')

    if predictedLabel == 'ALL-OTHERS':
        subTokenizer = AllGroupTokenizer()
        token = subTokenizer.getTokens(complaint)
        subModel = AlGroupsPredictionModel()
        predictedLabel = subModel.predictlabel(token)

    print(f'Predicted Value : {predictedLabel}')

    return predictedLabel
