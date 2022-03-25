from fastapi import FastAPI
from dtos.customerComplaint import CustomerComplaint
from tokenizers.groupZeroTokenizer import GroupZeroTokenizer
from tokenizers.subGroupTokenizer import SubGroupTokenizer
from models.lstm.groupZeroPredictionModel import GroupZeroPredictionModel
from models.lstm.subGroupsPredictionModel import SubGroupsPredictionModel
from tokenizers.allGroupsTokenizer import AllGroupTokenizer
from models.lstm.allGroupsPredictionModel import AlGroupsPredictionModel


app = FastAPI()


@app.post("/complaint/", response_model= CustomerComplaint)
async def predictcomplaint(complaint: CustomerComplaint):
    category = evaluatecomplaint(complaint.description)

    complaint.predictedcategory=category[0]

    return complaint


@app.get("/")
async def root():
    return {"message": "Hello World"}


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
