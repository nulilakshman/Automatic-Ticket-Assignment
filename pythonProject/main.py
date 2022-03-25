# This is a sample Python script.
import tensorflow
from tokenizers.groupZeroTokenizer import GroupZeroTokenizer
from tokenizers.subGroupTokenizer import SubGroupTokenizer
from models.lstm.groupZeroPredictionModel import GroupZeroPredictionModel
from models.lstm.subGroupsPredictionModel import SubGroupsPredictionModel
from tokenizers.allGroupsTokenizer import AllGroupTokenizer
from models.lstm.allGroupsPredictionModel import AlGroupsPredictionModel

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(complaint):
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
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {tensorflow.__version__}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('login disabled due to multiple incorrect attempts')
    #print_hi('job job fail job scheduler job job fail job scheduler')
    #print_hi('job hr payroll na u fail job scheduler job hr payroll na u fail job scheduler')
    #print_hi('hostname erp sid volume dev hd server space consume space available mb hostname erp sid volume dev hd server space consume space available mb')
    #print_hi('unable complete forecast unable complete forecast jochegtyhu vacation help fnqelwpk ahrskvln regional manager latin amerirtca fnqelwpk ahrskvln pm ftnijxup sbltduco help scm software jochegtyhu mean complete forecast fnqelwpk ahrskvln regional manager latin amerirtca')
    #print_hi('need access erp kp need access kp enter forecast cost center request receive access end access appear go again need')
    print_hi('vendor draw access not work no vendor access datum anymore vendor access work external website no vendor access drawing model situation get critical cause delay customer order vendor affect error message unable read document information erp see attachment')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
