import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# adjust your paths here. Recommended to keep it that way in order not to run into git conflicts
BASE_PATH = 'ABAW'

PATH_TO_FEATURES = {
    'VA': os.path.join(BASE_PATH, 'data/Aff-Wild2'),
    'EXPR': os.path.join(BASE_PATH, 'data/Aff-Wild2'),
    'AU': os.path.join(BASE_PATH, 'data/Aff-Wild2')
}

ACTIVATION_FUNCTIONS = {
    'VA': torch.nn.Softmax,
    'EXPR': torch.nn.Softmax,
    'AU': torch.nn.Softmax,
}

NUM_TARGETS = {
    'VA': 2,
    'EXPR': 8,
    'AU': 12
}

PATH_TO_LABELS = {
    'VA': os.path.join(BASE_PATH, 'data/Aff-Wild2/5th_ABAW_Annotations/VA_Estimation_Challenge'),
    'EXPR': os.path.join(BASE_PATH, 'data/Aff-Wild2/5th_ABAW_Annotations/EXPR_Classification_Challenge'),
    'AU': os.path.join(BASE_PATH, 'data/Aff-Wild2/5th_ABAW_Annotations/AU_Detection_Challenge'),
}

LABELS = ['0', '1', '2']

OUTPUT_PATH = '/mnt/wd0/home_back/shutao/ABAW/results'
LOG_FOLDER = os.path.join(OUTPUT_PATH, 'log')
DATA_FOLDER = os.path.join(OUTPUT_PATH, 'data')
MODEL_FOLDER = os.path.join(OUTPUT_PATH, 'model')
PREDICTION_FOLDER = os.path.join(OUTPUT_PATH, 'prediction')
