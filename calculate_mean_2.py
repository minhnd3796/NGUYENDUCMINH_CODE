import numpy as np
from os import listdir

for npyfile in listdir('../ISPRS_semantic_labeling_Vaihingen/top_15_channels'):
    img = np.load('../ISPRS_semantic_labeling_Vaihingen/top_15_channels' + npyfile)
    
