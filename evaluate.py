import numpy as np
from scipy.misc import imread

pred = imread('top_mosaic_09cm_area17.tif')
annotation= imread('../ISPRS_semantic_labeling_Vaihingen/gts_for_participants/top_mosaic_09cm_area17.tif')
height = np.shape(pred)[0]
width = np.shape(annotation)[1]
count = 0
for i in range(height):
    for j in range(width):
        if np.array_equal(pred[i,j,:],annotation[i,j,:]):
           count+=1
print("Accuracy: "+ str(count/(height*width)))
# 7  Accuracy: 0.8903082843132074
# 17 Accuracy: 0.8781703479730091
# 23 Accuracy: 0.8522438833297077
# 37 Accuracy: 0.868925821567948
#
# 7_Accuracy: 0.8863893685030587
# 17_Accuracy: 0.8812551463432892
# 23_Accuracy: 0.8460748914662796
# 37_Accuracy: 0.8716171691754436





# 37_Accuracy: 0.8807770930331842
# 23_Accuracy: 0.8600766392337893
# 17 Accuracy: 0.8761184942200549
# 7 Accuracy: 0.8918958296675751


# 7 Accuracy: 0.8910807101011614
# 17 Accuracy: 0.8760065445446088
# 23 Accuracy: 0.8588578665430487
# 37 Accuracy: 0.8798381223600082
