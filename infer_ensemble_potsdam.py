import numpy as np
from scipy.misc import imsave

def infer_ensemble(index):
    one=np.load('top_potsdam_'+str(index)+'_RGBIR_1.npy')
    # two = np.load('top_potsdam_' + str(index) + '_RGBIR_2.npy')
    three=np.load('top_potsdam_'+str(index)+'_RGBIR_3.npy')
    four=np.load('top_potsdam_'+str(index)+'_RGBIR_4.npy')
    five= np.load('top_potsdam_'+str(index)+'_RGBIR_5.npy')
    six=  one + three + four + five #+ two
    height = np.shape(one)[0]
    width = np.shape(one)[1]
    print(np.shape(six))
    predict_annotation_image = np.argmax(six, axis=2)
    print(np.shape(predict_annotation_image))
    output_image= np.zeros([height,width,3])
    print(np.shape(output_image))
    for i in range(height):
        for j in range(width):
            if predict_annotation_image[i,j]==0:
                output_image[i,j,:]=[255,255,255]
            elif predict_annotation_image[i,j]==1:
                output_image[i,j,:]=[0,0,255]
            elif predict_annotation_image[i,j]==2:
                output_image[i,j,:]=[0,255,255]
            elif predict_annotation_image[i,j]==3:
                output_image[i,j,:]=[0,255,0]
            elif predict_annotation_image[i,j]==4:
                output_image[i,j,:]=[255,255,0]
            elif predict_annotation_image[i,j]==5:
                output_image[i,j,:]=[255,0,0]
    imsave('top_potsdam_'+str(index)+'_RGBIR_class.tif',output_image)
    del one
    # del two
    del three
    del four
    del five
    del six


if __name__ == "__main__":
    list=['2_13','2_14','3_13','3_14','4_13','4_14','4_15','5_13','5_14','5_15','6_13','6_14','6_15','7_13']
    for index in list:
        infer_ensemble(index)