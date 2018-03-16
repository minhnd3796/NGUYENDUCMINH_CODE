import numpy as np
from scipy.misc import imsave

def infer_ensemble(index):
    one=np.load(str(index)+'_ckp1.npy')
    building_addresses=np.argmax(one,axis=2)
    height = np.shape(one)[0]
    width = np.shape(one)[1]
    for i in range(height):
        for j in range(width):
            if building_addresses[i,j]==1:
                one[i,j,1]=one[i,j,1] + 500



    three=np.load(str(index)+'_ckp3.npy')
    four=np.load(str(index)+'_ckp4.npy')
    five= np.load(str(index)+'_ckp5.npy')
    six=  three + four + five + one
    print(np.shape(six))
    predict_annotation_image = np.argmax(six, axis=2)
    print(np.shape(predict_annotation_image))
    # height= np.shape(predict_annotation_image)[0]
    # width= np.shape(predict_annotation_image)[1]
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
    imsave('top_mosaic_09cm_area'+str(index)+'_class__.tif',output_image)
    del one
    del three
    del four
    del five
    del six


if __name__ == "__main__":
    list=[2,4,6,8,10,12,14,16,20,22,24,27,29,31,33,35,38]
    for index in list:
        infer_ensemble(index)