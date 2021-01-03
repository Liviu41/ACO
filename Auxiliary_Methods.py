from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_HSI(verbose = False):
    X = loadmat('C:\Projects\ACO\DB\Indian_Pines\Indian_pines_corrected.mat')['indian_pines_corrected']
    y = loadmat('C:\Projects\ACO\DB\Indian_Pines\Indian_pines_gt.mat')['indian_pines_gt']
    if verbose == True:
        print(f"X shape: {X.shape}\ny shape: {y.shape}")
    return X, y


# The below code is used to extract the pixels from HSI and saves into the CSV 
# file and returns the pandas data frame.
def extract_pixels(X, y, verbose = False):
    q = X.reshape(-1, X.shape[2])
    df = pd.DataFrame(data = q)
    df = pd.concat([df, pd.DataFrame(data = y.ravel())], axis=1)  
    df.columns= [f'band{i}' for i in range(1, 1+X.shape[2])]+['class']
    df.to_csv('Dataset.csv') 
    if verbose == True:
        df.head()
    return df


# method to display an image based on a band of an HSI
def HSI_band_show(X, band):
    # show image
    scaledData = np.double(X)
    scaledData = (scaledData - np.min(scaledData)) / (np.max(scaledData) 
                                                      - np.min(scaledData)) 
    scaledDataForVisualising = scaledData.transpose(2, 0, 1)
    f, axarr = plt.subplots(1,1) 
    axarr.imshow(scaledDataForVisualising[band], cmap='jet')
    # uncomment to show multiple bands
    # axarr[1].imshow(scaledDataForVisualising[198], cmap='jet')
    plt.show()
    # show image