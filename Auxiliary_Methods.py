from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import scikitplot as skplt
import plotly.graph_objects as go

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
    
def scores(df, y_test_flat, y_pred_flat):
    print(f'Accuracy: {accuracy_score(y_test_flat, y_pred_flat)}%')
    
    skplt.metrics.plot_confusion_matrix(
        y_test_flat, 
        y_pred_flat,
        figsize=(12,12));
    
    fig = go.Figure(data=go.Heatmap(
                       z= confusion_matrix(y_test_flat, y_pred_flat),
                       x=[f'class-{i}' for i in np.unique(df.loc[:, 'class'].values)],
                       y=[f'class-{i}' for i in np.unique(df.loc[:, 'class'].values)],
                       hoverongaps = False))
    fig.show()
    
    print('Classification report:\n',classification_report(y_test_flat,y_pred_flat))