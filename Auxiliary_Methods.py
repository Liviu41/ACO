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
    
    
def show_gt_classifierMap_ACOMap(df, X, clmap, aco_bool = False, aco_map = None):
    plt.figure(figsize=(8, 6))
    plt.imshow(df.iloc[:, -1].values.reshape((X.shape[0], X.shape[0])), cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.title('Ground Truth')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(np.array(clmap).reshape((X.shape[0], X.shape[0])), cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.title('Classification Map (KNeighborsClassifier)')
    plt.savefig('Classification_map.png')
    plt.show()
    
    if aco_bool == True:
        plt.figure(figsize=(8, 6))
        plt.imshow(np.array(aco_map), cmap='jet')
        plt.colorbar()
        plt.axis('off')
        plt.title('ACO')
        plt.savefig('Classification_map.png')
        plt.show()        
            
    
    
def calcProbab(no_of_classes, i, j, k, tau, eta, alpha = 1, beta = 1):
    p = (tau[i][j][k] ** alpha) * (eta[i][j][k] ** beta)                            
    summ = 0
    
    for s in range(no_of_classes.size):
        summ += (tau[i][j][s] ** alpha) * (eta[i][j][s] ** beta)      
        
    p /= summ    
    return p
        
# am.update_tau_solution_based(tau[i][j][k], p[i][j][k], k, rho, deltaTau)          
def update_tau_solution_based(tau, p, i, j, k, rho, deltaTau):
    retVal = tau            
    
    # evaporation 
    retVal *= (1 - rho)
    
    p_max = np.argmax(p[i][j])
    
    # bonus
    if p_max == k:
        retVal += deltaTau 
        
    return retVal

def myScore(gt, clmap, aco_map, test_size):
    matrix_classifier = np.zeros((clmap.shape[0], clmap.shape[0]))
    matrix_aco = np.zeros((aco_map.shape[0], aco_map.shape[0]))
    
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i][j] != clmap[i][j]:                
                matrix_classifier[i][j] = 1
            if gt[i][j] != aco_map[i][j]:                
                matrix_aco[i][j] = 1
                
    score_classifier = np.sum(matrix_classifier)
    score_aco = np.sum(matrix_aco)
    
    score_classifier /= test_size
    score_aco /= test_size 
    
    score_classifier = 1 - score_classifier
    score_aco = 1 - score_aco
    
    print("Lower is better!")
    print(f'Classifier accuracy = {score_classifier:.3f}%')
    print(f'ACO accuracy = {score_aco:.3f}%')
    
    return score_classifier, score_aco
  
    
def computeF(i, j ,k, aco_map, no_of_classes, f):   
    # if i == 0 and j == 0:
    #     for a in range(i, i + 2):
    #         for b in range(j, j + 2):
    #             if aco_map[a][b] == k:
    #                 f[k] += 1 
                    
    # if i == 0 and j != 0:
    #     for a in range(i, i + 2):
    #         for b in range(j - 1, j + 2):
    #             if aco_map[a][b] == k:
    #                 f[k] += 1  
                    
    # if i != 0 and j == 0:
    #     for a in range(i - 1, i + 2):
    #         for b in range(j, j + 2):
    #             if aco_map[a][b] == k:
    #                 f[k] += 1

    # if i == 144 and j == 144:
    #     for a in range(i - 1, i + 1):
    #         for b in range(j - 1, j + 1):
    #             if aco_map[a][b] == k:
    #                 f[k] += 1  
                    
    # if i == 144 and j != 144:
    #     for a in range(i - 1, i + 1):
    #         for b in range(j - 1, j + 2):
    #             if aco_map[a][b] == k:
    #                 f[k] += 1  

    # if i != 144 and j == 144:
    #     for a in range(i - 1, i + 2):
    #         for b in range(j - 1, j + 1):
    #             if aco_map[a][b] == k:
    #                 f[k] += 1

    # if i == 0 and j == 144:
    #     for a in range(i, i + 2):
    #         for b in range(j - 1, j + 1):
    #             if aco_map[a][b] == k:
    #                 f[k] += 1

    # if i == 144 and j == 0:
    #     for a in range(i - 1, i + 1):
    #         for b in range(j, j + 2):
    #             if aco_map[a][b] == k:
    #                 f[k] += 1                          
                    
        
    for a in range(i - 1, i + 2):
        for b in range(j - 1, j + 2):
            if aco_map[a][b] == k:
                f[k] += 1
    return f[k]
    
















