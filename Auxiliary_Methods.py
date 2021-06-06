from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import scikitplot as skplt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import IncrementalPCA

def read_HSI(verbose = False):
    X = loadmat('..\..\DB\Indian_Pines\Indian_pines_corrected.mat')['indian_pines_corrected']
    y = loadmat('..\..\DB\Indian_Pines\Indian_pines_gt.mat')['indian_pines_gt']
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
    # plt.figure(figsize=(8, 6))
    # plt.imshow(df.iloc[:, -1].values.reshape((X.shape[0], X.shape[0])), cmap='jet')
    # plt.colorbar()
    # plt.axis('off')
    # plt.title('Ground Truth')
    # plt.show()

    # plt.figure(figsize=(8, 6))
    # plt.imshow(np.array(clmap).reshape((X.shape[0], X.shape[0])), cmap='jet')
    # plt.colorbar()
    # plt.axis('off')
    # plt.title('Classification Map (KNeighborsClassifier)')
    # plt.savefig('Classification_map.png')
    # plt.show()
    
    # if aco_bool == True:
    #     plt.figure(figsize=(8, 6))
    #     plt.imshow(np.array(aco_map), cmap='jet')
    #     plt.colorbar()
    #     plt.axis('off')
    #     plt.title('ACO')
    #     plt.savefig('Classification_map.png')
    #     plt.show()    
    
    gt = df.iloc[:, -1].values.reshape((X.shape[0], X.shape[1])) 
    classifier = np.array(clmap).reshape((X.shape[0], X.shape[1]))
    if aco_bool == True:
        aco = np.array(aco_map)
    
    rows = 1
    cols = 3
    axes=[]
    fig=plt.figure(figsize=(12, 6))
    
    axes.append(fig.add_subplot(rows, cols, 1))
    axes[-1].set_title("Ground Truth")  
    plt.imshow(gt, cmap='jet')
    
    axes.append(fig.add_subplot(rows, cols, 2))
    axes[-1].set_title("Classifier Result")  
    plt.imshow(classifier, cmap='jet')
    
    if aco_bool == True:
        axes.append(fig.add_subplot(rows, cols, 3))
        axes[-1].set_title("Proposed Algorithm Result")  
        plt.imshow(aco, cmap='jet')
    
    fig.tight_layout()    
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
    
    score_classifier *= 100
    score_aco *= 100
        
    print(f'Classifier Accuracy = {score_classifier:.2f}%')
    print(f'Proposed Algorithm Accuracy = {score_aco:.2f}%')
    
    return score_classifier, score_aco


def myScoreReloaded(gt, clmap, aco_map, train_size, total):
    matrix_classifier = np.zeros((clmap.shape[0], clmap.shape[0]))
    matrix_aco = np.zeros((aco_map.shape[0], aco_map.shape[0]))
    
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i][j] == clmap[i][j]:                
                matrix_classifier[i][j] = 1
            if gt[i][j] == aco_map[i][j]:                
                matrix_aco[i][j] = 1
                
    print(np.sum(matrix_classifier))
    print(np.sum(matrix_aco))
    print(total)
    print(train_size)
    score_classifier =  np.sum(matrix_classifier) - train_size
    score_aco = np.sum(matrix_aco) - train_size
    
    dif = total - train_size
    score_classifier /= dif
    score_aco /= dif
    
    score_classifier *= 100
    score_aco *= 100
    
    print("Lower is better!")
    print(f'Classifier accuracy = {score_classifier:.2f}%')
    print(f'ACO accuracy = {score_aco:.2f}%')
    
    return score_classifier, score_aco
  
    
def computeF(i, j ,k, aco_map, no_of_classes, f):   
    if i == 0 and j == 0:
        for a in range(i, i + 2):
            for b in range(j, j + 2):
                if aco_map[a][b] == k:
                    f[k] += 2 
                    
    if i == 0 and j != 0 and j != 144:
        for a in range(i, i + 2):
            for b in range(j - 1, j + 2):
                if aco_map[a][b] == k:
                    f[k] += 2  
                    
    if i != 0 and j == 0 and i != 144:
        for a in range(i - 1, i + 2):
            for b in range(j, j + 2):
                if aco_map[a][b] == k:
                    f[k] += 2

    if i == 144 and j == 144:
        for a in range(i - 1, i + 1):
            for b in range(j - 1, j + 1):
                if aco_map[a][b] == k:
                    f[k] += 2  
                    
    if i == 144 and j != 144 and j != 0:
        for a in range(i - 1, i + 1):
            for b in range(j - 1, j + 2):
                if aco_map[a][b] == k:
                    f[k] += 2  

    if i != 144 and j == 144 and i != 0:
        for a in range(i - 1, i + 2):
            for b in range(j - 1, j + 1):
                if aco_map[a][b] == k:
                    f[k] += 2

    if i == 0 and j == 144:
        for a in range(i, i + 2):
            for b in range(j - 1, j + 1):
                if aco_map[a][b] == k:
                    f[k] += 2

    if i == 144 and j == 0:
        for a in range(i - 1, i + 1):
            for b in range(j, j + 2):
                if aco_map[a][b] == k:
                    f[k] += 2   

    if i != 0 and i != 144 and j != 0 and j != 144:
        for a in range(i - 1, i + 2):
            for b in range(j - 1, j + 2):
                if aco_map[a][b] == k:
                    f[k] += 2                     
                    
        
    # for a in range(i - 1, i + 2):
    #     for b in range(j - 1, j + 2):
    #         if aco_map[a][b] == k:
    #             f[k] += 1
    return f[k]


def trainTestSplit(HSI, gt, TeRatio, randomState=41):
    X_train, X_test, y_train, y_test = train_test_split(HSI, gt, test_size=TeRatio, random_state=randomState, stratify=gt)
    return X_train, X_test, y_train, y_test


def pca(HSI, components = 80):
    HSI_flat = np.reshape(HSI, (-1, HSI.shape[2]))
    n_batches = 64
    # incremental PCA due to very high memory usage
    inc_pca = IncrementalPCA(n_components = components)
    for X_batch in np.array_split(HSI_flat, n_batches):
        inc_pca.partial_fit(X_batch)
    X_ipca = inc_pca.transform(HSI_flat)
    HSI_flat = np.reshape(X_ipca, (HSI.shape[0], HSI.shape[1], components))
    return HSI_flat


def results(X_test, y_test, model, classLabels):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    classification = classification_report(np.argmax(y_test, axis = 1), y_pred, target_names = classLabels)
    accuracy = accuracy_score(np.argmax(y_test, axis = 1), y_pred)
    confusion = confusion_matrix(np.argmax(y_test, axis = 1), y_pred)
    kappa = cohen_kappa_score(np.argmax(y_test, axis = 1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size = 32)
    Test_Loss =  score[0] * 100
    Test_accuracy = score[1] * 100
    return classification, confusion, Test_Loss, Test_accuracy, accuracy * 100, kappa * 100, classLabels, y_pred


def padding(HSI, margin):
    NHSI = np.zeros((HSI.shape[0] + 2 * margin, HSI.shape[1] + 2 * margin, HSI.shape[2]))    
    NHSI[margin:HSI.shape[0] + margin, margin:HSI.shape[1] + margin, :] = HSI
    return NHSI


# compute 11x11 3d pacthes
def HSI3dPatches(HSI, gt, windowSize):
    margin = int((windowSize - 1) / 2)
    NHSI = padding(HSI, margin=margin)
    # split patches
    patchesData = np.zeros((HSI.shape[0] * HSI.shape[1], windowSize, windowSize, HSI.shape[2]))
    patchesLabels = np.zeros((HSI.shape[0] * HSI.shape[1]))
    idx = 0
    
    for i in range(margin, NHSI.shape[0] - margin):
        for j in range(margin, NHSI.shape[1] - margin):
            patch = NHSI[i - margin:i + margin + 1, j - margin:j + margin + 1]   
            patchesData[idx, :, :, :] = patch
            patchesLabels[idx] = gt[i - margin, j - margin]
            idx = idx + 1
            
    patchesData = patchesData[patchesLabels > 0, :, :, :]
    patchesLabels = patchesLabels[patchesLabels > 0]
    patchesLabels -= 1
    return patchesData, patchesLabels


# Compute the Patch to Prepare for Ground Truths
def Patch(data,height_index,width_index, winSize):
    height_slice = slice(height_index, height_index + winSize)
    width_slice = slice(width_index, width_index + winSize)
    patch = data[height_slice, width_slice, :]
    return patch












