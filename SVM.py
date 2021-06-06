import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import Auxiliary_Methods as am
import random
from sklearn.svm import SVC

X, y = am.read_HSI()

#am.HSI_band_show(X, 180)

# The below method extracts the pixels from the HSI and saves into CSV 
df = am.extract_pixels(X, y)

gt = df.iloc[:, -1].values.reshape((X.shape[0], X.shape[0]))  

# X = (145*145, 200), y = (145*145, 1)
X_flat = df.iloc[:, :-1].values
y_flat = df.iloc[:, -1].values
X_flat.shape, y_flat.shape

X_train_flat, X_test_flat, y_train_flat, y_test_flat, indices_train_flat, indices_test_flat  = train_test_split(X_flat, y_flat,
                                                            range(X_flat.shape[0]), test_size = 0.7, random_state = 12, stratify=y_flat)
X_train_flat.shape, y_train_flat.shape


svm = SVC(kernel = 'poly', degree = 7)
svm.fit(X_train_flat, y_train_flat)
y_pred_flat = svm.predict(X_test_flat)

pre = y_pred_flat
clmap_flat = [0]*X_flat.shape[0]

for i in tqdm(range(len(indices_train_flat))):
    clmap_flat[indices_train_flat[i]] = y_flat[indices_train_flat[i]]

for i in tqdm(range(len(indices_test_flat))):
    clmap_flat[indices_test_flat[i]] = pre[i]

am.scores(df, y_test_flat, y_pred_flat)
am.show_gt_classifierMap_ACOMap(df, X, clmap_flat)
            

                
                
    








