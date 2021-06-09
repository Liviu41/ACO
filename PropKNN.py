import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import Auxiliary_Methods as am
import random

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


knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train_flat, y_train_flat)
y_pred_flat = knn.predict(X_test_flat)

pre = y_pred_flat
clmap_flat = [0]*X_flat.shape[0]

for i in tqdm(range(len(indices_train_flat))):
    clmap_flat[indices_train_flat[i]] = y_flat[indices_train_flat[i]]

for i in tqdm(range(len(indices_test_flat))):
    clmap_flat[indices_test_flat[i]] = pre[i]

# am.scores(df, y_test_flat, y_pred_flat)
# am.show_gt_classifierMap_ACOMap(df, X, clmap_flat)

# ACO starts here

""" 
Pseudocode

1) Set training data
    
2) Classify using standard classifier
    
3) Determine heuristic function eta 
    
    For example, if pixel 1 belongs to C1 and pixel 2 belongs to Cj, eta = 
    
    Pixel ID   C1   C2   ..    Cj
       1       1   0.5   ..   0.5
       2      0.5  0.5   ..    1
       ..     ..    ..   ..   ..
       
4) Pheromone matrix initialize, tau =
    
     Pixel ID     C1        C2     ..    Cj
       1       theta11   theta12   ..  theta1j
       2       theta21   theta22   ..  theta2j
       ..        ..        ..      ..    ..
       
4') Maybe step 8 could be used here.       
       
5) For each test node compute probabilities of all possible ant solutions
    
    p[i = pixel][j = edge/class][k = ant] = (tau[i][j]^alpha)*(eta[i][j]^beta) / 
                                            sum(tau[i][k=all edges/classes]^alpha * eta[i][k=all edges/classes]^beta)

6) Assign pixel to class according to the probability

7) Update pheromone based on ant solution:
    tau[i][j] = (1 - rho) * tau[i][j] + delta_tau[i][j]
    rho = constant = 0.5
    delta_tau = { 1, if ant chose edge j
                { 0, otherwise 
                    
8) Update tau based on neighbours:
    tau[i][j] += f[i][j]
    where f[i][j] = no. of edges being j, in the vicinity of pixel i
    
9) repeat from step 5 until desired no. of iterations has passed
    
"""

# two dimensional image in this implementation
clmap = np.array(clmap_flat).reshape((X.shape[0], X.shape[0]))

no_of_classes = np.unique(y)

tau = np.ones((X.shape[0], X.shape[0], no_of_classes.size))

eta = np.ones((X.shape[0], X.shape[0], no_of_classes.size))

# heuristic function: if pixel is assigned to class value = 1. otherwise it is 0.5
for i in range(eta.shape[0]):
    for j in range(eta.shape[1]):
        for k in range(eta.shape[2]):
            if clmap[i][j] != k:
                eta[i][j][k] = 0.5
                
aco_map = clmap.copy()

p = np.ones((aco_map.shape[0], aco_map.shape[0], no_of_classes.size))/17

# no loop for epochs yet
epochs = 1
alpha, beta, rho, deltaTau, epochs = 1.8, 0.2, 0.5, 1, 8

# x, y = pixel coordinates, k = edge/class
for ep in range(epochs):
    for i in range(aco_map.shape[0]):                    
        for j in range(aco_map.shape[1]):
            f = np.zeros((no_of_classes.size))
            for k in range(no_of_classes.size):                
                p[i][j][k] = am.calcProbab(no_of_classes, i, j, k, tau, eta, alpha, beta)                                                                                                                  
            
            for k in range(no_of_classes.size):                
                tau[i][j][k] = am.update_tau_solution_based(tau[i][j][k], p, i, j, k, rho, deltaTau)                                                                                           
                f[k] = am.computeF(i, j ,k, aco_map, no_of_classes, f)                        
                tau[i][j][k] += f[k]
            
            #ant_choice = np.random.choice(np.array(range(17)), p = p[i][j])
            #aco_map[i][j] = ant_choice  
            aco_map[i][j] = np.argmax(p[i][j])                      
                
am.show_gt_classifierMap_ACOMap(df, X, clmap, True, aco_map)            

am.myScore(gt, clmap, aco_map, y_test_flat.shape[0])
            

                
                
    








