import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import scikitplot as skplt
import plotly.graph_objects as go
from tqdm import tqdm
from Auxiliary_Methods import *


X, y = read_HSI()

# show image
scaledData = np.double(X)
scaledData = (scaledData - np.min(scaledData)) / (np.max(scaledData) 
                                                  - np.min(scaledData)) 
scaledDataForVisualising = scaledData.transpose(2, 0, 1)
f, axarr = plt.subplots(1,4) 
axarr[0].imshow(scaledDataForVisualising[199], cmap='jet')
axarr[1].imshow(scaledDataForVisualising[198], cmap='jet')
axarr[2].imshow(scaledDataForVisualising[197], cmap='jet')
axarr[3].imshow(scaledDataForVisualising[196], cmap='jet')
plt.show()
# show image

# The below method extracts the pixels from the HSI and saves into CSV 
df = extract_pixels(X, y)

# X = (145*145, 200), y = (145*145, 1)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X.shape, y.shape

X_train, X_test, y_train, y_test, indices_train, indices_test  = train_test_split(X, y,  range(X.shape[0]), 
                                                                                  test_size = 0.8, random_state = 12)
X_train.shape, X_test.shape


knn = KNeighborsClassifier(n_neighbors=16)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}%')

skplt.metrics.plot_confusion_matrix(
    y_test, 
    y_pred,
    figsize=(12,12));

fig = go.Figure(data=go.Heatmap(
                   z= confusion_matrix(y_test, y_pred),
                   x=[f'class-{i}' for i in np.unique(df.loc[:, 'class'].values)],
                   y=[f'class-{i}' for i in np.unique(df.loc[:, 'class'].values)],
                   hoverongaps = False))
fig.show()

print('Classification report:\n',classification_report(y_test,y_pred))

plt.figure(figsize=(8, 6))
plt.imshow(df.iloc[:, -1].values.reshape((145, 145)), cmap='jet')
plt.colorbar()
plt.axis('off')
plt.title('Ground Truth')
plt.savefig('ground_truth.png')
plt.show()

pre = y_pred

clmap = [0]*X.shape[0]

for i in tqdm(range(len(indices_train))):
    clmap[indices_train[i]] = y[indices_train[i]]

for i in tqdm(range(len(indices_test))):
    clmap[indices_test[i]] = pre[i]

plt.figure(figsize=(8, 6))
plt.imshow(np.array(clmap).reshape((145, 145)), cmap='jet')
plt.colorbar()
plt.axis('off')
plt.title('Classification Map (PCA + KNeighborsClassifier)')
plt.savefig('Classification_map.png')
plt.show()






