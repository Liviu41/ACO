import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import scikitplot as skplt
import plotly.graph_objects as go
from tqdm import tqdm
import Auxiliary_Methods as am
import matplotlib.pyplot as plt

X, y = am.read_HSI()

am.HSI_band_show(X, 180)

# The below method extracts the pixels from the HSI and saves into CSV 
df = am.extract_pixels(X, y)

# X = (145*145, 200), y = (145*145, 1)
X_flat = df.iloc[:, :-1].values
y_flat = df.iloc[:, -1].values
X_flat.shape, y_flat.shape

X_train_flat, X_test_flat, y_train_flat, y_test_flat, indices_train_flat, indices_test_flat  = train_test_split(X_flat, y_flat,  range(X_flat.shape[0]), 
                                                                                  test_size = 0.8, random_state = 12)
X_train_flat.shape, y_train_flat.shape


knn = KNeighborsClassifier(n_neighbors=16)
knn.fit(X_train_flat, y_train_flat)
y_pred_flat = knn.predict(X_test_flat)

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

plt.figure(figsize=(8, 6))
plt.imshow(df.iloc[:, -1].values.reshape((145, 145)), cmap='jet')
plt.colorbar()
plt.axis('off')
plt.title('Ground Truth')
plt.savefig('ground_truth.png')
plt.show()

pre = y_pred_flat

clmap = [0]*X_flat.shape[0]

for i in tqdm(range(len(indices_train_flat))):
    clmap[indices_train_flat[i]] = y_flat[indices_train_flat[i]]

for i in tqdm(range(len(indices_test_flat))):
    clmap[indices_test_flat[i]] = pre[i]

plt.figure(figsize=(8, 6))
plt.imshow(np.array(clmap).reshape((145, 145)), cmap='jet')
plt.colorbar()
plt.axis('off')
plt.title('Classification Map (KNeighborsClassifier)')
plt.savefig('Classification_map.png')
plt.show()






