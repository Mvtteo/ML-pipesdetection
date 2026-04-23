import numpy as np
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report
from skimage.transform import resize

data_dir = pathlib.Path('./data/Training_database_float16')
pipes_paths = list(data_dir.glob('pipes/*.npz'))
nopipes_paths = list(data_dir.glob('nopipes/*.npz'))

img_height, img_width = 224, 224

def load_and_normalize(path):
    image = np.load(path)['data'].astype(np.float32)
    image = resize(image, (img_height, img_width), preserve_range=True)
    image = np.nan_to_num(image, nan=0.0)
    image = (image - image.mean()) / (image.std() + 1e-7)
    return image.ravel()

features = []
labels = []
for path in pipes_paths:
    features.append(load_and_normalize(path))
    labels.append(1)
for path in nopipes_paths:
    features.append(load_and_normalize(path))
    labels.append(0)

features = np.array(features, dtype=np.float32)
labels = np.array(labels)

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=172
)

pca = PCA(n_components=0.95)
features_train = pca.fit_transform(features_train)
features_test = pca.transform(features_test)
print("components:", pca.n_components_)
print("variance explained:", pca.explained_variance_ratio_.sum())

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(features_train, labels_train)
predictions = knn.predict(features_test)

print("F1:", f1_score(labels_test, predictions))
print(classification_report(labels_test, predictions))