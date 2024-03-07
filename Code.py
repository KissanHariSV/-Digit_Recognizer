# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'digit-recognizer:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-competitions-data%2Fkaggle-v2%2F3004%2F861823%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240307%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240307T170013Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D36fc7d7946d52527351b321c5dd99522c36b37ac41a28f7ac3de54ea5e191e8cc83e54536e7ce3154eea40167fd02c1ba722ec65349e59f94c5115f440cada15fcb8ffdf9d6c9a212220f89e699d433fccaa80120812dae68523b53c169350eb6505560c563c2eb4cfa61f21dab0e43203c14dd787e1a81966e2054d783e3a606384fe759e59f5d09d82b900ac84320c94cc8ec0a856bb2d375f3144ebd53f86b5a6c0a12dee4491e49a8d823e1a601fcde9d0c1517c3d9b9a95bb60d0235513b939307d23e3a1a6829c4134e1763e683c7064becc66cb10a028d6414551ece0bca90a79057ea3198583b0a9c4a372a521f897ef6bdc5f0c432ed6401d213a44'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

#all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
%matplotlib inline
#read the train and test data
train_dataset = pd.read_csv("../input/digit-recognizer/train.csv")
test_dataset  = pd.read_csv("../input/digit-recognizer/test.csv")
print(train_dataset.shape)
print(test_dataset.shape)

#explore the data and data-preprocessing
X_train_pixels = train_dataset.iloc[:, 1:]
y_train_lables = train_dataset.iloc[:,0].astype('int32')
print(X_train_pixels.shape)
print(y_train_lables.shape)

X_train_pixels = X_train_pixels.values.reshape(-1,28,28,1)
X_train_pixels = X_train_pixels/255.0
X_test_pixels = test_dataset.values.reshape(-1,28,28,1)
X_test_pixels = X_test_pixels/255.0
print(X_train_pixels.shape)
print(X_test_pixels.shape)

#explore the data and data-preprocessing
#reshappinng the target(lable) dataset, equivalent to the train dataset. 
#here, the target data is categorized as the number of (classes) digits are 0-9
#and that's where keras.utils.to_categorical is used
y_train_lables = tf.keras.utils.to_categorical(y_train_lables,10)
print(y_train_lables.shape)
plt.imshow(X_train_pixels[7], cmap="binary")
print(y_train_lables[7])

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32,(3,3),activation = 'relu', input_shape=(28,28,1)),
  tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'Same'),
  tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'Same'),
  tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'Same'),
  tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'Same'),
  tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.50),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()

Optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0005, 
            beta_1=0.9, 
            beta_2=0.999, 
            epsilon=1e-07,
            name='Adam'
)
model.compile(optimizer=Optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_pixels, y_train_lables, epochs = 25, shuffle = True)
y_pred = model.predict(X_test_pixels)
y_pred = np.argmax(y_pred, axis = 1) 
print(y_pred)
print(y_pred.shape)
print(y_pred[2])
plt.imshow(X_test_pixels[2],cmap="binary")
sample_submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
sample_submission['Label'] = y_pred
sample_submission.to_csv('submission.csv',index=False)
