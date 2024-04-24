import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import cv2
import numpy as np

def load_images_and_labels(dataset_dir):
    image_data = []
    labels = []
    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder()

    # Iterate through each class folder
    classes = sorted(os.listdir(dataset_dir))
    print("Classes sono: ", classes)
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            # Load images from the class folder
            for image_filename in os.listdir(class_dir):
                if image_filename.endswith('.png'):
                    image_path = os.path.join(class_dir, image_filename)
                    image = cv2.imread(image_path)
                    image_data.append(image)
                    labels.append(class_name)

    # Encode labels
    label_encoder.fit(classes)
    encoded_labels = label_encoder.transform(labels)
    print("encoded labels intermedio",encoded_labels)
    encoded_labels = one_hot_encoder.fit_transform(encoded_labels.reshape(-1, 1)).toarray()


    return np.array(image_data), encoded_labels

# Example usage
dataset_dir = '../assets'
X, y = load_images_and_labels(dataset_dir)

#create the subplot
fig, axes = plt.subplots(3,3, figsize=(16, 16), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))

# plot a grid of images
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i], cmap='jet', interpolation='nearest')
    #nax.text(0.05, 0.05, str(y), transform=ax.transAxes, color='green')

plt.show()

print("prova",y)
