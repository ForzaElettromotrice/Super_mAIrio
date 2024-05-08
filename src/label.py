import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def print_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_and_preprocess_image(image_path, target_size=(16, 16)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # image = cv2.resize(image, target_size) # Risolvere, le immagini non sono tutte della stessa dimensione
    image = image[:16, :16] # Volendo possiamo tagliare le immagini piu grandi in piu immagini 16x16
    # print_image(image)
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image_array

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
                    image = load_and_preprocess_image(image_path)
                    image_data.append(image)
                    labels.append(class_name)

    # Encode labels
    label_encoder.fit(classes)
    encoded_labels = label_encoder.transform(labels)
    print("encoded labels intermedio",encoded_labels)
    encoded_labels = one_hot_encoder.fit_transform(encoded_labels.reshape(-1, 1)).toarray()


    return np.array(image_data), encoded_labels


if __name__ == '__main__':
    # Example usage
    dataset_dir = '../assets'
    X, y = load_images_and_labels(dataset_dir)

    # #create the subplot
    # fig, axes = plt.subplots(3,3, figsize=(16, 16), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))

    # # plot a grid of images
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(X[i], cmap='jet', interpolation='nearest')
    #     #nax.text(0.05, 0.05, str(y), transform=ax.transAxes, color='green')

    # plt.show()
    # show random image from X with cv2
    # X[12] = np.array(X[12], dtype=np.float32) / 255.0
    cv2.imshow("Image", X[20])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    
    # # Create an ImageDataGenerator with augmentation settings
    # datagen = ImageDataGenerator(
    #     rotation_range=20,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True,
    #     vertical_flip=False,
    #     fill_mode='nearest'
    # )
    
    # augmented_images = datagen.flow(X_train, y_train, batch_size=28)

    print("prova",y)
