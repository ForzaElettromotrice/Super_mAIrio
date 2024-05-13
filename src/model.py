import tensorflow as tf
import numpy as np
from keras import layers, models
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import load_model

from label import main as label_main
import os
import cv2

target_size = (16, 16, 3)

classes_name = ['goomba', 'mario', 'tiles']

def prepare_model1(target_size, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=target_size),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # layers.Conv2D(128, (3, 3), activation='relu'),
        # layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model():
    # Load images and labels
    X_train, X_val, y_train, y_val, test_generator, classes = label_main()
    classes_name = classes
    print("Classes: ", classes)
    # Prepare model
    num_classes = len(classes)
    model = prepare_model1(target_size, num_classes)
    # model.summary()
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train model
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(test_generator)
    
    print('Test accuracy:', test_acc)
    
    # model.save('model.keras')
    
    return model

def test_screen(model):
    image_folder = 'assets/test_screen/'
    resized_size = (256, 240)
    block_size = 16

    # Load and resize the image
    image = cv2.imread(os.path.join(image_folder, 'Screen1.png'))
    resized_image = cv2.resize(image, resized_size)

    # predictions = []
    # # Iterate over every block of 16x16
    # for i in range(0, resized_size[0] - 16, block_size):
    #     for j in range(0, resized_size[1], block_size):
    #         print("Blocco a coordinate: ", i, i+block_size, " : ", j, j+block_size)
    #         block = resized_image[i:i+block_size, j:j+block_size, :]
            
    #         # Predict the block
    #         block = np.expand_dims(block, axis=0)
    #         prediction = model.predict(block)
    #         index = np.argmax(prediction)
    #         predictions.append(classes[index])
    #         print("Prediction: ", classes[index])
    #     print()
    
    # Write predicted class on the resized image
    for i in range(144, resized_size[0] - 16, block_size):
        for j in range(0, resized_size[1], block_size):
            block = resized_image[i:i+block_size, j:j+block_size, :]
            block = np.expand_dims(block, axis=0)
            prediction = model.predict(block)
            index = np.argmax(prediction)
            class_name = classes_name[index]
            cv2.putText(resized_image, class_name, (j, i+block_size), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)

    # Show the image with predicted classes
    resized_image = cv2.resize(resized_image, (512, 480))
    cv2.imshow("Resized Image with Predicted Classes", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def test_images(model):
    tile = cv2.imread('assets/Dataset/tiles/tile1.png')
    tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    
    prediction = model.predict(np.expand_dims(tile, axis=0))
    print(classes_name[np.argmax(prediction)])
    
    mario = cv2.imread('assets/Dataset/mario/mariop1.png')
    mario = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    
    prediction = model.predict(np.expand_dims(mario, axis=0))
    print(classes_name[np.argmax(prediction)])

def main():
    # Load images and labels
    # model = train_model()
    model = load_model('model.keras')
    
    test_screen(model)


    

if __name__ == '__main__':
    main()