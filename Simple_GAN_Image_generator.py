import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam

#General image variables
GENERATE_RES = 2
GENERATE_SQUARE = 32 * GENERATE_RES
IMAGE_CHANNELS = 3

#Preview image
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16
SAVE_FREQ = 100

#Size of vector to generate images
SEED_SIZE = 150

#Configuration for example
DATA_PATH = r'PATH/TO/IMAGE/DATA'
EPOCHS = 10000
BATCH_SIZE = 128

print(f"generate {GENERATE_SQUARE}px square images.")

training_binary_path = os.path.join(DATA_PATH,f'bin_ds_{GENERATE_SQUARE}_{GENERATE_SQUARE}.npy')

print(f"Looking for file: {training_binary_path}")

if not os.path.isfile(training_binary_path):
    print("Loading training images")
    training_data = []
    faces_path = os.path.join(DATA_PATH,'YOUR TRAINING DATA FOLDER')
    # DATA_PATH
    for filename in tqdm(os.listdir(faces_path)):
        path = os.path.join(faces_path,filename)
        image = Image.open(path).resize((GENERATE_SQUARE,GENERATE_SQUARE),Image.ANTIALIAS)
        training_data.append(np.asarray(image))
    training_data = np.reshape(training_data,(-1,GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS))
    training_data = training_data / 127.5 - 1.

    print("Saving image data as binary")
    np.save(training_binary_path,training_data)
else:
    print("Loading bin data")
    training_data = np.load(training_binary_path)
    print("Data has been loaded")

def generator_net(seed_size, channels):
    gen_moment = 0.3

    model = Sequential()
    model.add(Dense(4*4*256,activation="relu",input_dim=seed_size))
    model.add(Reshape((4,4,256)))
    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=gen_moment))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=gen_moment))
    model.add(Activation("relu"))

#Output res
    for i in range(GENERATE_RES):
        model.add(UpSampling2D())
        model.add(Conv2D(128,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=gen_moment))
        model.add(Activation("relu"))
    #CNN layer
    model.add(Conv2D(channels,kernel_size=3,padding="same"))
    model.add(Activation("tanh"))

    input = Input(shape=(seed_size,))
    generated_image = model(input)

    return Model(input,generated_image)


def discriminator_net(image_shape):
    discr_moment = 0.3
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=discr_moment))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=discr_moment))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=discr_moment))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=discr_moment))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    input_image = Input(shape=image_shape)
    validity = model(input_image)

    return Model(input_image, validity)

def save_images(cnt, noise):
    image_array = np.full((
      PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE+PREVIEW_MARGIN)),
      PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 3),
      255, dtype=np.uint8)

    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

    image_count = 0
    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
            r = row * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
            c = col * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
            image_array[r:r+GENERATE_SQUARE,c:c+GENERATE_SQUARE] = generated_images[image_count] * 255
            image_count += 1


    output_path = os.path.join(DATA_PATH,'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = os.path.join(output_path,f"train-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)

image_shape = (GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS)
optimizer = Adam(2.0e-4,0.5) # learning rate and momentum adjusted from paper

discriminator = discriminator_net(image_shape)
discriminator.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
generator = generator_net(SEED_SIZE,IMAGE_CHANNELS)

random_input = Input(shape=(SEED_SIZE,))
generated_image = generator(random_input)
discriminator.trainable = False
validity = discriminator(generated_image)
combined = Model(random_input,validity)
combined.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])

import tensorflow as tf

global graph, model
graph = tf.compat.v1.get_default_graph()

y_orig = np.ones((BATCH_SIZE,1))
y_fake = np.zeros((BATCH_SIZE,1))

fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))
cnt = 1
for epoch in range(EPOCHS):
    idx = np.random.randint(0,training_data.shape[0],BATCH_SIZE)
    x_real = training_data[idx]

    #Generating images
    seed = np.random.normal(0,1,(BATCH_SIZE,SEED_SIZE))
    x_fake = generator.predict(seed)

    #Train discriminator
    discriminator_metric_real = discriminator.train_on_batch(x_real,y_orig)
    discriminator_metric_generated = discriminator.train_on_batch(x_fake,y_fake)
    discriminator_metric = 0.5 * np.add(discriminator_metric_real,discriminator_metric_generated)

    #Train generator
    generator_metric = combined.train_on_batch(seed,y_orig)

    #Progress
    if epoch % SAVE_FREQ == 0:
        save_images(cnt, fixed_seed)
        cnt += 1
        print(f"Epoch {epoch}, Discriminator accuarcy: {discriminator_metric[1]}, Generator accuracy: {generator_metric[1]}")

generator.save(os.path.join(DATA_PATH,"saved_GAN_NN.h5"))