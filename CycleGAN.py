import time
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, BatchNormalization, Activation, Add, Conv2DTranspose,ZeroPadding2D, LeakyReLU
from keras.optimizers import Adam
from keras_contrib.layers import InstanceNormalization
from scipy.misc import imread, imresize

def data_load(data_dir):
    dataA = glob(data_dir + '/testA/*.*')
    dataB = glob(data_dir + '/testB/*.*')
    imagesA = []
    imagesB = []
    for i,image in enumerate(dataA):
        imgA = imresize(imread(image, mode='RGB'), (128, 128))
        imgB = imresize(imread(dataB[i], mode='RGB'), (128, 128))

        if np.random.random() > 0.5:
            imgA = np.fliplr(imgA)
            imgB = np.fliplr(imgB)

        imagesA.append(imgA)
        imagesB.append(imgB)
    imagesA = np.array(imagesA) / 127.5 - 1.
    imagesB = np.array(imagesB) / 127.5 - 1.

    return imagesA, imagesB

def load_test_batch(data_dir, batch_size):
    imagesA = glob(data_dir + '/testA/*.*')
    imagesB = glob(data_dir + '/testB/*.*')

    imagesA = np.random.choice(imagesA, batch_size)
    imagesB = np.random.choice(imagesB, batch_size)

    allA = []
    allB = []

    for i in range(len(imagesA)):
        # Load images and resize images
        imgA = imresize(imread(imagesA[i], mode='RGB').astype(np.float32), (128, 128))
        imgB = imresize(imread(imagesB[i], mode='RGB').astype(np.float32), (128, 128))

        allA.append(imgA)
        allB.append(imgB)

    return np.array(allA) / 127.5 - 1.0, np.array(allB) / 127.5 - 1.0


def residual_block(x):

    res = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(x)
    res = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-5)(res)
    res = Activation('relu')(res)

    res = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(res)
    res = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-5)(res)

    return Add()([res, x])


def generator():
    input_layer = Input(shape=(128,128,3))
    
    x = Conv2D(filters=32, kernel_size=7, strides=1, padding="same")(input_layer)
    x = InstanceNormalization(axis=1)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)
    x = InstanceNormalization(axis=1)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(x)
    x = InstanceNormalization(axis=1)(x)
    x = Activation("relu")(x)

    residual_blocks=6
    for _ in range(residual_blocks):
        x= residual_block(x)

    x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization(axis=1)(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization(axis=1)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=3, kernel_size=7, strides=1, padding="same")(x)
    output = Activation('tanh')(x)

    model = Model(inputs=[input_layer], outputs=[output])

    return model

def discriminator():

	input_layer = Input(shape=(128, 128, 3))
	x = ZeroPadding2D(padding=(1, 1))(input_layer)

	x = Conv2D(filters=64, kernel_size=4, strides=2, padding="valid")(x)
	x = LeakyReLU(alpha=0.2)(x)
	x = ZeroPadding2D(padding=(1, 1))(x)

	hidden_layers = 3
	for i in range(1, hidden_layers + 1):
		x = Conv2D(filters=2 ** i * 64, kernel_size=4, strides=2, padding="valid")(x)
		x = InstanceNormalization(axis=1)(x)
		x = LeakyReLU(alpha=0.2)(x)

		x = ZeroPadding2D(padding=(1, 1))(x)

	output = Conv2D(filters=1, kernel_size=4, strides=1, activation="sigmoid")(x)

	model = Model(inputs=[input_layer], outputs=[output])

	return model

def save_images(originalA, generatedB, recosntructedA, originalB, generatedA, reconstructedB, path):

    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(originalA)
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(generatedB)
    ax.axis("off")
    ax.set_title("Generated")

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(recosntructedA)
    ax.axis("off")
    ax.set_title("Reconstructed")

    ax = fig.add_subplot(2, 3, 4)
    ax.imshow(originalB)
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(2, 3, 5)
    ax.imshow(generatedA)
    ax.axis("off")
    ax.set_title("Generated")

    ax = fig.add_subplot(2, 3, 6)
    ax.imshow(reconstructedB)
    ax.axis("off")
    ax.set_title("Reconstructed")

    plt.savefig(path)


def write_log(callback, name, loss, batch_no):

    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = loss
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()


if __name__ == '__main__':
    data_dir = "data/monet2photo/"
    batch_size = 1
    epochs = 500
    mode = 'train'

    if mode == 'train':

        imagesA, imagesB = data_load(data_dir=data_dir)

        common_optimizer = Adam(0.0002, 0.5)

        discriminatorA = discriminator()
        discriminatorB = discriminator()

        discriminatorA.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])
        discriminatorB.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

        generatorAToB = generator()
        generatorBToA = generator()

        inputA = Input(shape=(128, 128, 3))
        inputB = Input(shape=(128, 128, 3))

        generatedB = generatorAToB(inputA)
        generatedA = generatorBToA(inputB)

        # Reconstruct images back to original images
        reconstructedA = generatorBToA(generatedB)
        reconstructedB = generatorAToB(generatedA)

        generatedAId = generatorBToA(inputA)
        generatedBId = generatorAToB(inputB)

        discriminatorA.trainable = False
        discriminatorB.trainable = False

        probsA = discriminatorA(generatedA)
        probsB = discriminatorB(generatedB)

        adversarial_model = Model(inputs=[inputA, inputB],
                                  outputs=[probsA, probsB, reconstructedA, reconstructedB,
                                           generatedAId, generatedBId])
        adversarial_model.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                                  loss_weights=[1, 1, 10.0, 10.0, 1.0, 1.0],
                                  optimizer=common_optimizer)

        tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), write_images=True, write_grads=True,
                                  write_graph=True)
        tensorboard.set_model(generatorAToB)
        tensorboard.set_model(generatorBToA)
        tensorboard.set_model(discriminatorA)
        tensorboard.set_model(discriminatorB)

        real_labels = np.ones((batch_size, 7, 7, 1))
        fake_labels = np.zeros((batch_size, 7, 7, 1))

        for epoch in range(epochs):
            print("Epoch:{}".format(epoch))

            dis_losses = []
            gen_losses = []

            num_batches = int(min(imagesA.shape[0], imagesB.shape[0]) / batch_size)
            print("Number of batches:{}".format(num_batches))

            for index in range(num_batches):
                print("Batch:{}".format(index))

                batchA = imagesA[index * batch_size:(index + 1) * batch_size]
                batchB = imagesB[index * batch_size:(index + 1) * batch_size]

                generatedB = generatorAToB.predict(batchA)
                generatedA = generatorBToA.predict(batchB)

                dALoss1 = discriminatorA.train_on_batch(batchA, real_labels)
                dALoss2 = discriminatorA.train_on_batch(generatedA, fake_labels)

                dBLoss1 = discriminatorB.train_on_batch(batchB, real_labels)
                dbLoss2 = discriminatorB.train_on_batch(generatedB, fake_labels)

                d_loss = 0.5 * np.add(0.5 * np.add(dALoss1, dALoss2), 0.5 * np.add(dBLoss1, dbLoss2))

                print("d_loss:{}".format(d_loss))

                g_loss = adversarial_model.train_on_batch([batchA, batchB],
                                                          [real_labels, real_labels, batchA, batchB, batchA, batchB])

                print("g_loss:{}".format(g_loss))

                dis_losses.append(d_loss)
                gen_losses.append(g_loss)

            write_log(tensorboard, 'discriminator_loss', np.mean(dis_losses), epoch)
            write_log(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)

            if epoch % 10 == 0:
                batchA, batchB = load_test_batch(data_dir=data_dir, batch_size=2)

                generatedB = generatorAToB.predict(batchA)
                generatedA = generatorBToA.predict(batchB)

                reconsA = generatorBToA.predict(generatedB)
                reconsB = generatorAToB.predict(generatedA)

                for i in range(len(generatedA)):
                    save_images(originalA=batchA[i], generatedB=generatedB[i], recosntructedA=reconsA[i],
                                originalB=batchB[i], generatedA=generatedA[i], reconstructedB=reconsB[i],
                                path="results/gen_{}_{}".format(epoch, i))

        generatorAToB.save_weights("generatorAToB.h5")
        generatorBToA.save_weights("generatorBToA.h5")
        discriminatorA.save_weights("discriminatorA.h5")
        discriminatorB.save_weights("discriminatorB.h5")

    elif mode == 'predict':
        generatorAToB = generator()
        generatorBToA = generator()

        generatorAToB.load_weights("generatorAToB.h5")
        generatorBToA.load_weights("generatorBToA.h5")

        batchA, batchB = load_test_batch(data_dir=data_dir, batch_size=2)

        generatedB = generatorAToB.predict(batchA)
        generatedA = generatorBToA.predict(batchB)

        reconsA = generatorBToA.predict(generatedB)
        reconsB = generatorAToB.predict(generatedA)

        for i in range(len(generatedA)):
            save_images(originalA=batchA[i], generatedB=generatedB[i], recosntructedA=reconsA[i],
                        originalB=batchB[i], generatedA=generatedA[i], reconstructedB=reconsB[i],
                        path="results/test_{}".format(i))	