import numpy as np
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d

import tensorflow.keras
from tensorflow.keras.layers import Input, Conv1D, Conv2D, UpSampling2D, Reshape
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity

import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import imageio
import io





def loaddata1():
    # The loaddata1 function does multiple things:
    # 1. It loads image and histogram data from numpy files.
    # 2. It applies several transformations to the histograms: moving average smoothing, coarse binning.
    # 3. It applies a quantization process to the image data.
    # 4. It normalizes the data.
    # 5. It splits the data into training and testing sets.

    # Initialize empty lists to store image and histogram data
    images = []
    prehistograms = []

    # Load data from specific paths and append to the respective lists
    for i in range(1, 4):
        for j in range(1, 951):
            # Load the image data and append to images list
            datafolder = np.load(
                r"/home/paul/PycharmProjects/LIDAR/images/image_{}_{}.npy".format(i, j))
            images.append(datafolder)

            # Load the histogram data, ensure it's shape is (1249,) by padding if needed, and append to prehistograms list
            data2folder = np.load(
                r"/home/paul/PycharmProjects/LIDAR/histograms/histogram_{}_{}.npy".format(i, j))
            data2folder = data2folder[1, :].astype(np.int64)
            if data2folder.shape != (1249,):
                pad_width = (0, 1249 - data2folder.shape[0])
                data2folder = np.pad(data2folder, pad_width, mode='constant')
            prehistograms.append(data2folder)

    # Convert the lists to numpy arrays for efficient calculations
    images = np.array(images)
    histograms = np.vstack(prehistograms)

    # Apply a moving average filter to the histograms for smoothing
    # This process helps to reduce high frequency noise
    window_size = 5
    filterz = np.ones(window_size) / window_size
    smoothed_histograms = np.zeros_like(histograms)
    for i in range(histograms.shape[0]):
        smoothed_histograms[i] = np.convolve(histograms[i], filterz, mode='same')

    # Apply coarse binning to the histograms
    # This process groups adjacent bins together to reduce the granularity of the histograms
    bin_reduction_factor = 3
    coarser_histograms = np.zeros_like(histograms)
    for i in range(histograms.shape[0]):
        padding = bin_reduction_factor - (histograms.shape[1] % bin_reduction_factor)
        padded_histogram = np.pad(histograms[i], (0, padding), mode='constant', constant_values=0)
        reduced_bins = np.mean(padded_histogram.reshape(-1, bin_reduction_factor), axis=1)
        x_original = np.linspace(0, histograms.shape[1], histograms.shape[1], endpoint=False)
        x_reduced = np.linspace(0, histograms.shape[1], len(reduced_bins), endpoint=False)
        interpolator = interp1d(x_reduced, reduced_bins, kind='linear', fill_value='extrapolate')
        coarser_histograms[i] = interpolator(x_original)

    # # Apply quantization to the image data
    # # This process reduces the range of pixel intensity values to a smaller set of levels
    # num_levels = 2
    # threshold = 4000
    #
    # # Apply quantization only to pixel values above the threshold
    # scaled_images = np.where(images > threshold, ((images - threshold) * (num_levels - 1) / value_range).astype(int) * (
    #             value_range / (num_levels - 1)) + threshold, images)
    # value_range = np.linspace(0, threshold, num_levels, endpoint=True)
    # scaled_images = np.digitize(images, value_range) - 1


    # Define the number of quantization levels and the threshold value (basically everything higher than 3300 - a depth i chosen based on looking at it, make it either 3300 or 4000
    num_levels = 2
    threshold = 3300

    # Calculate the range of values above the threshold
    value_range = images.max() - threshold

    # Apply quantization only to pixel values above the threshold
    scaled_images = np.where(images > threshold, ((images - threshold) * (num_levels - 1) / value_range).astype(int) * (
            value_range / (num_levels - 1)) + threshold, images)

    # Normalize the data
    # Normalizing the data ensures all features have the same scale
    # This is especially important for machine learning algorithms that use a distance metric
    # or when features have different units
    images[images > 4000] = 4000
    images = images - np.min(images)
    images = images / np.max(images)
    histograms = histograms / np.max(histograms)

    scaled_images[scaled_images > 4000] = 4000
    # scaled_images[scaled_images < 00] = 1000
    scaled_images = scaled_images - np.min(scaled_images)
    scaled_images = scaled_images / np.max(scaled_images)
    smoothed_histograms = smoothed_histograms / np.max(smoothed_histograms)
    coarser_histograms = coarser_histograms / np.max(coarser_histograms)

    # Split the data into training and testing sets
    # This ensures that we have unseen data to evaluate our model's performance
    # We use a standard split of 80% training data and 20% testing data
    # We set a random seed for reproducibility
    scaled_images_train, scaled_images_test, histograms_train, histograms_test = train_test_split(scaled_images,
                                                                                                  histograms,
                                                                                                  test_size=0.2,
                                                                                                  random_state=42)

    # Reshape the data to add an extra dimension
    scaled_images_train = scaled_images_train.reshape(-1, scaled_images_train.shape[1], scaled_images_train.shape[2], 1)
    scaled_images_test = scaled_images_test.reshape(-1, scaled_images_test.shape[1], scaled_images_test.shape[2], 1)
    histograms_train = histograms_train.reshape(-1, histograms_train.shape[1], 1)
    histograms_test = histograms_test.reshape(-1, histograms_test.shape[1], 1)

    # For smoother and coarser_histograms
    smoothed_histograms_train, smoothed_histograms_test, coarser_histograms_train, coarser_histograms_test = train_test_split(
        smoothed_histograms, coarser_histograms, test_size=0.2, random_state=42)

    smoothed_histograms_train = smoothed_histograms_train.reshape(-1, smoothed_histograms_train.shape[1], 1)
    smoothed_histograms_test = smoothed_histograms_test.reshape(-1, smoothed_histograms_test.shape[1], 1)

    coarser_histograms_train = coarser_histograms_train.reshape(-1, coarser_histograms_train.shape[1], 1)
    coarser_histograms_test = coarser_histograms_test.reshape(-1, coarser_histograms_test.shape[1], 1)

    # Save and display the original and scaled images for 5 examples
    # num_examples = 5
    # example_indices = np.linspace(0, images.shape[0] - 1, num_examples, dtype=int)
    #
    # for i in example_indices:
    #     # Original image
    #     plt.figure(figsize=(10, 6))
    #     plt.imshow(images[i], cmap='gray')  # Adjust the colormap (cmap) as needed
    #     plt.axis('off')
    #     plt.savefig(f'original_image_{i + 1}.png', bbox_inches='tight', pad_inches=0)
    #     plt.close()
    #
    #     # Quantized image
    #     plt.figure(figsize=(10, 6))
    #     plt.imshow(scaled_images[i], cmap='gray', vmin=0, vmax=4000)  # Adjust the colormap (cmap) as needed
    #     plt.axis('off')
    #     plt.savefig(f'quantized_image_{i + 1}.png', bbox_inches='tight', pad_inches=0)
    #     plt.close()
    ######################################################################################################

    # Return both the original and processed data for further usage
    return images, histograms, scaled_images, smoothed_histograms, coarser_histograms, scaled_images_train, scaled_images_test, coarser_histograms_train, coarser_histograms_test, smoothed_histograms_train, smoothed_histograms_test, histograms_train, histograms_test

def load_and_plot_data(images, scaled_images, histograms, smoothed_histograms, coarser_histograms):
    num_examples = 5
    # example_indices = np.random.choice(range(len(images)), num_examples, replace=False)
    example_indices =  [1,100,200,250,300]
    fig, axs = plt.subplots(num_examples, 5, figsize=(15, num_examples*3))
    np.min(scaled_images[scaled_images > 0])
    for idx, i in enumerate(example_indices):
        # Original image
        im = axs[idx, 0].imshow(images[i].reshape(60,80), cmap='viridis_r', vmin=0, vmax=1)  # Adjust the colormap (cmap) as needed
        axs[idx, 0].axis('off')
        axs[idx, 0].set_title(f'Original Image {i + 1}')

        # Quantized image
        im = axs[idx, 1].imshow(scaled_images[i].reshape(60,80), cmap='viridis_r', vmin=0.1, vmax=1)  # Adjust the colormap (cmap) as needed
        axs[idx, 1].axis('off')
        axs[idx, 1].set_title(f'Quantized Image {i + 1}')

        # Original histogram
        axs[idx, 2].plot(histograms[i])
        axs[idx, 2].set_title(f'Original Histogram {i + 1}')
        axs[idx, 2].set_xlabel('Bins')
        axs[idx, 2].set_ylabel('Frequency')

        # Smoothed histogram
        axs[idx, 3].plot(smoothed_histograms[i])
        axs[idx, 3].set_title(f'Smoothed Histogram {i + 1}')
        axs[idx, 3].set_xlabel('Bins')
        axs[idx, 3].set_ylabel('Frequency')

        # Coarse histogram
        axs[idx, 4].plot(coarser_histograms[i])
        axs[idx, 4].set_title(f'Coarse Histogram {i + 1}')
        axs[idx, 4].set_xlabel('Bins')
        axs[idx, 4].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
#
#

def LIDAR(epochs, feats, batch, kern_int_e, kern_int_d, kern_reg, kse, ksd, train_h, train_i, test_h, test_i):

    # Encoding
    inp = Input(shape=(1249, 1))

    conv1 = Conv1D(feats, 3, activation=tensorflow.keras.activations.relu, padding='same', kernel_regularizer=kern_reg,
                   kernel_initializer=kern_int_e)(inp)
    conv2 = Conv1D(feats * 2, kse, activation=tensorflow.keras.activations.relu, padding='same',
                   kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, strides=2)(conv1)
    conv3 = Conv1D(feats * 2, kse, activation=tensorflow.keras.activations.relu, padding='same',
                   kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, strides=2)(conv2)
    conv4 = Conv1D(feats * 4, kse, activation=tensorflow.keras.activations.relu, padding='same',
                   kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, strides=2)(conv3)
    conv5 = Conv1D(feats * 4, kse, activation=tensorflow.keras.activations.relu, padding='same',
                   kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, strides=2)(conv4)
    conv6 = Conv1D(feats * 4, kse, activation=tensorflow.keras.activations.relu, padding='same',
                   kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, strides=2)(conv5)
    conv7 = Conv1D(feats * 4, 3, activation=tensorflow.keras.activations.relu, padding='valid',
                   kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, strides=2)(conv6)

    # Change the strides and padding
    conv8 = Conv1D(feats * 4, 3, activation=tensorflow.keras.activations.relu, padding='valid',
                   kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, strides=1)(conv7)
    conv9 = Conv1D(feats * 4, 3, activation=tensorflow.keras.activations.relu, padding='valid',
                   kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, strides=1)(conv8)

    # Add a final Conv1D layer with kernel size 3 and padding 'valid'
    conv10 = Conv1D(feats * 4, 4, activation=tensorflow.keras.activations.relu, padding='valid',
                    kernel_regularizer=kern_reg, kernel_initializer=kern_int_e)(conv9)

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################

    # Add a final Conv1D layer with kernel size 3 and padding 'valid'
    # conv10 = Conv1D(feats * 4, 3, activation=tensorflow.keras.activations.relu, padding='valid',
    #                 kernel_regularizer=kern_reg, kernel_initializer=kern_int_e)(conv9)

    shape = tensorflow.keras.layers.Reshape((3, 4, feats * 4))(conv10)

    # tconv1 = Conv2D(feats * 4, ksd, activation=tensorflow.keras.activations.relu, padding='same',
    #                 kernel_regularizer=kern_reg, kernel_initializer=kern_int_d)(shape)
    ###################################################################################################################
    ###################################################################################################################
    #         Decoding
    ###################################################################################################################

    # Assuming the input shape is (3, 4, feats * 4)
    tconv1 = Conv2D(feats * 4, ksd, activation=tensorflow.keras.activations.relu, padding='same',
                    kernel_regularizer=kern_reg, kernel_initializer=kern_int_d)(shape)
    tconv1 = UpSampling2D(size=(2, 2))(tconv1)

    tconv2 = Conv2D(feats * 4, ksd, activation=tensorflow.keras.activations.relu, padding='same',
                    kernel_regularizer=kern_reg, kernel_initializer=kern_int_d)(tconv1)
    tconv2 = UpSampling2D(size=(2, 2))(tconv2)

    tconv3 = Conv2D(feats * 4, ksd, activation=tensorflow.keras.activations.relu, padding='same',
                    kernel_regularizer=kern_reg, kernel_initializer=kern_int_d)(tconv2)
    tconv3 = UpSampling2D(size=(5, 5))(tconv3)

    tconv4 = Conv2D(feats * 2, ksd, activation=tensorflow.keras.activations.relu, padding='same',
                    kernel_regularizer=kern_reg, kernel_initializer=kern_int_d)(tconv3)

    tconv5 = Conv2D(feats * 2, (3, 3), activation=tensorflow.keras.activations.relu, padding='same',
                    kernel_regularizer=kern_reg, kernel_initializer=kern_int_d)(tconv4)
    tconv6 = Conv2D(feats * 2, (3, 3), activation=tensorflow.keras.activations.relu, padding='same',
                    kernel_regularizer=kern_reg, kernel_initializer=kern_int_d)(tconv5)
    tconv7 = Conv2D(feats, (1, 1), activation=tensorflow.keras.activations.relu, padding='same',
                    kernel_regularizer=kern_reg, kernel_initializer=kern_int_d)(tconv6)

    out = Conv2D(1, 1, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d)(tconv7)

    #################################################
    #
    model = tensorflow.keras.Model(inputs=inp, outputs=out)
    #
    # adam = tensorflow.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999)
    # # model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
    # model.compile(loss=tensorflow.keras.losses.Huber(), optimizer=adam, metrics=['accuracy', 'mse', 'mae'])
    #
    #
    # # %% Summary


    model.summary()
    #
    # # %% Train, evaluate, save
    # #
    # model.fit(train_h, train_i, epochs=epochs, validation_split=0.1, shuffle=True, batch_size=batch,
    #           verbose=1)
    # huber_loss, test_acc, msetest, maetest = model.evaluate(test_h,  test_i, verbose=1)
##################################################################################################################################################################

    # Define the exponential decay learning rate scheduler
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True,
    )

    # def soft_iou_loss(train_i, predicted_images):
    #     intersection = tensorflow.reduce_sum(train_i * predicted_images, axis=[1, 2, 3])
    #     union = tensorflow.reduce_sum(train_i + predicted_images, axis=[1, 2, 3]) - intersection
    #     iou = intersection / (union + tensorflow.keras.backend.epsilon())
    #     return 1 - iou

    def combined_loss(alpha, delta):
        def loss(train_i, predicted_images):
            # Huber Loss
            huber_loss = tensorflow.keras.losses.Huber(delta=delta)(train_i, predicted_images)

            # SSIM Loss
            ssim_loss = 1 - tensorflow.reduce_mean(tensorflow.image.ssim(train_i, predicted_images, max_val=1.0))

            # Combine the losses
            combined = alpha * huber_loss + (1 - alpha) * ssim_loss
            return combined

        return loss

    # Set the parameters for the combined loss
    alpha = 0.5  # Weight for Huber loss (0 <= alpha <= 1)
    delta = 0.5  # Huber loss delta value

    # Compile the model with the learning rate scheduler
    adam = tensorflow.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999)
    # model.compile(loss=soft_iou_loss, optimizer=adam, metrics=['accuracy', 'mse', 'mae'])

    model.compile(loss=tensorflow.keras.losses.Huber(delta=1.0), optimizer=adam, metrics=['accuracy', 'mse', 'mae'])
    # model.compile(loss=combined_loss(alpha,delta), optimizer=adam, metrics=['accuracy', 'mse', 'mae'])



    # Create a LearningRateScheduler callback
    lr_callback = LearningRateScheduler(lr_schedule)

    # Train the model with the learning rate scheduler callback
    model.fit(train_h, train_i, epochs=epochs, validation_split=0.1, shuffle=True, batch_size=batch,
              verbose=1)



    import glob

    folder = '/for lisa'
    scnn = sorted(glob.glob(f'.{folder}/PHOTON*.h5'))
    model = tensorflow.keras.models.load_model(scnn[0])




# Evaluate and predict as before
    # Evaluate the model using Keras metrics
    huber_loss, test_acc, msetest, maetest = model.evaluate(test_h, test_i, verbose=1)

    # Make predictions using the model
    predicted_images = model.predict(test_h)

    # Calculate additional evaluation metrics
    mse_custom = mean_squared_error(test_i.flatten(), predicted_images.flatten())
    mae_custom = mean_absolute_error(test_i.flatten(), predicted_images.flatten())
    ssim = structural_similarity(test_i, predicted_images, multichannel=True)

    # Print Keras evaluation metrics
    print("Keras Huber Loss:", huber_loss)
    print("Keras Test Accuracy:", test_acc)
    print("Keras MSE:", msetest)
    print("Keras MAE:", maetest)

    # Print additional evaluation metrics
    print("Custom Mean Squared Error (MSE):", mse_custom)
    print("Custom Mean Absolute Error (MAE):", mae_custom)
    print("Structural Similarity Index (SSIM):", ssim)

#
    predictions = model.predict(test_h)
    fig1 = plt.figure(figsize=(12, 10))
    plt.axis("off")
    fig2 = plt.figure(figsize=(12, 10))
    plt.axis("off")
    fig3 = plt.figure(figsize=(12, 10))


    for co in range(0, 199, 10):
        # fig=plt.figure(figsize=(80,60))
        fig1.suptitle("SCNN Reconstruction from histogram with background")
        ax = fig1.add_subplot(4, 5, ((co / 10) + 1))
        ax.imshow(predictions[co].reshape(60,80), cmap='viridis_r', vmin=0.02, vmax=1)

        # fig = plt.figure(figsize=(80, 60))
        fig2.suptitle("Ground truth")
        cx = fig2.add_subplot(4, 5, ((co / 10) + 1))
        cx.imshow(test_i[co].reshape(60,80), cmap='viridis_r', vmin=0, vmax=1)
        # fig = plt.figure(figsize=(80, 60))

        fig3.suptitle("Histogram")
        bx = fig3.add_subplot(4, 5, ((co / 10) + 1))
        bx.plot(np.squeeze(test_h[co]))
        #
        # plt.subplot(gs[1])
        # plt.title('Histogram')
        # plt.plot(np.squeeze(test_h[co])) # need to see why this isnt printing actual values!!!
        # plt.subplot(gs[2])
        # plt.title('Ground truth')
        # plt.imshow(test_i[co].reshape(60,80),cmap='viridis_r',vmin=0.02,vmax=4000)
        # #plt.colorbar()

    # plt.show()
    fig1.savefig(
        "./{}_{}_e{}_f{}_b{}_SCNN_Photon_Multi_i.png".format(kse, ksd, epochs,
                                                                                feats, batch))
    fig2.savefig(
        "./{}_{}_e{}_f{}_b{}_SCNN_Photon_Multi_i_GT.png".format(kse, ksd,
                                                                                   epochs, feats, batch))
    fig3.savefig(
        "./{}_{}_e{}_f{}_b{}_SCNN_Photon_Multi_Histo.png".format(kse, ksd, epochs,
                                                                               feats, batch))

    # frames = []
    # # Loop over your data
    # for co in range(0, 999):
    #     # Create a figure
    #     fig = plt.figure(figsize=(24, 5))
    #     gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    #
    #     # Plot the SCNN Reconstruction
    #     ax0 = plt.subplot(gs[0])
    #     ax0.set_title("SCNN Reconstruction from histogram with background")
    #     im0 = ax0.imshow(predictions[co].reshape(60,80), cmap='viridis_r', vmin=0.02, vmax=1)
    #     plt.colorbar(im0, ax=ax0)
    #
    #     # Plot the Ground Truth
    #     ax1 = plt.subplot(gs[1])
    #     ax1.set_title('Ground truth')
    #     im1 = ax1.imshow(test_i[co].reshape(60,80), cmap='viridis_r', vmin=0, vmax=1)
    #     plt.colorbar(im1, ax=ax1)
    #
    #     # Save the figure to a BytesIO object
    #     buf = io.BytesIO()
    #     plt.savefig(buf, format='png')
    #     buf.seek(0)
    #
    #     # Load the image from the BytesIO object into PIL
    #     img = Image.open(buf)
    #
    #     # Convert the image to numpy array and append to frames
    #     frames.append(np.array(img))
    #
    #     # Close the figure
    #     plt.close(fig)

    # Save as a GIF
    # imageio.mimsave('output.gif', frames, 'GIF', duration=0.05)
    #
    NEW_test = model.predict(test_h)

    IOUscnn = []
    rSNR = []
    BrSNR = []
    AbsRel = []
    SqRel = []
    RMSE = []
    RMSElog = []
    acc = []
    acc2 = []
    acc3 = []

    for unseen in [0, 5, 146, 279, 340]:

        fig = plt.figure(figsize=(24, 5))
        fig.suptitle(5)
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
        plt.subplot(gs[0])
        plt.title('SCNN Reconstruction from histogram with background')
        plt.imshow(NEW_test[unseen].reshape(60,80), cmap='viridis_r', vmin=0, vmax=1)
        plt.colorbar()
        plt.subplot(gs[1])
        plt.title("Histogram")
        plt.plot(test_h[unseen])
        plt.subplot(gs[2])
        plt.title('Ground truth')
        plt.imshow(test_i[unseen].reshape(60,80), cmap='viridis_r', vmin=0, vmax=1)
        plt.colorbar()

        fig.savefig("unseen_test_hist_{}_{}_e{}_f{}_b{}_{}.png".format(kse, ksd, epochs, feats, batch, unseen))



    # for unseen in [119, 275, 336, 379, 399]:

    for unseen in [0, 5, 146, 279, 340]:

        ssim1 = tensorflow.image.ssim(
            tensorflow.image.convert_image_dtype(NEW_test[unseen].reshape(60, 80, 1), tensorflow.float32),
            tensorflow.image.convert_image_dtype(test_i[unseen].reshape(60, 80, 1), tensorflow.float32),
            max_val=1, filter_size=5, filter_sigma=1.5, k1=0.01, k2=0.03)

        test = (test_i[unseen].reshape(60, 80)) - (NEW_test[unseen].reshape(60, 80))
        G0 = (test > 0.01).sum()
        L0 = (test < -0.01).sum()
        E0 = (4800 - G0 - L0).sum()

        GT = test_i[unseen].reshape(60, 80)
        Recon = NEW_test[unseen].reshape(60, 80)

        min = np.min(GT[GT>0])#1 - .99  # np.min(GT[GT > (np.min(GT[GT > 0]))])#

        # GT[GT >= min] = 1
        # GT[GT < 1] = 0
        #
        # Recon[Recon >= min] = 1
        # Recon[Recon < 1] = 0

        IOUscnn += [np.sum(np.logical_and(GT, Recon)) / np.sum(np.logical_or(GT, Recon))]

        GTimg = test_i[unseen].reshape(60, 80)
        GTimg[GTimg==0]=0.001
        # XGTimg = GTimg#test_i[unseen].reshape(60, 80)

        BGTimg = GT * GTimg
        BGTimg[BGTimg > 0] = 1

        Reconimg = NEW_test[unseen].reshape(60, 80)
        Reconimg[Reconimg < 0]=0
        Reconimg[Reconimg == 0]=0.001
        # XReconimg = Reconimg

        BReconimg = Recon * Reconimg
        BReconimg[BReconimg > 0] = 1

        rSNR += [10 * np.log10(np.sum(GTimg ** 2) / np.sum((Reconimg - GTimg) ** 2))]
        BrSNR += [10 * np.log10(np.sum(BGTimg ** 2) / np.sum((BReconimg - BGTimg) ** 2))]

        AbsRel += [np.mean(abs(Reconimg - GTimg) / GTimg)]
        SqRel += [np.mean(abs(Reconimg - GTimg) ** 2 / GTimg)]
        RMSE += [np.sqrt(np.mean(abs(Reconimg - GTimg) ** 2))]
        RMSElog += [np.sqrt(np.mean(np.square(
            np.log(np.abs(Reconimg)) / 2 - np.log(GTimg) + np.mean(np.log(GTimg) - np.log(np.abs(Reconimg))))))]

        accmax = np.maximum((Reconimg / GTimg), (GTimg / Reconimg)).reshape(4800)
        acc += [np.sum(i < 1.25 for i in accmax) / 4800]
        acc2 += [np.sum(i < (1.25 ** 2) for i in accmax) / 4800]
        acc3 += [np.sum(i < (1.25 ** 3) for i in accmax) / 4800]

        # fig = plt.figure(figsize=(20,4))
        # fig.suptitle('SCNN')
        # gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])
        # plt.subplot(gs[0])
        # plt.title(f'SCNN Reconstruction MASK max {min+1}')
        # plt.imshow(BReconimg)#(test_i[unseen].reshape(60, 80)) - (NEW_test[unseen].reshape(60, 80)), cmap='gnuplot2', vmin=-2, vmax=2)
        # plt.subplot(gs[1])
        # plt.title('SCNN Reconstruction from histogram with background')
        # plt.imshow(NEW_test[unseen].reshape(60, 80), cmap='gnuplot2', vmin=0, vmax=1)
        # # plt.colorbar()
        # plt.subplot(gs[2])
        # plt.title("{} Photon {} TIRF Histogram".format(photons, (1/tirfstd)))
        # plt.plot(test_histograms_new[unseen])
        # plt.subplot(gs[3])
        # plt.title('Ground truth')
        # plt.imshow(test_i[unseen].reshape(60, 80), cmap='gnuplot2', vmin=0, vmax=1)
        # # plt.colorbar()
        # # plt.show()
        # fig.savefig(
        #     f".{folder}/PHOTON{photons}TIRF{(1 / tirfstd)}_unseen_test_hist_max_{min+1}_{kse}_{ksd}_e{epochs}_f{feats}_b{batch}_{unseen}.png")
        # # print(ssim1)


    # Specify the file name
    file_name = 'output_{}.txt'.format(datetime.datetime.today())
    # Open the file in write mode
    with open(file_name, 'w') as f:
        print(f"IoU = {np.mean(IOUscnn)}, fullsnr = {np.mean(rSNR)}, rsnr = {np.mean(BrSNR)},\
          AbsRel = {np.mean(AbsRel)}, SqRel = {np.mean(SqRel)}, RMSE = {np.mean(RMSE)}, RMSElog = {np.mean(RMSElog)},\
          acc = {np.mean(acc)}, acc2 = {np.mean(acc2)}, acc3 = {np.mean(acc3)}")
        # plt.show()
        print("")
        print("")
        print("")
        print("Keras Huber Loss:", huber_loss, file=f)
        print("Keras Test Accuracy:", test_acc, file=f)
        print("Keras MSE:", msetest, file=f)
        print("Keras MAE:", maetest, file=f)

        print("Custom Mean Squared Error (MSE):", mse_custom, file=f)
        print("Custom Mean Absolute Error (MAE):", mae_custom, file=f)
        print("Structural Similarity Index (SSIM):", ssim, file=f)

    ###########################################################################################################################################################

    # model.save(r'PHOTON{}TIRF{}_{}_{}_e{}_f{}_b{}_{}.h5'.format(name, (1 / tirfstd), kse, ksd, epochs, feats, batch,
    #                                                             datetime.datetime.today()))

        print("SCNN UNSEEN:Huber loss = {}, Test_acc = {}, MSE = {}, MAE = {}".format(huber_loss, test_acc,
                                                                                        msetest, maetest), file=f)

    # print("")
    # #
    # # with open(file_name, 'w') as f:
        print(f"IoU = {np.mean(IOUscnn)}, fullsnr = {np.mean(rSNR)}, rsnr = {np.mean(BrSNR)},\
             AbsRel = {np.mean(AbsRel)}, SqRel = {np.mean(SqRel)}, RMSE = {np.mean(RMSE)}, RMSElog = {np.mean(RMSElog)},\
             acc = {np.mean(acc)}, acc2 = {np.mean(acc2)}, acc3 = {np.mean(acc3)}", file=f)

    # print("UNSEEN:{} {} Test Loss ={}, Test Acc = {}".format(photons, (1 / tirfstd), test_loss, test_acc))
# ########################################################################################################################
#




# train_histograms, train_images, test_histograms, test_images, train_scaled_images, test_scaled_images,...
# train_smoothed_histograms, test_smoothed_histograms, train_coarser_histograms, test_coarser_histograms, image1k, hist1k = loaddata()

(images, histograms, scaled_images, smoothed_histograms, coarser_histograms, scaled_images_train, scaled_images_test,
coarser_histograms_train, coarser_histograms_test, smoothed_histograms_train, smoothed_histograms_test, histograms_train, histograms_test) = loaddata1()

load_and_plot_data(images, scaled_images, histograms, smoothed_histograms, coarser_histograms)

features_list = [128]

for features in features_list:
    print(f"Running LIDAR with {features} features...")

    LIDAR(epochs=1, feats=features, batch=64, kern_int_e='he_normal', kern_int_d='he_normal', kse=5, ksd=5,
          kern_reg=None, train_i=scaled_images_train, test_i=scaled_images_test, train_h=histograms_train, test_h=histograms_test)
