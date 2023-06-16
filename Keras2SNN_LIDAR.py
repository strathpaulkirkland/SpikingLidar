# coding: utf-8

# # Converting a Keras model to a spiking neural network
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nengo/nengo-dl/blob/master/docs/examples/keras-to-snn.ipynb)
#
# A key feature of NengoDL is the ability to convert non-spiking networks into spiking networks. We can build both spiking and non-spiking networks in NengoDL, but often we may have an existing non-spiking network defined in a framework like Keras that we want to convert to a spiking network. The [NengoDL Converter](https://www.nengo.ai/nengo-dl/converter.html) is designed to assist in that kind of translation. By default, the converter takes in a Keras model and outputs an exactly equivalent Nengo network (so the Nengo network will be non-spiking). However, the converter can also apply various transformations during this conversion process, in particular aimed at converting a non-spiking Keras model into a spiking Nengo model.
#
# The goal of this notebook is to familiarize you with the process of converting a Keras network to a spiking neural network. Swapping to spiking neurons is a significant change to a model, which will have far-reaching impacts on the model's behaviour; we cannot simply change the neuron type and expect the model to perform the same without making any other changes to the model. This example will walk through some steps to take to help tune a spiking model to more closely match the performance of the original non-spiking network.

# In[ ]:
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import random
import gc
import keras
from scipy.interpolate import interp1d
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import applications, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling1D, Conv1D, Input, BatchNormalization, Conv2DTranspose, UpSampling2D, UpSampling1D, ZeroPadding1D
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2, l1, l1_l2

from matplotlib import gridspec
import datetime
# from sklearn.model_selection import train_test_split
# Helper libraries
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import random
import nengo
import nengo_dl
import glob

print(keras.__version__)

seed = 0
np.random.seed(seed)
random.seed(0)

# def loaddata1(data, tirfstd,  fig):
images = []
prehistograms = []
histograms = []
images_new = []
prehistograms_new = []
histograms_new = []
for i in range(1, 4):
    for j in range(1, 951):
        datafolder = np.load(r"/home/paul/PycharmProjects/LIDAR/images/image_{}_{}.npy".format(i, j))
        data2folder = np.load(r"/home/paul/PycharmProjects/LIDAR/histograms/histogram_{}_{}.npy".format(i, j))
        data2folder = data2folder[1, :].astype(np.int64)
        if data2folder.shape != (1249,):
            pad_width = (0, 1249 - data2folder.shape[0])
            data2folder = np.pad(data2folder, pad_width, mode='constant')
        images.append(datafolder)
        prehistograms.append(data2folder)

images = np.array(images)
histograms = np.vstack(prehistograms)
histograms = np.array(histograms)

window_size = 5
filterz = np.ones(window_size) / window_size
smoothed_histograms = np.zeros_like(histograms)
for i in range(histograms.shape[0]):
    smoothed_histograms[i] = np.convolve(histograms[i], filterz, mode='same')

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

num_levels = 2
threshold = 3300
value_range = images.max() - threshold
scaled_images = np.where(images > threshold, ((images - threshold) * (num_levels - 1) / value_range).astype(int) * (value_range / (num_levels - 1)) + threshold, images)


# Normalise the data
images[images > 4000] = 4000
images = images - np.min(images)
images = images / np.max(images)
histograms = histograms / np.max(histograms)

scaled_images[scaled_images > 4000] = 4000
scaled_images = scaled_images - np.min(scaled_images)
scaled_images = scaled_images / np.max(scaled_images)
smoothed_histograms = smoothed_histograms / np.max(smoothed_histograms)

print(histograms.shape)

# split the data into training and testing data
train_images, test_images, train_histograms, test_histograms = train_test_split(images, histograms, test_size=0.2, random_state=42)

train_scaled_images, test_scaled_images, train_smoothed_histograms, test_smoothed_histograms = train_test_split(
    scaled_images, smoothed_histograms, test_size=0.2, random_state=42)

# add an extra dimension with size 1 to the end of the arrays
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)
train_histograms = np.expand_dims(train_histograms, axis=-1)
test_histograms = np.expand_dims(test_histograms, axis=-1)

train_scaled_images = np.expand_dims(train_scaled_images, axis=-1)
test_scaled_images = np.expand_dims(test_scaled_images, axis=-1)
train_smoothed_histograms = np.expand_dims(train_smoothed_histograms, axis=-1)
test_smoothed_histograms = np.expand_dims(test_smoothed_histograms, axis=-1)

# print the shapes of the training and testing data
print("train_images shape:", train_images.shape)
print("train_histograms shape:", train_histograms.shape)
print("test_images shape:", test_images.shape)
print("test_histograms shape:", test_histograms.shape)

# Normalize the coarser_histograms
coarser_histograms = coarser_histograms / np.max(coarser_histograms)

# Split the data into training and testing data for coarser_histograms
train_coarser_histograms, test_coarser_histograms = train_test_split(
    coarser_histograms, test_size=0.2, random_state=42)

# Add an extra dimension with size 1 to the end of the arrays for coarser_histograms
train_coarser_histograms = np.expand_dims(train_coarser_histograms, axis=-1)
test_coarser_histograms = np.expand_dims(test_coarser_histograms, axis=-1)


# # add single timestep to training data
# train_histograms = train_histograms[:, None, :]
# train_images = train_images[:, None, :]

train_scaled_images = train_scaled_images.reshape((train_scaled_images.shape[0], 1, -1))
train_coarser_histograms = train_coarser_histograms.reshape((train_coarser_histograms.shape[0], 1, -1))
test_scaled_image = test_scaled_images.reshape((test_scaled_images.shape[0], 1, -1))
test_coarser_histograms = test_coarser_histograms.reshape((test_coarser_histograms.shape[0], 1, -1))

# when testing our network with spiking neurons we will need to run it
# over time, so we repeat the input/target data for a number of
# timesteps.
# n_steps = 300
# test_images = np.tile(test_images[:, None, :], (1, n_steps, 1))
# test_histograms = np.tile(test_histograms[:, None, :], (1, n_steps, 1))

# #             Converting a Keras model to a Nengo network
#
# Next we'll build a simple convolutional network. This architecture is chosen to be a quick and easy solution for
# this task; other tasks would likely require a different architecture, but the same general principles will apply.

#
# train_histograms = np.expand_dims((train_histograms),axis=-1)
# test_histograms = np.expand_dims((test_histograms),axis=-1)

# test_images = np.reshape(test_images, (test_images.shape[0],1,64,64))
# train_images = np.reshape(train_images, (train_images.shape[0],1,64,64))



#######################################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################
##########################LOADED DATA END ###################################################################################################################################################################################################
#######################################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################


# model.summary()
# Once the Keras model is created, we can pass it into the NengoDL Converter. The Converter tool is designed to
# automate the translation from Keras to Nengo as much as possible. You can see the full list of arguments the
# Converter accepts in the [documentation](https://www.nengo.ai/nengo-dl/reference.html?highlight=converter#nengo_dl
# .Converter).
# model = tensorflow.keras.models.load_model('KerasCNNLidar.h5')

folder = ''
scnn = sorted(glob.glob(f'.{folder}/PHOTON*.h5'))

model = tensorflow.keras.models.load_model(scnn[9])

# Now we are ready to train the network. It's important to note that we are using standard (non-spiking) ReLU neurons
# at this point.
#
# To make this example run a bit more quickly we've provided some pre-trained weights that will be downloaded below;
# set `do_training=True` to run the training yourself.

converter = nengo_dl.Converter(model, scale_firing_rates=1.0001)


# do_training = True
# if do_training:
#     with nengo_dl.Simulator(converter.net, minibatch_size=100) as sim:
#         # run training
#         sim.compile(
#             optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False),
#             loss="mse",
#             metrics=['accuracy'],
#         )
#
#         sim.fit(
#             {converter.inputs[inp]: train_histograms},
#             {converter.outputs[out]: train_images},
#             validation_data=(
#                 {converter.inputs[inp]: test_histograms},
#                 {converter.outputs[out]: test_images},
#             ),
#             epochs=60,
#         )
#
#         # save the parameters to file
#         sim.save_params("./Keras_SNN_LIDAR_params10")
# else:
#     # download pretrained weights
#     # urlretrieve(
#     #     "https://drive.google.com/uc?export=download&"
#     #     "id=1lBkR968AQo__t8sMMeDYGTQpBJZIs2_T",
#     #     "keras_to_snn_params.npz")
#     print("Loaded pretrained weights")


# #
# Now that we have our trained weights, we can begin the conversion to spiking neurons. To help us in this process
# we're going to first define a helper function that will build the network for us, load weights from a specified
# file, and make it easy to play around with some other features of the network.

# In[ ]:


def run_network(activation, params_file="Keras_SNN_LIDAR_params10", n_steps=60,
                scale_firing_rates=1.0001, synapse=None, n_test=10):
    # convert the keras model to a nengo network
    nengo_converter = nengo_dl.Converter(
        model,
        swap_activations={tensorflow.keras.activations.relu: activation},
        scale_firing_rates=scale_firing_rates,
        synapse=synapse,
    )
    # # turn off the optimizer completely, as to not change the number of tensors
    # with converter.net:
    #     nengo_dl.configure_settings(simplifications=[])

    # get input/output objects
    inp = model.inputs[0]
    out = model.outputs[0]
    conv1 = model.layers[2]
    nengo_input = nengo_converter.inputs[inp]
    nengo_output = nengo_converter.outputs[out]

    # add a probe to the first convolutional layer to record activity
    with nengo_converter.net:
        conv1_probe = nengo.Probe(nengo_converter.layers[conv1])

    # repeat inputs for some number of timesteps ##### Could edit this to do sequence data with LMUs
    tiled_test_coarser_histograms = np.tile(test_coarser_histograms[:n_test], (1, n_steps, 1))

    # set some options to speed up simulation
    with nengo_converter.net:
        nengo_dl.configure_settings(stateful=False)

    # build network, load in trained weights, run inference on test histograms
    with nengo_dl.Simulator(
            nengo_converter.net, minibatch_size=10,
            progress_bar=True) as nengo_sim:
        # nengo_sim.load_params(params_file)
        data = nengo_sim.predict({nengo_input: tiled_test_coarser_histograms})

    # compute accuracy on test data, using output of network on
    # last timestep

    predictions = data[nengo_output][:, -1]
    mse = ((predictions - test_scaled_image[:n_test, 0, :]) ** 2).mean()
    print(
        "###################################################################################################################################################")
    print("\nTest Batch MSE: %.5f%%\n" % mse)
    print("Activation Type: %s" % activation)
    print("Number of Steps: %d" % n_steps)
    print("Synapse=" + str(synapse))
    print("Scale=" + str(scale_firing_rates))
    print("\n\n")

    #####################################################################################################################
    # plot the results

    for ii in range(4):
        fig = plt.figure(figsize=(12, 4))

        gs0 = gridspec.GridSpec(1, 3)
        gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[2])
        gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[0])
        # plt.figure(figsize=(12, 4))

        plt.subplot(gs01[0])
        plt.title("Input Histogram")
        plt.plot(test_coarser_histograms[ii, 0])
        plt.axis('off')
        plt.subplot(gs01[1])
        plt.title("GT image")
        plt.imshow(test_scaled_image[ii, 0].reshape(60, 80), cmap='gnuplot2', vmin=0.02, vmax=1)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        sample_neurons = np.linspace(
            0,
            data[conv1_probe].shape[-1],
            1000,
            endpoint=False,
            dtype=np.int32,
        )
        activityrate = np.sum(data[conv1_probe] > 0) / np.size(data[conv1_probe]) * 100
        scaled_data = data[conv1_probe][ii, :, sample_neurons].T * scale_firing_rates
        if isinstance(activation, nengo.SpikingRectifiedLinear):
            scaled_data *= 0.001
            rates = np.sum(scaled_data, axis=0) / (n_steps * nengo_sim.dt)
            plt.ylabel('Number of spikes')
        else:
            rates = scaled_data
            plt.ylabel('Firing rates (Hz)')
        plt.xlabel('Timestep')
        plt.title(
            "Neural activities (conv1 mean=%dHz max=%dHz activity rate=%d%%)" % (
                rates.mean(), rates.max(), activityrate)
        )
        plt.plot(scaled_data)

        plt.subplot(gs0[2])
        plt.title("Output Predictions of Image Over Time")

        for xx in range(3):
            for yy in range(3):
                plt.subplot(gs00[xx, yy])
                # plt.title("Output Predictions of Image Over Time")
                # imgn = (3 * xx + yy + 1) * (n_steps / 10) + (n_steps / 10 - 1)
                plt.imshow(
                    (data[nengo_output][ii, int((3 * xx + yy + 1) * (n_steps / 10) + (n_steps / 10 - 1))]).reshape(60,80),
                    cmap='gnuplot2', vmin=0.02, vmax=1)
                # plt.legend([str(j) for j in range(10)], loc="upper left")
                # plt.xlabel('Timestep')
                # plt.ylabel("Probability")
                plt.axis('off')

        # plt.title("Output Predictions of Image Over Time")
        # plt.subplot(gs0[2])
        # plt.title("Output Predictions of Image Over Time")
        plt.tight_layout()

        from PIL import Image
        image_array = data[nengo_output][ii].reshape(60, 60, 80)
        for jj in range(n_steps):
            img_data = (image_array[jj] * 255 / image_array[jj].max()).astype(np.uint8)

            # Create a grayscale image
            height, width = 60, 80

            # Convert the image data to an 8-bit indexed color image
            img = Image.fromarray(img_data, mode='P')

            # Apply the colormap using the color palette
            colormap = plt.get_cmap('gnuplot2')
            colors = (colormap(np.arange(256))[:, :3] * 255).astype(np.uint8)
            img.putpalette(colors.ravel())

            # Save the image with the colormap
            file_name = f"image_instance_{ii}_timestep_{jj}.png"
            img.save(file_name)

            print(f"Saved {file_name}")

    plt.show()


########################################################################################################################
# #                             Now to run our trained network all we have to do is:

# run_network(activation=nengo.RectifiedLinear(), n_steps=10)

# Note that we're plotting the output over time for consistency with future plots, but since our network doesn't have
# any temporal elements (e.g. spiking neurons), the output is constant for each digit.
########################################################################################################################
########################################################################################################################

########################################################################################################################
# #                             Converting to a spiking neural network
#
# Now that we have the non-spiking version working in Nengo, we can start converting the network into spikes. Using
# the NengoDL converter, we can swap all the `relu` activation functions to `nengo.SpikingRectifiedLinear`.
#
# run_network(activation=nengo.SpikingRectifiedLinear(), n_steps=10)

# In this naive conversion we are getting random accuracy (~10%), which indicates that the network is not functioning
# well. Next, we will look at various steps we can take to improve the performance of the spiking model.
########################################################################################################################
########################################################################################################################

########################################################################################################################
# #                             Presentation time
#
# If we look at the neural activity plots above, we can see one thing that's going wrong: the activities are all
# zero! (The non-zero final output is just a result of the internal biases). Referring back to the neural activity
# plot from our non-spiking network further up, we can gain a bit of insight into why this occurs. We can see that
# the firing rates are all below 100 Hz. 100 Hz means that a neuron is emitting approximately 1 spike every 10
# timesteps (given the simulator timestep of 1ms). We're simulating for 10 time steps for each image, so we wouldn't
# really expect many of our neurons to be spiking within that 10 timestep window. If we present each image for longer
# we should start seeing some activity.
#
# run_network(
#     activation=nengo.SpikingRectifiedLinear(),
#     n_steps=50,
# )
# We can see now that while initially there's no network activity, eventually we do start getting some spikes. Note
# that although we start seeing spikes in the `conv1` layer around the 10th timestep, we don't start seeing activity
# in the output layer until around the 40th timestep. That is because each layer in the network is adding a similar
# delay as we see in `conv1`, so when you put those all together in series it takes time for the activity to
# propagate through to the final output layer.
########################################################################################################################
########################################################################################################################

########################################################################################################################
# #                             Synaptic smoothing
#
# Even with the increased presentation time, the test accuracy is still very low. This is because, as we can see in
# the output prediction plots, the network output is very noisy. Spikes are discrete events that exist for only a
# single time step and then disappear; we can see the literal "spikes" in the plots. Even if the neuron corresponding
# to the correct output is spiking quite rapidly, it's still not guaranteed that it will spike on exactly the last
# timestep (which is when we are checking the test accuracy).
#
# One way that we can compensate for this rapid fluctuation in the network output is to apply some smoothing to the
# spikes. This can be achieved in Nengo through the use of synaptic filters. The default `synapse` used in Nengo is a
# low-pass filter, and when we specify a value for the `synapse` parameter, that value is used as the low-pass filter
# time constant. When we pass a `synapse` value in the `run_network` function, it will create a low-pass filter with
# that time constant on the output of all the spiking neurons.
#
# Intuitively, we can think of this as computing a running average of each neuron's activity over a short window of
# time (rather than just looking at the spikes on the last timestep).
#
# Below we show results from the network running with three different low-pass filters. Note that adding synaptic
# filters will further increase the delay before neurons start spiking, because the filters will add their own "ramp
# up" time on each layer. So we'll run the network for even longer in these tests.
#
# for s in [0.001, 0.005, 0.01]:
#     print("Synapse=%.3f" % s)
#     run_network(
#         activation=nengo.SpikingRectifiedLinear(),
#         n_steps=120,
#         synapse=s,
#     )

# We can see that adding synaptic filtering smooths the output of the model and thereby improves the accuracy. With `synapse=0.01` we're achieving ~80% test accuracy; still not great, but significantly better than what we started with.
#
# However, increasing the magnitude of the synaptic filtering also increases the latency before we start seeing
# output activity. We can see that with `synapse=0.01` we don't start seeing output activity until around the 70th
# timestep. This means that with more synaptic filtering we have to present the input histograms for a longer period
# of time, which takes longer to simulate and adds more latency to the model's predictions. This is a common tradeoff
# in spiking networks (latency versus accuracy).
########################################################################################################################
########################################################################################################################

########################################################################################################################
# #                                 Firing rates
#
# Another way that we can improve network performance is by increasing the firing rates of the neurons. Neurons that
# spike more frequently update their output signal more often. This means that as firing rates increase,
# the behaviour of the spiking model will more closely match the original non-spiking model (where the neuron is
# directly outputting its true firing rate every timestep).
#
# #### Post-training scaling
#
# We can increase firing rates without retraining the model by applying a linear scale to the input of all the
# neurons (and then dividing their output by the same scale factor). Note that because we're applying a linear scale
# to the input and output, this will likely only work well with linear activation functions (like ReLU). To apply
# this scaling using the NengoDL Converter, we can use the `scale_firing_rates` parameter.

for scale in [10000]:
    print("Scale=%d" % scale)
    run_network(
        activation=nengo.SpikingRectifiedLinear(),
        scale_firing_rates=scale,
        synapse=0.002
    )

# We can see that as the frequency of spiking increases, the accuracy also increases. And we're able to achieve good
# accuracy (very close to the original non-spiking network) without adding too much latency.
#
# Note that if we increase the firing rates enough, spiking model eventually becomes equivalent to a non-spiking model:

# run_network(
#     activation=nengo.SpikingRectifiedLinear(),
#     scale_firing_rates=10000,
#     n_steps=150,
#     synapse=0.001
# )


# While this looks good from an accuracy perspective, it also means that we have lost many of the advantages of a
# spiking model (e.g. sparse communication, as indicated by the very high firing rates). This is another common
# tradeoff (accuracy versus firing rates) that can be customized depending on the demands of a particular application.
########################################################################################################################
########################################################################################################################

########################################################################################################################
# #                             Regularizing during training
#
# Rather than using `scale_firing_rates` to upscale the firing rates after training, we can also directly optimize
# the firing rates during training. We'll add loss functions that compute the mean squared error (MSE) between the
# output activity of each of the convolutional layers and some target firing rates we specify. We can think of this
# as applying L2 regularization to the firing rates, but we've shifted the regularization point from 0 to some target
# value.  One of the benefits of this method is that it is also effective for neurons with non-linear activation
# functions, such as LIF neurons.

# we'll encourage the neurons to spike at around 250Hz
# target_rate = 500
#
# # convert keras model to nengo network
# converter = nengo_dl.Converter(model, scale_firing_rates=1.0001)
#
# # add probes to the convolutional layers, which
# # we'll use to apply the firing rate regularization
# with converter.net:
#     inp = model.inputs[0]
#     out = model.outputs[0]
#     conv1 = model.layers[2]
#     output_p = converter.outputs[out]
#     conv1_p = nengo.Probe(converter.layers[conv1])
#     # conv5_p = nengo.Probe(converter.layers[conv5])
#
# with nengo_dl.Simulator(converter.net, minibatch_size=5) as sim:
#     # add regularization loss functions to the convolutional layers
#
#     sim.compile(
#         tensorflow.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False),
#         loss={
#             output_p: tensorflow.losses.mse,
#             conv1_p: tensorflow.losses.mse,
#             # conv5_p: tensorflow.losses.mse,
#         },
#         loss_weights={output_p: 1, conv1_p: 1e-3}#, conv5_p: 1e-3}
#     )
#
#     do_training = False
#     if do_training:
#         # run training (specifying the target rates for the convolutional layers)
#         sim.fit(
#             {converter.inputs[inp]: train_histograms},
#             {
#                 output_p: train_images,
#                 conv1_p: np.ones((train_images.shape[0], 1, conv1_p.size_in))
#                 * target_rate,
#                 #conv5_p: np.ones((train_images.shape[0], 1, conv5_p.size_in))
#                 #* target_rate,
#             },
#             epochs=5)
#
#         # save the parameters to file
#         sim.save_params("./Keras_SNN_LIDAR_regularized_params")
#     else:
#         # download pretrained weights
#         # urlretrieve(
#         #     "https://drive.google.com/uc?export=download&"
#         #     "id=1xvIIIQjiA4UM9Mg_4rq_ttBH3wIl0lJx",
#         #     "keras_to_snn_regularized_params.npz")
#         print("Loaded pretrained weights")

########################################################################################################################
#       Now we can examine the firing rates in the non-spiking network.
#
# run_network(
#     activation=nengo.RectifiedLinear(),
#     params_file="Keras_SNN_LIDAR_regularized_params",
#     n_steps=10,
# )


# In the neuron activity plot we can see that the firing rates are around the magnitude we specified (we could adjust
# the regularization function/weighting to refine this further).
########################################################################################################################
########################################################################################################################

########################################################################################################################
#       Now we can convert it to spiking neurons, without applying any scaling.

# run_network(
#     activation=nengo.SpikingRectifiedLinear(),
#     params_file="Keras_SNN_LIDAR_regularized_params",
#     synapse=0.01,
# )


# We can see that this network, because we trained it with spiking neurons in mind, can be converted to a spiking
# network without losing much performance or requiring any further tweaking.

# #                            Conclusions
#
# In this example we've gone over the process of converting a non-spiking Keras model to a spiking Nengo network.
# We've shown some of the common issues that crop up, and how to go about diagnosing/addressing them. In particular,
# we looked at presentation time, synaptic filtering, and firing rates, and how adjusting those factors can affect
# various properties of the model (such as accuracy, latency, and temporal sparsity).  Note that a lot of these
# factors represent tradeoffs that are application dependent. The particular parameters that we used in this example
# may not work or make sense in other applications, but this same workflow and thought process should apply to
# converting any kind of network to a spiking Nengo model.
