#!/usr/bin/env python3
import numpy as np

import ipdb

from scipy.signal import fftconvolve

def dim_activation_conv(w, X, v=[], Y=np.array([]), iterations=30, trueRange=[]):
    """
    TODO change MATLAB comment
    w = a cell array of size {N}, where N is the number of distinct neuron types.
        Each element w{i} is a 3-dimensional matrix. w{i}(:,:,j) is a convolution
        mask specifying the synaptic weights for neuron type i's RF in input channel
        j.
    X = a three dimensional matrix. X(a,b,j) specifies the bottom-up input to
        the DIM neural network at location a,b in channel j.
    v = synaptic weights, defined as for w, but strength normalised differently.
    Y = a three dimensional matrix. Y(a,b,i) specifies the prediction node
        activation for type i neurons at location a,b.
    R = a three dimensional matrix. R(a,b,j) specifies the reconstruction of the input at location a,b in channel j
    E = a three dimensional matrix. E(a,b,j) specifies the error in the reconstruction of the input at loaction a,b in channel j
    iterations = number of iterations performed by the DIM algorithm. Default is 30.
    trueRange = range of Y, E and R to keep at the end of processing. Used to try to
                avoid edge effects by having input (X) larger than original image, and
                then cropping edges of outputs, so that ouputs are resized to be same
                size as the original image.

    Python:
    w = templates
    X = query image"""

    # Simulate grayscale
    # X = X[:,:,:2]
    # for i in range(len(w)):
    #     w[i] = w[i][:,:,:2]

    # Shape of query image
    a, b, nInputChannels = X.shape
    # Number of templates
    nMasks = len(w)
    # Shape of templates
    c, d, nChannels = w[0].shape

    if len(v) == 0:
        # Set feedback weights equal to feedforward weights normalized by maximum value for each node
        for i in range(nMasks): # Loop for each template
            denominator = max(1e-6, np.max(w[i])) # Prevent division by 0, use either 1e-6 or maximum value in template

            # Set v equal to the template / denominator (max of template)
            # This scales w to between 0 and 1
            # If value is < 0, set equal to 0 - shouldnt be necessary but was in MATLAB code so will keep here
            v.append(np.maximum(0, w[i] / denominator))


    # I think Y will record the similarity map for each template
    if len(Y) == 0:
        Y = np.zeros((a, b, nMasks), dtype=np.float32)

    for i in range(nMasks):
        # Normalise feedforward weights _to sum to one_ for each node
        w[i] = w[i] / np.maximum(1e-6, np.sum(w[i]))

        # Rotate feedforward weights so that convolution can be used to apply the filtering
        # So really its a correlation that is being caculated
        w[i] = np.rot90(w[i], 2)

    # v -> list of len 4, containing np arrays of shape (45, 37, 6)
    # np.stack(v, axis=3) -> stack synaptic weights for each template on third axis (45, 37, 6, 4)
    # np.sum(..., 3) -> sum stacked template weights (45, 37, 6)
    # np.sum(..., 0) -> sum along 0 axis (37, 6)
    # np.sum(..., 0) -> sum along 0 axis (6)
    sumV = np.sum(np.sum(np.sum(np.stack(v, axis=3), 3), 0), 0)
    # So sumV:
    #  - Sums template weights together
    #  - Then sums to find maximum weight for each summed channel

    # Set parameters
    epsilon2 = 1e-2
    epsilon1 = epsilon2 / np.max(sumV)

    # Iterate DIM equations to determine neural responses
    # iterations = 10
    for t in range(iterations):
        # print(f"dim_conv: {t}")


        # Update error-detecting neuron responses
        R = np.zeros((a,b,nInputChannels), dtype="float32") # R is same size as query image
        # for j in range(nChannels): # Number of channels per image, 6 for the processed images
        for i in range(nMasks): # Number of templates

            # Calc predictive reconstruction of the input
            # for i in range(nMasks): # Number of templates
            for j in range(nChannels): # Number of channels per image, 6 for the processed images
                # Sum reconstruction over each RF type
                if not ( # Skip empty filters and response arrays: they don't add anything
                        v[i][:,:,j].size == 0 or np.all(v[i][:,:,j] == 0) or
                        Y[:,:,i].size == 0 or np.all(Y[:,:,i] == 0)
                ):
                    # Convolve the current template correlation map (Y[:,:,i]) with synaptic weights v
                    # Note that these are stored in R[:,:,j], so the resulting convolution for each template is summed
                    # to the same respecive channel
                    # Since these are all summed, this must be the part in which the explaining away happens
                    # Remember that a convolution multiplies element wise and then adds all of the results
                    # Ok so this seems to result in a non-flipped result
                    #
                    # Locations in the feature map with a higher value result in the convolution
                    R[:, :, j] += fftconvolve(Y[:,:,i], v[i][:,:,j], mode='same') #

        # ipdb.set_trace()
        R[R<0] = 0

        # Calc error between reconstruction and actual input: using values of input
        # that change over time (if these are provided)
        # ipdb.set_trace()
        # E is the error minimised in gradient descent
        E = X / np.maximum(epsilon2, R)


        # Update prediction neuron responses
        # This is the gradient descent part somehow
        # Remember that gradient descent uses the ideal result and output result to adjust the neurons closer to what
        # would have lead to the ideal result
        for i in range(nMasks): # Number of templates
            input = np.zeros((a, b), dtype='float32')

            # Correlate for each channel of template
            for j in range(nChannels):
                # Sum inputs to prediction neurons from each channel
                if not ( # Skip empty filters and error arrays: they don't add anything
                        w[i][:,:,j].size == 0 or np.all(w[i][:,:,j] == 0) or
                        E[:,:,j].size == 0 or np.all(E[:,:,j] == 0)
                ):
                    # Areas with the greatest error will result in input being larger at that location
                    # However input is greatest at best match location, _why_?
                    #
                    # I need to take into account the multiple templates
                    input += fftconvolve(E[:,:,j], w[i][:,:,j], mode='same')

            # if t == 9:
            #     ipdb.set_trace()

            # Multiply Y (or epsilon 1 if Y is less than it) by the resulting summed correlation map
            Y[:, :, i] = np.maximum(epsilon1, Y[:,:,i]) * input

        Y[Y<0] = 0

    if len(trueRange) > 0:
        # R = R[np.ix_(trueRange[0], trueRange[1], np.arange(Y.shape[2]))]
        # E = E[np.ix_(trueRange[0], trueRange[1], np.arange(Y.shape[2]))]
        Y = Y[np.ix_(trueRange[0], trueRange[1], np.arange(Y.shape[2]))]

    # return Y, E, R
    # Y is of the shape (x, y, num_tempaltes)
    # The 2nd axis correlates to how well each template matched
    return Y
