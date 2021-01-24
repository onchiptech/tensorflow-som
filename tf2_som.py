# MIT License
#
# Copyright (c) 2018 Chris Gorman
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =================================================================================
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

__author__ = "Chris Gorman"
__email__ = "chris@cgorman.net"

"""
Adapted from code by Sachin Joglekar
https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
"""


class SelfOrganizingMap(tf.keras.layers.Layer):
    """
    2-D rectangular grid planar Self-Organizing Map with Gaussian neighbourhood function
    """

    def __init__(self, m, n, dim, max_epochs=100, initial_radius=None, batch_size=128, initial_learning_rate=0.1,
                 std_coeff=0.5, softmax_activity=False, output_sensitivity=-1.0, name='Self-Organizing-Map'):
        """
        Initialize a self-organizing map on the tensorflow graph
        :param m: Number of rows of neurons
        :param n: Number of columns of neurons
        :param dim: Dimensionality of the input data
        :param max_epochs: Number of epochs to train for
        :param initial_radius: Starting value of the neighborhood radius - defaults to max(m, n) / 2.0
        :param batch_size: Number of input vectors to train on at a time
        :param initial_learning_rate: The starting learning rate of the SOM. Decreases linearly w/r/t `max_epochs`
        :param graph: The tensorflow graph to build the network on
        :param std_coeff: Coefficient of the standard deviation of the neighborhood function
        :param model_name: The name that will be given to the checkpoint files
        :param softmax_activity: If `True` the activity will be softmaxed to form a probability distribution
        :param gpus: The number of GPUs to train the SOM on
        :param output_sensitivity The constant controlling the width of the activity gaussian. Numbers further from zero
                elicit activity when distance is low, effectively introducing a threshold on the distance w/r/t activity.
                See the plot in the readme file for a little introduction.
        :param session: A `tf.Session()` for executing the graph
        """

        super(SelfOrganizingMap, self).__init__(name=name)
        self._m = abs(int(m))
        self._n = abs(int(n))
        self._dim = abs(int(dim))
        if initial_radius is None:
            self._initial_radius = max(m, n) / 2.0
        else:
            self._initial_radius = float(initial_radius)
        self._max_epochs = abs(int(max_epochs))
        self._batch_size = abs(int(batch_size))
        self._std_coeff = abs(float(std_coeff))
        self._softmax_activity = bool(softmax_activity)

        if output_sensitivity > 0:
            output_sensitivity *= -1
        elif output_sensitivity == 0:
            output_sensitivity = -1
        # The activity equation is kind of long so I'm naming this c for brevity
        self._c = float(output_sensitivity)

        # Initialized later, just declaring up here for neatness and to avoid warnings
        self._weights = None
        self._location_vects = None
        self._epoch = None
        self._training_op = None
        self._centroid_grid = None
        self._locations = None
        self._activity_op = None
        self._saver = None
        self._merged = None
        self._activity_merged = None
        # This will be the collection of summaries for this subgraph. Add new summaries to it and pass it to merge()
        self._initial_learning_rate = initial_learning_rate
        #tf.Variable(initial_value=initial_value, name='weights')
        # Matrix of size [m*n, 2] for SOM grid locations of neurons.
        # Maps an index to an (x,y) coordinate of a neuron in the map for calculating the neighborhood distance
        self._location_vects = tf.constant(np.array(
            list(self._neuron_locations())), name='Location_Vectors')

    def _neuron_locations(self):
        """ Maps an absolute neuron index to a 2d vector for calculating the neighborhood function """
        for i in range(self._m):
            for j in range(self._n):
                yield np.array([i, j])

    def call(self, inputs, training=False, epoch=None):
                
        # Randomly initialized weights for all neurons, stored together
        # as a matrix Variable of shape [num_neurons, input_dims]
        
        # Start by computing the best matching units / winning units for each input vector in the batch.
        # Basically calculates the Euclidean distance between every
        # neuron's weight vector and the inputs, and returns the index of the neurons which give the least value
        # Since we are doing batch processing of the input, we need to calculate a BMU for each of the individual
        # inputs in the batch. Will have the shape [batch_size]


        # Distance between weights and the input vector
        # Note we are reducing along 2nd axis so we end up with a tensor of [batch_size, num_neurons]
        # corresponding to the distance between a particular input and each neuron in the map
        # Also note we are getting the squared distance because there's no point calling sqrt or tf.norm
        # if we're just doing a strict comparison        
        input_vect, weights = inputs
        squared_distance = tf.reduce_sum(
            input_tensor=tf.pow(tf.subtract(tf.expand_dims(weights, axis=0),
                                tf.expand_dims(input_vect, axis=1)), 2), axis=2)

        # Get the index of the minimum distance for each input item, shape will be [batch_size],
        bmu_indices = tf.argmin(input=squared_distance, axis=1)

        # This will extract the location of the BMU in the map for each input based on the BMU's indices
        # Using tf.gather we can use `bmu_indices` to index the location vectors directly
        bmu_locs = tf.reshape(tf.gather(self._location_vects, bmu_indices), [-1, 2])

        if training == False:
            return bmu_locs
        else:
            epoch = tf.cast(epoch, tf.float32)
            # With each epoch, the initial sigma value decreases linearly
            radius = tf.subtract(self._initial_radius,
                                    tf.multiply(epoch,
                                                tf.divide(tf.cast(tf.subtract(self._initial_radius, 1),
                                                                tf.float32),
                                                        tf.cast(tf.subtract(self._max_epochs, 1),
                                                                tf.float32))))

            alpha = tf.multiply(self._initial_learning_rate,
                                tf.subtract(1.0, tf.divide(tf.cast(epoch, tf.float32),
                                                            tf.cast(self._max_epochs, tf.float32))))
                
            # Construct the op that will generate a matrix with learning rates for all neurons and all inputs,
            # based on iteration number and location to BMU

            # Start by getting the squared difference between each BMU location and every other unit in the map
            # bmu_locs is [batch_size, 2], i.e. the coordinates of the BMU for each input vector.
            # location vects shape should be [1, num_neurons, 2]
            # bmu_locs should be [batch_size, 1, 2]
            # Output needs to be [batch_size, num_neurons], i.e. a row vector of distances for each input item
            bmu_distance_squares = tf.reduce_sum(input_tensor=tf.pow(tf.subtract(
                tf.expand_dims(self._location_vects, axis=0),
                tf.expand_dims(bmu_locs, axis=1)), 2), axis=2)

            # Using the distances between each BMU, construct the Gaussian neighborhood function.
            # Basically, neurons which are close to the winner will move more than those further away.
            # The radius tensor decreases the width of the Gaussian over time, so early in training more
            # neurons will be affected by the winner and by the end of training only the winner will move.
            # This tensor will be of shape [batch_size, num_neurons] as well and will be the value multiplied to
            # each neuron based on its distance from the BMU for each input vector
            neighbourhood_func = tf.exp(tf.divide(tf.negative(tf.cast(
                bmu_distance_squares, "float32")), tf.multiply(
                tf.square(tf.multiply(radius, self._std_coeff)), 2)))

            # Finally multiply by the learning rate to decrease overall neuron movement over time
            learning_rate_op = tf.multiply(neighbourhood_func, alpha)            
            # The batch formula for SOMs multiplies a neuron's neighborhood by all of the input vectors in the batch,
            # then divides that by just the sum of the neighborhood function for each of the inputs.
            # We are writing this in a way that performs that operation for each of the neurons in the map.


            # The numerator needs to be shaped [num_neurons, dimensions] to represent the new weights
            # for each of the neurons. At this point, the learning rate tensor will be
            # shaped [batch_size, neurons].
            # The end result is that, for each neuron in the network, we use the learning
            # rate between it and each of the input vectors, to calculate a new set of weights.
            numerator = tf.reduce_sum(input_tensor=tf.multiply(tf.expand_dims(learning_rate_op, axis=-1),
                                                    tf.expand_dims(input_vect, axis=1)), axis=0)

            # The denominator is just the sum of the neighborhood functions for each neuron, so we get the sum
            # along axis 1 giving us an output shape of [num_neurons]. We then expand the dims so we can
            # broadcast for the division op. Again we transpose the learning rate tensor so it's
            # [num_neurons, batch_size] representing the learning rate of each neuron for each input vector
            denominator = tf.expand_dims(tf.reduce_sum(input_tensor=learning_rate_op,
                                                        axis=0) + float(1e-12), axis=-1)

            # With multi-gpu training we collect the results and do the weight assignment on the CPU
            # Divide them            
            weights = tf.divide(numerator, denominator)
            #self._weights.assign(new_weights)
            
            return bmu_locs, weights


if __name__ == "__main__":
    

       
        dims = 10
        # This is more neurons than you need but it makes the visualization look nicer
        m = 20
        n = 20
        batch_size = 8

        X = np.ones((batch_size, dims)).astype(np.float32)

        # Build the SOM object and place all of its ops on the graph
        som = SelfOrganizingMap(m=m, n=n, dim=dims, max_epochs=20, batch_size=batch_size, initial_learning_rate=0.1)
        
        @tf.function
        def train_graph(x, epoch):
            arg1, arg2 = som(x, training=True, epoch=epoch)
            return arg1, arg2

        @tf.function
        def infer_graph(x):
            arg1 = som(x, training=False)
            return arg1

        arg1, arg2 = train_graph(X, 0)
        print('arg1: ', arg1.shape)
        print('arg2: ', arg2.shape)
        arg1 = infer_graph(X)
        print('arg1: ', arg1.shape)
        print('output_weights: ', som.output_weights)


