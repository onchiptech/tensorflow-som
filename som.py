import tensorflow as tf
import numpy as np
from tensorflow.raw_ops import Pack as pack

class SOM(tf.keras.layers.Layer):

    def __init__(self, m, n, dim, num_epochs=100, alpha=None, sigma=None, name=None):
        
        super(SOM, self).__init__(name=name)
        # Assign required variables first
        self.m = m; self.n = n; self.dim=dim
        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = max(m, n) / 2.0
        else:
            self.sigma = float(sigma)
        
        self.num_epochs = abs(int(num_epochs))
        # To save data, create weight vectors and their location vectors
        #self.weightage_vects = tf.random_normal( [m * n, dim])

        self.location_vects = tf.constant(np.array(list(self.neuron_locations(m, n))))

    def call(self, inputs, training=False, epoch=None):        
            
        input_vector, weights = inputs

        input_vector = pack(values=[input_vector for _ in range(self.m * self.n)])
        # Training Operation  # tf.pack result will be [ (m*n),  dim ]
        diff = tf.math.subtract(weights, input_vector)
        distance = tf.math.sqrt(tf.reduce_sum(tf.math.pow(diff, 2), axis=1))
        bmu_index = tf.math.argmin(distance, 0) 
                
        slice_input = tf.pad(tf.reshape(bmu_index, [1]), np.array([[0, 1]]))
        bmu_loc = tf.reshape(tf.slice(self.location_vects, slice_input, 
                    tf.constant(np.array([1, 2]), dtype=tf.int64) ), [2])                                    

        if training == False:
          return bmu_loc

        # Iteration number
        epoch = tf.cast(epoch, tf.float32)

        # To compute the alpha and sigma values based on iteration number
        learning_rate_op = tf.math.subtract(1.0, tf.math.divide(epoch, self.num_epochs))
        alpha_op = tf.math.multiply(self.alpha, learning_rate_op)
        sigma_op = tf.math.multiply(self.sigma, learning_rate_op)

        # learning rates for all neurons, based on iteration number and location w.r.t. BMU.
        diff = tf.math.subtract(self.location_vects, pack( values=[bmu_loc for _ in range(self.m * self.n)] ) )
        bmu_distance_squares = tf.reduce_sum(tf.math.pow(diff, 2), axis=1)

        neighbourhood_func = tf.math.exp(tf.math.negative(tf.math.divide(tf.cast(
            bmu_distance_squares, tf.float32), tf.math.pow(sigma_op, 2))))

        learning_rate_op = tf.math.multiply(alpha_op, neighbourhood_func)

        # Finally, the op that will use learning_rate_op to update the weightage vectors of all neurons
        learning_rate_multiplier = pack(values=[tf.tile(tf.slice(
            learning_rate_op, np.array([i]), np.array([1])), [self.dim]) for i in range(self.m * self.n)] )

        ### Strucutre of updating weight ###
        ### W(t+1) = W(t) + W_delta ###
        ### wherer, W_delta = L(t) * ( V(t)-W(t) ) ###

        # W_delta = L(t) * ( V(t)-W(t) )
        weightage_delta = tf.math.multiply(
            learning_rate_multiplier,
            tf.math.subtract(input_vector, weights))

        # W(t+1) = W(t) + W_delta
        new_weights = tf.math.add(weights, weightage_delta)

        return bmu_loc, new_weights

    def neuron_locations(self, m, n):

        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

if __name__ == "__main__":
    

       
        dim = 10
        # This is more neurons than you need but it makes the visualization look nicer
        m = 20
        n = 20
       
        X = np.ones((dim,)).astype(np.float32)

        # Build the SOM object and place all of its ops on the graph
        som = SOM(m=m, n=n, dim=dim, num_epochs=20, name='SOM')
        
        @tf.function
        def train_graph(x, weights, epoch):
            inputs = (x, weights)
            loc, new_weights = som(inputs, training=True, epoch=epoch)
            return loc, new_weights

        @tf.function
        def infer_graph(x, weights):
            inputs = (x, weights)
            loc = som(inputs, training=False)
            return loc

        print('Training...')
        weigths = np.random.normal(size=m * n * dim).reshape((m * n, dim))
        loc, new_weights = train_graph(X, weigths, 0)
        print('loc: ', loc.shape)
        print('new_weights: ', new_weights.shape)
        print('new_weights: ', new_weights)
        
        print('Inference...')
        loc = infer_graph(X, new_weights)
        print('loc: ', loc.shape)
        