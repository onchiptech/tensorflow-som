import tensorflow as tf
import numpy as np
from tensorflow.raw_ops import Pack as pack
from matplotlib import pyplot as plt

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

        # input_vector shape -> [batch_size, m * n, dim]
        input_vector = tf.tile(tf.expand_dims(input_vector, axis=1), [1, self.m * self.n, 1])
        
        # weights shape -> [1, m * n, dim]
        weights = tf.expand_dims(weights, axis=0)
        
        # diff shape -> [batch_size, m * n, dim]
        diff = tf.math.subtract(weights, input_vector)

        # distance shape -> [batch_size, m * n]
        distance = tf.math.sqrt(tf.reduce_sum(tf.math.pow(diff, 2), axis=2))

        # bmu_index shape -> [batch_size, m * n]
        bmu_index = tf.math.argmin(distance, axis=1) 
                
        #slice_input = tf.pad(tf.reshape(bmu_index, [1]), np.array([[0, 1]]))
        #bmu_loc = tf.reshape(tf.slice(self.location_vects, slice_input, 
        #            tf.constant(np.array([1, 2]), dtype=tf.int64) ), [2])                                    
        bmu_loc = tf.reshape(tf.gather(self.location_vects, bmu_index), [-1, 2])
        if training == False:
          return bmu_loc

        # Iteration number
        epoch = tf.cast(epoch, tf.float32)
        

        # To compute the alpha and sigma values based on iteration number
        learning_rate_op = tf.math.subtract(1.0, tf.math.divide(epoch, self.num_epochs))
        alpha_op = tf.math.multiply(self.alpha, learning_rate_op)
        sigma_op = tf.math.multiply(self.sigma, learning_rate_op)

        # Start by getting the squared difference between each BMU location and every other unit in the map
        # bmu_loc is [batch_size, 2], i.e. the coordinates of the BMU for each input vector.
        # location vects shape should be [1, num_neurons, 2]
        # bmu_loc should be [batch_size, 1, 2]
        # Output needs to be [batch_size, num_neurons], i.e. a row vector of distances for each input item
        bmu_distance_squares = tf.reduce_sum(input_tensor=tf.math.pow(tf.math.subtract(
                tf.expand_dims(self.location_vects, axis=0),
                tf.expand_dims(bmu_loc, axis=1)), 2), axis=2)

        # location vects shape should be [1, num_neurons, 2]
        # bmu_loc should be [batch_size, 1, 2]
        # diff should be [batch_size, num_neurons, 2]
        diff = tf.math.subtract(tf.expand_dims(self.location_vects, axis=0),  tf.expand_dims(bmu_loc, axis=1))
        
        # bmu_distance_squares -> [batch_size, num_neurons]
        bmu_distance_squares = tf.cast(tf.reduce_sum(tf.math.pow(diff, 2), axis=2), tf.float32)
        bmu_distance = tf.math.divide(bmu_distance_squares, tf.math.pow(sigma_op, 2))
        neighbourhood_func = tf.math.exp(tf.math.negative(bmu_distance))
        
        # learning_rate_op -> [batch_size, num_neurons]
        learning_rate_op = tf.math.multiply(alpha_op, neighbourhood_func)
        learning_rate_multiplier = tf.expand_dims(learning_rate_op, axis=-1)
        # Finally, the op that will use learning_rate_op to update the weightage vectors of all neurons
        #learning_rate_multiplier = pack(values=[tf.tile(tf.slice(
        #    learning_rate_op, np.array([i]), np.array([1])), [self.dim]) for i in range(self.m * self.n)] )

        ### Strucutre of updating weight ###
        ### W(t+1) = W(t) + W_delta ###
        ### wherer, W_delta = L(t) * ( V(t)-W(t) ) ###

        # W_delta = L(t) * ( V(t)-W(t) )
        #weightage_delta -> [batch_size, num_neurons, dim]
        weightage_delta = tf.math.multiply(
            learning_rate_multiplier,
            tf.math.subtract(input_vector, weights))

        # W(t+1) = W(t) + W_delta
        new_weights = tf.reduce_mean(tf.math.add(weights, weightage_delta), axis=0)

        return new_weights
        


    def neuron_locations(self, m, n):

        for i in range(m):
            for j in range(n):
                yield np.array([i, j])


    def get_centroids(self, weights):

        centroid_grid = [[] for i in range(self.m)]
        locations = list(self.neuron_locations(self.m, self.n))
        for i, loc in enumerate(locations):
            centroid_grid[loc[0]].append(weights[i])

        return centroid_grid            

if __name__ == "__main__":
    

    """      
    dim = 10
    # This is more neurons than you need but it makes the visualization look nicer
    m = 20
    n = 20
    
    X = np.ones((1,dim)).astype(np.float32)

    # Build the SOM object and place all of its ops on the graph
    som = SOM(m=m, n=n, dim=dim, num_epochs=20, name='SOM')
    
    @tf.function
    def train_graph(x, weights, epoch):
        inputs = (x, weights)
        new_weights = som(inputs, training=True, epoch=epoch)
        return new_weights

    @tf.function
    def infer_graph(x, weights):
        inputs = (x, weights)
        loc = som(inputs, training=False)
        return loc

    print('Training...')
    weigths = np.random.normal(size=m * n * dim).reshape((m * n, dim))
    new_weights = train_graph(X, weigths, 0)        
    print('new_weights: ', new_weights.shape)
    print('new_weights: ', new_weights)
    
    print('Inference...')
    loc = infer_graph(X, new_weights)
    print('loc: ', loc.shape)
    """
                
    #Training inputs for RGBcolors
    colors = np.array(
        [[0., 0., 0.],
          [0., 0., 1.],
          [0., 0., 0.5],
          [0.125, 0.529, 1.0],
          [0.33, 0.4, 0.67],
          [0.6, 0.5, 1.0],
          [0., 1., 0.],
          [1., 0., 0.],
          [0., 1., 1.],
          [1., 0., 1.],
          [1., 1., 0.],
          [1., 1., 1.],
          [.33, .33, .33],
          [.5, .5, .5],
          [.66, .66, .66]])
    color_names = \
        ['black', 'blue', 'darkblue', 'skyblue',
        'greyblue', 'lilac', 'green', 'red',
        'cyan', 'violet', 'yellow', 'white',
        'darkgrey', 'mediumgrey', 'lightgrey']
    
    #Train a 20x30 SOM with 400 iterations
    m = 20
    n = 30
    dim = 3
    num_epochs = 400
    som = SOM(m=m, n=n, dim=dim, num_epochs=num_epochs, name='SOM')
    @tf.function
    def train_graph(x, weights, epoch):
        inputs = (x, weights)
        new_weights = som(inputs, training=True, epoch=epoch)
        return new_weights

    @tf.function
    def infer_graph(x, weights):
        inputs = (x, weights)
        loc = som(inputs, training=False)
        return loc    
    print('Training...')
    num_inputs = len(colors)
    weights = np.random.normal(size=m * n * dim).reshape((m * n, dim))
    _colors = tf.convert_to_tensor(colors)
    weights = tf.convert_to_tensor(weights)
    for epoch in range(num_epochs):      
      #for i in range(num_inputs):
      #  weights = train_graph(_colors[i:i+1], weights, epoch).numpy()
      weights = train_graph(_colors, weights, tf.convert_to_tensor(epoch))
    weights = weights.numpy()
    #Get output grid
    image_grid = som.get_centroids(weights)
    print('Inference...')
    #Map colours to their closest neurons
    mapped = infer_graph(colors, weights).numpy().tolist()
    
    #Plot
    plt.imshow(image_grid)
    plt.title('Color SOM')
    for i, m in enumerate(mapped):
        plt.text(m[1], m[0], color_names[i], ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.show()