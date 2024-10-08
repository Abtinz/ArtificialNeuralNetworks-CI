
from math import sqrt
import numpy as np

# The Conv2D class is an implementation of a 2D convolutional layer for use in a convolutional neural network. 
# This layer is responsible for extracting features from the input data by sliding a kernel over the input and 
# computing the dot product between the kernel and the input at each image position.
class Conv2D:
    #properties
    # in_channels -> the number of input data channels
    # out_channels -> the number of output data channels
    # name -> a string identifier for the layer, we will use it for distinguishing con layers from fully connected layers
    # kernel_size ->  the size of the kernel for height and width
    # stride -> the stride(steps) for the kernel when sliding over the input, its for height and width sliding.
    # padding: the amount of padding to add to the input
    # initialize_method: a string indicating the method to use for initializing the weights of the kernel
    # parameters -> layer kernel weights and bias
    
    def __init__(self, in_channels, out_channels, name, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), initialize_method="random"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.name = name
        self.initialize_method = initialize_method
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size #for tuple mode or directive integer passing
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        #initializing kernel weights and bias
        self.parameters = [self.initialize_weights(), self.initialize_bias()]


    #  Initializing kernel weights.
    # returns: weights: initialized kernel with shape: (kernel_size[0], kernel_size[1], in_channels, out_channels)
    def initialize_weights(self):
       
        if self.initialize_method == "random":
            # Initialize weights with random values from a normal distribution
            return np.random.randn(self.in_channels ,self.out_channels, self.kernel_size[0], self.kernel_size[1])

        elif self.initialize_method == "xavier":
            # Initialize weights with random values from a uniform distribution with a range determined by the Xavier initialization
            range_value = np.sqrt( 1 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1] + self.out_channels))
            return np.random.uniform(-range_value, range_value, (self.in_channels ,self.out_channels, self.kernel_size[0], self.kernel_size[1]))
         
        elif self.initialize_method == "he":
            # Initialize weights with random values from a normal distribution with a standard deviation determined by the He initialization
            return np.random.randn(self.in_channels ,self.out_channels, self.kernel_size[0], self.kernel_size[1]) *  sqrt(2 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1]))

        else:
            raise ValueError("Invalid initialization method")
    
    
    # Initialize bias with zeros and layer out put count which will show the number of our perceptrons.      
    def initialize_bias(self):
        return np.zeros((self.out_channels, 1))

    # this function will calculate the shape of the convolutional layer output.
    # argument -> input_shape: shape of the input of the convolutional layer
    #       batch_size -> the number of examples in a batch.
    #       chanel_count is the number of channels in each example, which is typically 1 for grayscale images or 3 for RGB images! ...
    #       height is the height of the input image in pixels.
    #       width is the width of the input image in pixels.
    # returns -> target_shape: shape of the output of the convolutional layer -> batch_size , output Chanel's count and dimensions
    def target_shape(self, input_shape):
        
        #extracting input and kernel and stride and padding for output shape calculation
        batch_size, _ , height, width = input_shape
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        padding_height, padding_width = self.padding
        
        # Calculate the height and width of the output
        output_height = (height + 2*padding_height - kernel_height)/stride_height + 1
        output_width = (width + 2*padding_width - kernel_width)/stride_width + 1
        chanel_count = self.out_channels

        return (batch_size, chanel_count , output_height, output_width)
    
    # Pad the input with zeros.
    # args:
    #     A: input to be padded
    #     padding: tuple of padding for height and width
    #     pad_value: value to pad with
    # returns:
    #     A_padded: padded input
    def pad(self, A, padding, pad_value=0):
        A_padded = np.pad(A, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode="constant", constant_values=(pad_value, pad_value))
        return A_padded
    
    # this function will convolve a slice of the input with the kernel.
    # at first, we will do element-wise multiplication between the slice of input and the kernel!
    # then we will do summation over all entries of the volume s, at last, we will Add bias to the result of the dot product.
    # arguments ->
    #       a_slic_prev: slice of the input data
    #       w_kernel: kernel
    #       bias: bias
    # returns:  convolved value
    def single_step_convolve(self, a_slic_prev, w_kernel, bias):
        
        summation_of_dot_product = np.sum(np.multiply(a_slic_prev, w_kernel) ) #dot product
        convolved_product = float(summation_of_dot_product + bias)
        return convolved_product

    # this function will help us to implement forwarding and pass for convolutional layer.
    # arguments ->
    #         A_prev: activations from previous layer (or input data)
    #         A_prev.shape = (batch_size, H_prev, W_prev, C_prev)
    #     returns:
    #         A: output of the convolutional layer
    def forward(self, A_prev):

        # Get kernel shape and bias
        kernel_weights , bias = self.parameters
        (batch_size , kernel_channels , kernel_height, kernel_width) = kernel_weights.shape
        input_shape =  kernel_weights.shape
        
        # Get input shape
        (batch_size,C_prev, H_prev, W_prev) = A_prev.shape
    
       
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding

        _ , chanel_count , output_height, output_width = self.target_shape(input_shape)
        
        # Initialize output with zeros
        output = np.zeros((batch_size, output_height , output_width , chanel_count))
    
        # Pad input tensor
        A_prev_pad = self.pad(A_prev, (padding_h, padding_w))
    
        # processing over the batch
        for i in range(batch_size):
            # Loop over vertical axis of output volume
            for h in range(output_height):
                h_start = h * stride_h
                h_end = h_start + kernel_height
                # Loop over horizontal axis of output volume
                for w in range(output_width):
                    w_start = w * stride_w
                    w_end = w_start + kernel_width
                    # Loop over channels of output volume
                    for c in range(kernel_channels):
                        # Extract slice of input volume
                        a_slice_prev = A_prev_pad[i, h_start:h_end, w_start:w_end, :]
                        # Convolve slice of input volume with kernel and add bias
                        output[i, h, w, c] = self.single_step_convolve(a_slice_prev, output_width[..., c], bias[..., c])
    
        return output
    
    # Backward pass for convolutional layer.
    # arguments:
    #     dZ: gradient of the cost with respect to the output of the convolutional layer
    #     A_prev: activations from previous layer (or input data)
    #     A_prev.shape = (batch_size, H_prev, W_prev, C_prev)
    # returns:
    #     dA_prev: gradient of the cost with respect to the input of the convolutional layer
    #     gradients: list of gradients with respect to the weights and bias
    def backward(self, dZ, A_prev):
        
        # Get kernel weights and bias
        W, b = self.parameters
        # Get shapes of input, kernel, and output tensors
        (batch_size, previous_height, previous_width, kernel_channels) = A_prev.shape
        
        #get the shape of kernel
        (batch_size  ,kernel_height, kernel_width, kernel_channels) = W.shape

        stride_height, stride_width = self.stride
        padding_height, padding_width = self.padding

        H, W = dZ.shape[1:3]
        # Initialize dA_prev, dW, and db to zeros
        dA_prev = np.zeros((batch_size, previous_height, previous_width, kernel_channels))
        dW = np.zeros((kernel_height, kernel_width, kernel_channels, batch_size))
        db = np.zeros((1, 1, 1, batch_size))
        # Pad input tensor
        A_prev_pad = self.pad(A_prev, (padding_height, padding_width))
        dA_prev_pad = self.pad(dA_prev, (padding_height, padding_width))
        # Loop over batch
        for i in range(batch_size):
        # Loop over vertical axis of output volume
            for h in range(H):
                h_start = h * stride_height
                h_end = h_start + kernel_height
                # Loop over horizontal axis of output volume
                for w in range(W):
                    w_start = w * stride_width
                    w_end = w_start + kernel_width
                    # Loop over channels of output volume
                    for c in range(batch_size):
                        # Extract slice of input volume
                        a_slice = A_prev_pad[i, h_start:h_end, w_start:w_end, :]
                        # Update gradients
                        dA_prev_pad[i, h_start:h_end, w_start:w_end, :] += np.multiply(W[..., c], dZ[i, h, w, c])
                        dW[..., c] += np.multiply(a_slice, dZ[i, h, w, c])
                        db[..., c] += dZ[i, h, w, c]

        # Remove padding from dA_prev_pad
        dA_prev = dA_prev_pad[:, padding_height:-padding_height, padding_width:-padding_width, :]
        # Return gradients
        grads = [dW, db]
        return dA_prev, grads
    
  
    # this function will Update parameters of the convolutional layer.
    # arguments: 
    #     optimizer: optimizer to use for updating parameters
    #     grads: list of gradients with respect to the weights and bias
    def update_parameters(self, optimizer, grads):
        self.parameters = optimizer.update(grads, self.name)