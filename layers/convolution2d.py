
import numpy as np

# The Conv2D class is an implementation of a 2D convolutional layer for use in a convolutional neural network. 
# This layer is responsible for extracting features from the input data by sliding a kernel over the input and 
# computing the dot product between the kernel and the input at each image position.
class Conv2D:
    #properties
    
    # in_channels -> the number of input data channels
    # out_channels -> the number of output data channels
    # name -> a string identifier for the layer
    # kernel_size ->  the size of the kernel for height and width
    # stride -> the stride(steps) for the kernel when sliding over the input, its for height and width sliding.
    # padding: the amount of padding to add to the input
    # initialize_method: a string indicating the method to use for initializing the weights of the kernel
    #parameters -> layer kernel weights and bias
    
    def __init__(self, in_channels, out_channels, name, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), initialize_method="random"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.name = name
        self.initialize_method = initialize_method

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.parameters = [self.initialize_weights(), self.initialize_bias()]


    #  Initializing kernel weights.
    # returns: weights: initialized kernel with shape: (kernel_size[0], kernel_size[1], in_channels, out_channels)
    def initialize_weights(self):
       
        if self.initialize_method == "random":  return np.random.randn(self.input_size, self.output_size)

        #This method initializes the weights with random numbers drawn from a uniform distribution with a specific range.
        elif self.initialize_method == "xavier":
            #uniform range
            range_value = np.sqrt( 1 / (self.input_size + self.output_size))
            return np.random.uniform(-range_value, range_value, (self.input_size, self.output_size))
         

        elif self.initialize_method == "he":
            #using gaussian distribution
            return np.random.randn((self.input_size, self.output_size)) *  sqrt(2 / self.input_size)

        else:
            raise ValueError("Invalid initialization method")
    
    
    #Initialize bias with zeros and layer out put count which will show the number of our perceptrons.      
    def initialize_bias(self):
        return np.zeros((self.output_size, 1))
    

    #this function will calculate the shape of the convolutional layer output.
    #argument -> input_shape: shape of the input of the convolutional layer{
    # 
    # batch_size -> the number of examples in a batch.
    # chanel_count is the number of channels in each example, which is typically 1 for grayscale images or 3 for RGB images! ...
    # height is the height of the input image in pixels.
    # width is the width of the input image in pixels.
    # }
    #returns -> target_shape: shape of the output of the convolutional layer -> batch_size , output Chanel's count and dimensions
    def target_shape(self, input_shape):
        
        #extracting input and kernel and stride and padding for output shape calculation
        batch_size, _ , height, width = input_shape
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        padding_height, padding_width = self.padding
        
        # Calculate the height and width of the output
        output_height = (height + 2*padding_height - kernel_height)//stride_height + 1
        output_width = (width + 2*padding_width - kernel_width)//stride_width + 1
        chanel_count = self.out_channels

        return (batch_size, chanel_count , output_height, output_width)
    
    def pad(self, A, padding, pad_value=0):
        """
        Pad the input with zeros.
        args:
            A: input to be padded
            padding: tuple of padding for height and width
            pad_value: value to pad with
        returns:
            A_padded: padded input
        """
        A_padded = np.pad(A, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode="constant", constant_values=(pad_value, pad_value))
        return A_padded
    
    def single_step_convolve(self, a_slic_prev, W, b):
        """
        Convolve a slice of the input with the kernel.
        args:
            a_slic_prev: slice of the input data
            W: kernel
            b: bias
        returns:
            Z: convolved value
        """
        # TODO: Implement single step convolution
        Z = None    # hint: element-wise multiplication
        Z = None    # hint: sum over all elements
        Z = None    # hint: add bias as type float using np.float(None)
        return Z

    def forward(self, A_prev):
        """
        Forward pass for convolutional layer.
            args:
                A_prev: activations from previous layer (or input data)
                A_prev.shape = (batch_size, H_prev, W_prev, C_prev)
            returns:
                A: output of the convolutional layer
        """
        # TODO: Implement forward pass
        W, b = None
        (batch_size, H_prev, W_prev, C_prev) = None
        (kernel_size_h, kernel_size_w, C_prev, C) = None
        stride_h, stride_w = None
        padding_h, padding_w = None
        H, W = None
        Z = None
        A_prev_pad = None # hint: use self.pad()
        for i in range(None):
            for h in range(None):
                h_start = None
                h_end = h_start + None
                for w in range(None):
                    w_start = None
                    w_end = w_start + None
                    for c in range(None):
                        a_slice_prev = A_prev_pad[i, h_start:h_end, w_start:w_end, :]
                        Z[i, h, w, c] = None # hint: use self.single_step_convolve()
        return Z

    def backward(self, dZ, A_prev):
        """
        Backward pass for convolutional layer.
        args:
            dZ: gradient of the cost with respect to the output of the convolutional layer
            A_prev: activations from previous layer (or input data)
            A_prev.shape = (batch_size, H_prev, W_prev, C_prev)
        returns:
            dA_prev: gradient of the cost with respect to the input of the convolutional layer
            gradients: list of gradients with respect to the weights and bias
        """
        # TODO: Implement backward pass
        W, b = None
        (batch_size, H_prev, W_prev, C_prev) = None
        (kernel_size_h, kernel_size_w, C_prev, C) = None
        stride_h, stride_w = None
        padding_h, padding_w = None
        H, W = None
        dA_prev = None  # hint: same shape as A_prev
        dW = None    # hint: same shape as W
        db = None    # hint: same shape as b
        A_prev_pad = None # hint: use self.pad()
        dA_prev_pad = None # hint: use self.pad()
        for i in range(None):
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            for h in range(None):
                for w in range(None):
                    for c in range(None):
                        h_start = None
                        h_end = h_start + None
                        w_start = None
                        w_end = w_start + None
                        a_slice = a_prev_pad[h_start:h_end, w_start:w_end, :]
                        da_prev_pad += None # hint: use element-wise multiplication of dZ and W
                        dW[..., c] += None # hint: use element-wise multiplication of dZ and a_slice
                        db[..., c] += None # hint: use dZ
            dA_prev[i, :, :, :] = None # hint: remove padding (trick: pad:-pad)
        grads = [dW, db]
        return dA_prev, grads
    
    def update_parameters(self, optimizer, grads):
        """
        Update parameters of the convolutional layer.
        args:
            optimizer: optimizer to use for updating parameters
            grads: list of gradients with respect to the weights and bias
        """
        self.parameters = optimizer.update(grads, self.name)