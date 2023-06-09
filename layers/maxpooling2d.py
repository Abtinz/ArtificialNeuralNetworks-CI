import numpy as np

# this class will help us to implement our MaxPool Layer for pooling operation ...
# properties ->
#       kernel_size: size of the kernel
#       stride: stride of the kernel
#       mode: max or average

class MaxPool2D:

    def __init__(self, kernel_size=(3, 3), stride=(1, 1), mode="max"):
      
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.mode = mode
    
    # this function will calculate the shape of the pooling layer output.
    # argument -> input_shape: shape of the input of the convolutional layer{ 
    #       batch_size -> the number of examples in a batch.
    #       chanel_count is the number of channels in each example, which is typically 1 for grayscale images or 3 for RGB images! ...
    #       height is the height of the input image in pixels.
    #       width is the width of the input image in pixels.
    # returns -> target_shape: shape of the output of the pooling layer -> batch_size , output Chanel's count and dimensions
    def target_shape(self, input_shape):
        # Get input shape
        (batch_size, prev_height, prev_width, C_prev) = input_shape
        # Get kernel and stride sizes
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        # Compute output shape
        output_height = int(1 + (prev_height - kernel_height) / stride_height)
        output_width = int(1 + (prev_width - kernel_width) / stride_width)
        return (batch_size, output_height, output_width, C_prev)
    
    # Forward pass for max pooling layer.
    # argument: A_prev: activations from previous layer (or input data)
    #  returns: output of the max pooling layer

    def forward(self, A_prev):
       
        # Get input shape
        (batch_size, H_prev, W_prev, C_prev) = A_prev.shape

        (batch_size, output_height, output_width, C_prev)  = self.target_shape(A_prev.shape)

        # Get kernel and stride sizes
        (kernel_height, kernel_width) = self.kernel_size
        stride_height, stride_width = self.stride
        
        # Initialize output tensor
        output = np.zeros((batch_size, output_height, output_width, C_prev))

        # Loop over batch
        for i in range(batch_size):
            # Loop over vertical axis of output volume
            for h in range(output_height):
                h_start = h * stride_height
                h_end = h_start + kernel_height
                # Loop over horizontal axis of output volume
                for w in range(output_width):
                    w_start = w * stride_width
                    w_end = w_start + kernel_width
                    # Loop over channels of output volume
                    for c in range(C_prev):
                        # Extract slice of input volume
                        a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, c]
                        if self.mode == "max":
                            # Compute max value of slice
                            output[i, h, w, c] = np.max(a_prev_slice)
                        elif self.mode == "average":
                            # Compute average value of slice
                            output[i, h, w, c] = np.mean(a_prev_slice)
                        else:
                            raise ValueError("Invalid mode\options: max, average")

        return output
    
   
    #  Create a mask from an input matrix x, to identify the max entry of x.
    #  args:      x: numpy array
    #  returns:   mask: numpy array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    def create_mask_from_window(self, x):

        mask = x == np.max(x) # Create a binary mask with True where the max value is located
        return mask
    
    # Distribute the input value in the matrix of dimension shape.
    # args:
    #      dz: input scalar
    #      shape: the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    def distribute_value(self, dz, shape):
        # Get shape of output matrix
        (n_H, n_W) = shape
        # Compute average value to be distributed
        average = dz / (n_H * n_W)
        # Create a matrix of the same shape as the output matrix, with each element equal to the average value
        averageMatrix = np.ones(shape) * average
        return averageMatrix
    
    