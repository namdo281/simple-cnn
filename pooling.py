import numpy as np
class MaxPooling:
    def __init__(self, input_size, filter_size = (2,2), stride=2):
        self.pooling_map = np.zeros(input_size)
        self.input_size = input_size
        self.filter_size = filter_size
        self.stride = stride
        self.target_height = (self.input_size[1] - self.filter_size[0]) // self.stride+1
        self.target_width = (self.input_size[2] - self.filter_size[1]) // self.stride+1

    
    def initializeY(self):
        return np.zeros((self.input_size[0], self.target_height, self.target_width))

    def maxPool(self, c): 
        y_c = np.zeros((self.target_height, self.target_width))
        mask = np.zeros(c.shape)
        for i in range(0, c.shape[0], self.stride):
            for j in range(0, c.shape[1], self.stride): 
                region = c[i:i+self.filter_size[0], j:j+self.filter_size[1]]
                nums = region.reshape(-1)
                _max = max(nums)
                y_c[(i-1)//self.stride, (j-1)//self.stride] = _max
                _argmax = (i + np.argmax(nums) // self.filter_size[1], j + np.argmax(nums) % self.filter_size[0])
                mask[_argmax[0], _argmax[1]] = 1 
        return y_c, mask
    def forward(self, x):
        y = self.initializeY()
        for i, c in enumerate(x):
            y[i], self.pooling_map[i] = self.maxPool(c)
        return y 

    def backward(self, dy):
        dx = self.pooling_map
        for i, dy_c  in enumerate(dy):
            for j in range(dy_c.shape[0]):
                for k in range(dy_c.shape[1]):
                    dx[
                        j*self.stride: j*self.stride+self.filter_size[0],
                        k*self.stride: k*self.stride+self.filter_size[1] 
                    ] *= dy_c[j, k]
        return dx