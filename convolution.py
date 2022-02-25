import numpy as np
class Convolution:
    def __init__(self, input_size, n_filter, filter_size = (3,3), stride = 1, padding = 'zero'):
        self.W = np.zeros((n_filter, input_size[0],filter_size.shape[0], filter_size.shape[1]))
        self.b = np.zeros((n_filter))
        self.input_size = input_size
        self.filter_size = filter_size
        self.n_filter = n_filter
        self.stride = stride 
        self.padding = padding
        self.f_half_height = self.filter_size.shape[0]//2
        self.f_half_width = self.filter_size.shape[1]//2

    #convolution on gray image with 1 filter channel
    def convolChannel(self, x_c, f_c, target_shape):
        target = np.zeros(target_shape)
        for i in range(0,target_shape[0],self.stride):
            for j in range(0, target_shape[1], self.stride):                
                target[i, j] =  sum(
                                    np.multiply(
                                        x_c[
                                            i:i+self.filter_size.shape[0], 
                                            j:j+self.filter_size.shape[1]
                                        ],
                                        f_c
                                    ).reshape[-1]    
                                )
        return target
    #convolution on multi channel image
    def convol(self, x, f, target_shape):
        assert x.shape[0] == f.shape[0]
        y = np.zeros(target_shape)
        for i in range(x.shape[0]):
            target = self.convolChannel(x[i], f[i], target_shape)
            y += target
        return y

    def flip(self, x):
        edge = np.array([x.shape[0]-1, x.shape[1]-1])
        new_x = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                new_x[edge[0] - i, edge[1]-j] = x[i,j]
        return new_x

    def initializeY(self):
        if self.padding == "zero":
            y = np.zeros(
                self.n_filter,
                (self.input_size[1]+2-self.filter_size[0])/self.stride+1, \
                (self.input_size[2]+2-self.filter_size[1])/self.stride+1 \
            )            
        else:
            y = np.zeros((
                self.n_filter,
                (self.input_size[1]-self.filter_size[0])/self.stride+1, 
                (self.input_size[2]-self.filter_size[1])/self.stride+1 
            ))
        return y
    def pad(self, x, h_pad, w_pad):
        if self.padding == 'zero':
            x_bar = np.zeros((x.shape[0], x.shape[1]+2*h_pad, x.shape[2]+2*w_pad))
            x_bar[:,
                h_pad:-h_pad,
                w_pad:-w_pad
            ] = x
        else:
            x_bar = x
        return x_bar
    def forward(self, x):
        self.x = self.pad(x, self.f_half_height, self.f_half_width)
        y = self.initializeY(x) 
        for k, f in enumerate(self.W):
            y[k] = self.convol(self.x, f, y[k].shape)+self.b[k]
        self.y = y
        return y
    
    def backward(self, dy):
        W = self.W
        x = self.x
        b = self.b
        dW = np.zeros(W.shape)
        for i, k in enumerate(W):
            for j, c in enumerate(k):
                df_c = self.convolChannel(x[j], dy[i], c.shape)
                dW[k,c] += df_c

        dx = np.zeros(x.shape)

        dy_pad = self.pad(dy, self.f_half_height, self.f_half_width)
        for i, k in enumerate(W):
            for j, c in enumerate(k):
                flipped_k = self.flip(k)
                flipped_dc = self.convolChannel(dy_pad[i], flipped_k, x[j].shape)
                dx_c = self.flip(flipped_dc)
                dx[c] += dx_c
        db = sum(dy.reshape(-1))
        self.W = W - dW
        self.b = b - db
        return dx[self.f_half_height: -self.f_half_height, self.f_half_width: -self.f_half_width]