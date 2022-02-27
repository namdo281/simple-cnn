import numpy as np
class Convolution:
    def __init__(self, input_size, n_filter, filter_size = (3,3), stride = 1, padding = 'zero'):
        self.W = np.ones((n_filter, input_size[0],filter_size[0], filter_size[1]))/(filter_size[1]*filter_size[0])
        self.b = np.ones((n_filter))/(filter_size[1]*filter_size[0])
        self.input_size = input_size
        self.filter_size = filter_size
        self.n_filter = n_filter
        self.stride = stride 
        self.padding = padding
        self.f_half_height = self.filter_size[0]//2
        self.f_half_width = self.filter_size[1]//2

    #convolution on gray image with 1 filter channel
    def convolChannel(self, x_c, f_c, target_shape):
        #print("ran convolChannel")
        #print(f_c.shape)
        target = np.zeros(target_shape)
        #print("target shape:", target.shape)
        for a1 in range(0,target_shape[0],self.stride):
            #print(1)
            for a2 in range(0, target_shape[1], self.stride):
                target[a1, a2] =  sum(
                                    np.multiply(
                                        x_c[
                                            a1:a1+f_c.shape[0], 
                                            a2:a2+f_c.shape[1]
                                        ], 
                                        f_c
                                    ).reshape(-1)    
                                )
        #print("abcxyz")
        #print(target)
        return target
    def convolChannelBackward(self, x_c, f_c, target_shape):
        #print("ran convolChannel")
        #print(f_c.shape)
        target = np.zeros(target_shape)
        #print("target shape:", target.shape)
        for a1 in range(0,target_shape[0],self.stride):
            #print(1)
            for a2 in range(0, target_shape[1], self.stride):
                target[a1, a2] =  sum(
                                    np.multiply(
                                        x_c[
                                            a1:a1+f_c.shape[0], 
                                            a2:a2+f_c.shape[1]
                                        ], 
                                        f_c
                                    ).reshape(-1)    
                                )
        #print("abcxyz")
        #print(target)
        return target
    #convolution on multi channel image
    def convol(self, x, f, target_shape):
        assert x.shape[0] == f.shape[0]
        
        y = np.zeros(target_shape)
        for i in range(x.shape[0]):
            #print(x)
            #print(f)
            #print(target_shape)
            target = self.convolChannel(x[i], f[i], target_shape)
            #print(target)
            y += target
        #print(y)
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
            y = np.zeros((
                self.n_filter,
                (self.input_size[1]+2-self.filter_size[0])//self.stride+1, \
                (self.input_size[2]+2-self.filter_size[1])//self.stride+1 \
            ))            
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
        #print("conv")
        for i, c in enumerate(x):
            x[i] = c/np.linalg.norm(c)
        #print(x.shape)
        try: 
            self.x = self.pad(x, self.f_half_height, self.f_half_width)
            y = self.initializeY() 
            #print("output_shape: ", y.shape)
            #print("forward")
            for k, f in enumerate(self.W):
                y[k] = self.convol(self.x, f, y[k].shape)+self.b[k]
            self.y = y
            #print(y)
            return y
        except:
            print("convol x: ", x)
    
    def backward(self, dy):
        #print("backward")
        W = self.W
        x = self.x
        b = self.b
        dW = np.zeros(W.shape)
        #print(x.shape)
        dx = np.zeros((x.shape[0], x.shape[1]-2*self.f_half_height, x.shape[2]-2*self.f_half_width))
        #print("abc")
        dy_pad = self.pad(dy, self.f_half_height, self.f_half_width)
        #print(x.shape)
        for i, k in enumerate(W):
            for j, c in enumerate(k):
                df_c = self.convolChannelBackward(x_c = x[j], f_c = dy[i], target_shape=c.shape)
                dW[i,j] += df_c
                
                flipped_c = self.flip(c)
                flipped_dc = self.convolChannelBackward(dy_pad[i], flipped_c, (x[j].shape[0]-2*self.f_half_height, x[j].shape[1]-2*self.f_half_width))
                dx_c = self.flip(flipped_dc)
                #print(dx_c.shape)
                #print(dx.shape)
                dx[j] += dx_c

        db = sum(dy.reshape(-1))
        self.W = W - 1e-3*dW
        self.b = b - 1e-3*db
        return dx