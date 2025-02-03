# SINDy autoencoder 
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# The SINDy library will be composed of the following functions: $1$, $\mathbf{z}$, $\mathbf{z}^2$, $\mathbf{z}^3$, $\sin(\mathbf{z})$, $\cos(\mathbf{z})$, $\dot{\mathbf{z}}$, $\dot{\mathbf{z}}^2$, $\dot{\mathbf{z}}^3$, and $\mathbf{z}\dot{\mathbf{z}}$. 

class SindyAutoEncoder(nn.Module):
    def __init__(self, layer_sizes=[2601,1200,600,200,80,10,1]):
        super(SindyAutoEncoder,self).__init__()

        # Layers for the encoder
        self.encoder_layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.encoder_layers.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))

        # Layers for the decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1,0,-1):
            self.decoder_layers.append(nn.Linear(layer_sizes[i],layer_sizes[i-1]))

        # Layer that computes the SINDY approximation of the second derivative.
        # The weights are the sindy coefficients for the 10 functions in the library.
        # (NOTE: Should the bias in nn.Linear count as the constant term?)
        self.sindy = nn.Linear(10,layer_sizes[-1]) 

        # Sequential thresholding mask 
        self.mask = torch.ones_like(self.sindy.weight).to(device)
        
        self.initialize_weights()

    # Initialize weights
    def initialize_weights(self):
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Linear):
                if m==self.sindy:
                    nn.init.constant_(m.weight, 1.0)
                else:
                    #nn.init.xavier_normal_(m.weight) #sigmoid
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='sigmoid')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    # Encode with linear layers and sigmoid activation functions
    #except the last one
    def encode(self, x):
        for layer in self.encoder_layers[:-1]:
            x = torch.sigmoid(layer(x))
        x = self.encoder_layers[-1](x)
        return x

    # Decode with linear layers and sigmoid activation functions
    #including the last one
    def decode(self, x):
        for layer in self.decoder_layers:
            x = torch.sigmoid(layer(x))
        return x

    # Forward pass through the network
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    # Compute the approximation to the second derivative of the latent space (z) 
    # (2nd order dif. eq.), as a function of z and zdot. Assumes the latent dimension = 1
    def sindy_library(self, z, zdot):
        theta_ddot_lib = []
        for i in range(z.shape[0]):
            ddot = torch.stack([torch.ones(z[i].shape).to(device),
                                (z[i]).to(device),
                                (z[i]*z[i]).to(device),
                                (z[i]*z[i]*z[i]).to(device),
                                (torch.sin(z[i])).to(device),
                                (torch.cos(z[i])).to(device),
                                (zdot[i]).to(device),
                                (zdot[i]*zdot[i]).to(device),
                                (zdot[i]*zdot[i]*zdot[i]).to(device),
                                (z[i]*zdot[i]).to(device)
                               ])
            theta_ddot_lib.append(torch.flatten(ddot))
        return torch.stack(theta_ddot_lib)

    # Compute the time derivatives for the encoder
    def tderiv_encoder(self, x, xdot, xddot):
        # Obtain the weights (We) of the encoder layers
        weights = []
        for layer in self.encoder_layers:
            weights.append(layer.weight)

        # Obtain the ai's
        a = [torch.sigmoid(self.encoder_layers[0](x))]
        for i in range(1,len(self.encoder_layers)-1):
            a.append(torch.sigmoid(self.encoder_layers[i](a[-1])))
        a.append(self.encoder_layers[-1](a[-1]))
        
        # Obtain the time derivative of the ai's
        adot = [a[0]*(1-a[0])*torch.mm(weights[0],xdot.t()).t()]
        for i in range(1,len(self.encoder_layers)-1):
            adot.append(a[i]*(1-a[i])*torch.mm(weights[i],adot[-1].t()).t())
        adot.append(torch.mm(weights[-1],adot[-1].t()).t())
        
        # Obtain the 2nd time derivative of the ai's
        addot = [(2*a[0]**3-3*a[0]**2+a[0])*(torch.mm(weights[0],xdot.t()).t())**2+a[0]*(1-a[0])*torch.mm(weights[0],xddot.t()).t()]
        for i in range(1,len(self.encoder_layers)-1):
            addot.append((2*a[i]**3-3*a[i]**2+a[i])*(torch.mm(weights[i],adot[i-1].t()).t())**2+a[i]*(1-a[i])*torch.mm(weights[i],addot[-1].t()).t())
        addot.append(torch.mm(weights[-1],addot[-1].t()).t())
        
        z = a[-1]
        zdot = adot[-1]
        zddot = addot[-1]
        return z, zdot, zddot

    # Compute the time derivatives for the decoder
    def tderiv_decoder(self, z, zdot, zddot):
        # Obtain the weights (Wd) of the decoder layers
        weights = []
        for layer in self.decoder_layers:
            weights.append(layer.weight)

        # Obtain the ai's
        a = [torch.sigmoid(self.decoder_layers[0](z))]
        for i in range(1,len(self.decoder_layers)):
            a.append(torch.sigmoid(self.decoder_layers[i](a[-1])))
        
        # Obtain the time derivative of the ai's
        adot = [a[0]*(1-a[0])*torch.mm(weights[0],zdot.t()).t()]
        for i in range(1,len(self.decoder_layers)):
            adot.append(a[i]*(1-a[i])*torch.mm(weights[i],adot[-1].t()).t())
        
        # Obtain the 2nd time derivative of the ai's
        addot = [(2*a[0]**3-3*a[0]**2+a[0])*(torch.mm(weights[0],zdot.t()).t())**2+a[0]*(1-a[0])*torch.mm(weights[0],zddot.t()).t()]
        for i in range(1,len(self.decoder_layers)):
            addot.append((2*a[i]**3-3*a[i]**2+a[i])*(torch.mm(weights[i],adot[i-1].t()).t())**2+a[i]*(1-a[i])*torch.mm(weights[i],addot[-1].t()).t())

        x = a[-1]
        xdot = adot[-1]
        xddot = addot[-1]
        return x, xdot, xddot