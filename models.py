import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(torch.nn.Module):
    def __init__(self, hidden_dims=[], input_dim=28*28, output_dim=10):
        super(SimpleMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, h_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = h_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)
    
class SimpleCNN(torch.nn.Module):
    def __init__(self, input_channels=1, hidden_channels=[32, 64], output_dim=10):
        super(SimpleCNN, self).__init__()
        conv_part = []
        in_channels = input_channels
        for i in range(len(hidden_channels)):
            conv_part.append(torch.nn.Conv2d(in_channels, hidden_channels[i], kernel_size=3, padding=1))
            conv_part.append(torch.nn.ReLU())
            in_channels = hidden_channels[i]
        self.last_channels = hidden_channels[-1]
        self.conv_part = torch.nn.Sequential(*conv_part)
        self.pool = torch.nn.MaxPool2d(2, 2)
        mlp_part = []
        mlp_part.append(torch.nn.Linear(self.last_channels * 14 * 14, 128))
        mlp_part.append(torch.nn.ReLU())
        mlp_part.append(torch.nn.Dropout(0.25))
        mlp_part.append(torch.nn.Linear(128, output_dim))
        self.mlp_part = torch.nn.Sequential(*mlp_part)
    def forward(self, x):
        x = self.conv_part(x) # shape: (batch_size, last_channels, 28, 28)
        x = self.pool(x)      # shape: (batch_size, last_channels, 14, 14)
        x = x.view(-1, self.last_channels * 14 * 14) # flatten
        x = self.mlp_part(x)
        return x

class CVAE_MLP(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), hidden_dims=[400],
                 latent_dim=20, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.input_dim = torch.prod(torch.tensor(input_shape)).item()
        
        # Encoder: image (784) + one hot label (10)
        encoder = []
        in_dim = self.input_dim + num_classes
        for h in hidden_dims:
            encoder.append(nn.Linear(in_dim, h))
            encoder.append(nn.ReLU())
            in_dim = h
        self.encoder = nn.Sequential(*encoder)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # Decoder: latent z + label (one hot)
        decoder = []
        in_dim = latent_dim + num_classes
        for h in reversed(hidden_dims):
            decoder.append(nn.Linear(in_dim, h))
            decoder.append(nn.ReLU())
            in_dim = h
        self.decoder = nn.Sequential(*decoder)
        self.fc_out = nn.Linear(in_dim, self.input_dim)

    def encode(self, x, y_onehot):
        x = x.view(-1, self.input_dim)
        inp = torch.cat([x, y_onehot], dim=1)
        h = F.relu(self.encoder(inp))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y_onehot):
        inp = torch.cat([z, y_onehot], dim=1)
        h = F.relu(self.decoder(inp))
        x_hat = torch.sigmoid(self.fc_out(h))
        return x_hat.view(-1, *self.input_shape)

    def forward(self, x, y_onehot):
        mu, logvar = self.encode(x, y_onehot)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y_onehot)
        return x_hat, mu, logvar

class CVAE_CNN(nn.Module):
    def __init__(
        self,
        input_channels=1,
        hidden_channels=[32, 64],
        latent_dim=16,
        num_classes=10,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels

        # -------------------------------------------------
        #  ENCODER: Conv + pooling + class conditioning
        # -------------------------------------------------

        # We inject class information by repeating a one-hot map
        # of shape (B, num_classes, 28, 28)
        in_channels = input_channels + num_classes
        conv_blocks = []

        for out_channels in hidden_channels:
            conv_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_blocks.append(nn.ReLU())
            in_channels = out_channels

        self.encoder_conv = nn.Sequential(*conv_blocks)
        self.pool = nn.MaxPool2d(2, 2)  # downsample to 14×14

        # After pooling, size = last_channel × 14 × 14
        flat_dim = hidden_channels[-1] * 14 * 14

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # -------------------------------------------------
        #  DECODER: MLP → ConvTranspose pipeline + conditioning
        # -------------------------------------------------

        self.fc_decode = nn.Linear(latent_dim + num_classes, flat_dim)

        decoder_channels = list(reversed(hidden_channels))

        deconv_blocks = []
        in_channels = decoder_channels[0]

        for i, out_channels in enumerate(decoder_channels[1:] + [input_channels]):

            # --- KEY POINT ---
            # Only the *last* hidden layer should perform upsampling:
            # 14x14 → 28x28
            if i == 0:
                # 1st deconv: keep size (14→14)
                deconv_blocks.append(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )

            elif i == len(decoder_channels) - 1:
                # FINAL deconv: upsample 14x14 → 28x28
                deconv_blocks.append(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,          # clean ×2 upsampling
                    )
                )

            else:
                # Optional intermediate blocks (kept at 14×14)
                deconv_blocks.append(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )

            # add activation unless final output
            if out_channels != input_channels:
                deconv_blocks.append(nn.ReLU())

            in_channels = out_channels

        self.decoder_conv = nn.Sequential(*deconv_blocks)


    # -----------------------------------------------------
    #  Utility: convert label y → a one-hot feature map
    # -----------------------------------------------------

    def make_label_map(self, y, H, W):
        """
        Convert labels (B,) into a spatial tensor (B, num_classes, H, W)
        Each pixel has the same one-hot vector.
        """
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        return y_onehot[:, :, None, None].expand(-1, -1, H, W)

    # -----------------------------------------------------
    #  ENCODER
    # -----------------------------------------------------

    def encode(self, x, y):
        B, _, H, W = x.shape

        y_map = self.make_label_map(y, H, W)  # (B, C_class, 28, 28)
        inp = torch.cat([x, y_map], dim=1)

        h = self.encoder_conv(inp)
        h = self.pool(h)  # (B, C_last, 14, 14)

        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # -----------------------------------------------------
    #  DECODER
    # -----------------------------------------------------

    def decode(self, z, y):
        B = z.size(0)

        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        latent_input = torch.cat([z, y_onehot], dim=1)

        h = self.fc_decode(latent_input)
        h = h.view(B, self.hidden_channels[-1], 14, 14)

        x_hat = self.decoder_conv(h)
        x_hat = torch.sigmoid(x_hat)
        return x_hat

    # -----------------------------------------------------
    #  FORWARD
    # -----------------------------------------------------

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y)
        return x_hat, mu, logvar
