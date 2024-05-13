import torch
import torch.nn as nn

from collections import namedtuple

USE_GPU = torch.cuda.is_available()

Modality_input = namedtuple("Modality_input", ["agentview", "eye_in_hand", "force", "proprio", "frontview"])
Modality_output = namedtuple("Modality_output", ["agentview_recon", "eye_in_hand_recon", "contact", 'proprio', 'frontview_recon'])

def safe_cuda(x):
    if USE_GPU:
        return x.cuda()
    return x

def product_of_experts(m_vect, v_vect):
    T_vect = 1.0 / v_vect

    mu = (m_vect * T_vect).sum(2) * (1 / T_vect.sum(2))
    var = 1 / T_vect.sum(2)

    return mu, var


class CausalConv1d(nn.Conv1d):
    """
    Causal Convolution in 1d.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=True):
        self._padding = (kernel_size - 1) * dilation

        super().__init__(in_channels,
                         out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=self._padding,
                         dilation=dilation,
                         bias=bias)

    def forward(self, x):
        y = super().forward(x)
        if self.padding != 0:
            return y[:, :, :-self._padding]
        return y

def conv2d_leakyrelu(in_channels, out_channels, kernel_size, stride=1, dilation=1, alpha=0.1, bias=True):

    dilated_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    same_padding = (dilated_kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=same_padding,
                  dilation=dilation,
                  bias=bias),
        nn.LeakyReLU(alpha, inplace=True)
    )

def crop_like(inp, target):
    if inp.size()[2:] == target.size()[2:]:
        return inp
    else:
        return inp[:, :, :target.size(2), :target.size(3)]

def deconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False,
        ),
        nn.LeakyReLU(0.1, inplace=True))
    

class ForceEncoder(nn.Module):
    def __init__(self, force_dim, z_dim, alpha=0.1):
        super().__init__()
        self.force_dim = force_dim
        self.z_dim = z_dim

        self.layers = torch.nn.Sequential(CausalConv1d(self.force_dim, 32, kernel_size=2, stride=2),
                                          nn.LeakyReLU(alpha, inplace=True),
                                          # CausalConv1d(16, 32, kernel_size=2, stride=2),
                                          # nn.LeakyReLU(alpha, inplace=True),
                                          CausalConv1d(32, 64, kernel_size=2, stride=2),
                                          nn.LeakyReLU(alpha, inplace=True),
                                          CausalConv1d(64, 128, kernel_size=2, stride=2),
                                          nn.LeakyReLU(alpha, inplace=True),
                                          CausalConv1d(128, 2 * self.z_dim, kernel_size=2, stride=2),
                                          nn.LeakyReLU(alpha, inplace=True),
                                          )
            
    def forward(self, x):
        return torch.split(self.layers(x), self.z_dim, dim=1)

class ContactDecoder(nn.Module):
    def __init__(self, z_dim, alpha=0.1):
        super().__init__()
        self.z_dim = z_dim

        self.layers = torch.nn.Sequential(nn.Linear(z_dim, 256),
                                          nn.LeakyReLU(alpha, inplace=True),
                                          # nn.Linear(128, 128),
                                          # nn.LeakyReLU(alpha, inplace=True), 
                                          nn.Linear(256, 1),
                                          torch.nn.Sigmoid(),
                                          )

    def forward(self, x):
        return self.layers(x)

class EEDeltaEncoder(nn.Module):
    def __init__(self, z_dim, ee_dim, alpha=0.1):
        super().__init__()
        self.z_dim = z_dim
        self.ee_dim = ee_dim
        self.layers = torch.nn.Sequential(nn.Linear(ee_dim, 256),
                                          nn.LeakyReLU(alpha, inplace=True),
                                          nn.Linear(256, 256),
                                          nn.LeakyReLU(alpha, inplace=True), 
                                          nn.Linear(256, 2 * z_dim),
                                          )
        
    def forward(self, x):
        return torch.split(self.layers(x).unsqueeze(-1), self.z_dim, dim=1)
    
class EEDeltaDecoder(nn.Module):
    def __init__(self, z_dim, ee_dim, alpha=0.1):
        super().__init__()
        self.z_dim = z_dim
        self.ee_dim = ee_dim
        self.layers = torch.nn.Sequential(nn.Linear(z_dim, 128),
                                          nn.LeakyReLU(alpha, inplace=True),
                                          nn.Linear(128, 128),
                                          nn.LeakyReLU(alpha, inplace=True), 
                                          nn.Linear(128, ee_dim),
                                          )
        
    def forward(self, x):
        return self.layers(x)
        
# Image
class ImageEncoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()

        self.z_dim = z_dim

        self.conv1 = conv2d_leakyrelu(in_channels=3, out_channels=16, kernel_size=7, stride=2)
        self.conv2 = conv2d_leakyrelu(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.conv3 = conv2d_leakyrelu(in_channels=32, out_channels=64, kernel_size=5, stride=2)
        self.conv4 = conv2d_leakyrelu(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.conv5 = conv2d_leakyrelu(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv6 = conv2d_leakyrelu(in_channels=128, out_channels=self.z_dim, kernel_size=3, stride=2)

        self.embedding_layer = nn.Linear(4 * self.z_dim, 2 * self.z_dim)

        
        # (Yifeng) Might need to try kaiming initialization
        
    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)

        out_convs = (out_conv1,
                     out_conv2,
                     out_conv3,
                     out_conv4,
                     out_conv5,
                     out_conv6
                     )

        img_out = self.embedding_layer(torch.flatten(out_conv6, start_dim=1)).unsqueeze(2)
        return img_out, out_convs

class ImageDecoder(nn.Module):
    def __init__(self, z_dim, use_skip_connection=True):
        super().__init__()

        self.z_dim = z_dim
        self.embedding_conv = conv2d_leakyrelu(self.z_dim, 64, kernel_size=1, stride=1)

        self.use_skip_connection = use_skip_connection

        if self.use_skip_connection:
            self.deconv6 = deconv(64, 64)
            self.deconv5 = deconv(64, 32)
            self.deconv4 = deconv(96, 32)
            self.deconv3 = deconv(96, 32)
            self.deconv2 = deconv(64, 32)
            self.deconv1 = deconv(48, 3)
        else:
            self.deconv6 = deconv(64, 64)
            self.deconv5 = deconv(64, 32)
            self.deconv4 = deconv(32, 32)
            self.deconv3 = deconv(32, 32)
            self.deconv2 = deconv(32, 32)
            self.deconv1 = deconv(32, 3)

    def forward(self, x, out_convs):
        out_conv1, out_conv2, out_conv3, out_conv4, out_conv5, out_conv6 = out_convs

        out_deconv_embedding = self.embedding_conv(x)

        if self.use_skip_connection:
            out_deconv6 = self.deconv6(out_deconv_embedding)
            out_deconv5 = self.deconv5(out_deconv6)
            out_deconv4 = self.deconv4(torch.cat((out_deconv5, out_conv4), 1))
            out_deconv3 = self.deconv3(torch.cat((out_deconv4, out_conv3), 1))
            out_deconv2 = self.deconv2(torch.cat((out_deconv3, out_conv2), 1))
            out_deconv1 = self.deconv1(torch.cat((out_deconv2, out_conv1), 1))

        else:
            out_deconv6 = self.deconv6(out_deconv_embedding)
            out_deconv5 = self.deconv5(out_deconv6)
            out_deconv4 = self.deconv4(out_deconv5)
            out_deconv3 = self.deconv3(out_deconv4)
            out_deconv2 = self.deconv2(out_deconv3)
            out_deconv1 = self.deconv1(out_deconv2)


        # print(out_deconv6.shape,
        #       out_deconv5.shape,
        #       out_deconv4.shape,
        #       out_deconv3.shape,
        #       out_deconv2.shape)
        # print(out_conv6.shape,
        #       out_conv5.shape,
        #       out_conv4.shape,
        #       out_conv3.shape,
        #       out_conv2.shape)

        return torch.sigmoid(out_deconv1)
    

class ImageVAE(torch.nn.Module):
    def __init__(self, z_dim=128, use_skip_connection=True):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = ImageEncoder(z_dim=z_dim)
        self.decoder = ImageDecoder(z_dim=z_dim, use_skip_connection=use_skip_connection)

    def forward(self, x):
        h, convs = self.encoder(x)
        mu, logvar = torch.split(h, self.z_dim, dim=1)

        z = self.sampling(mu, logvar)
        out = self.decoder(z.view(z.size(0), self.z_dim, 1, 1).expand(-1, -1, 2, 2), convs)
        return out, mu, logvar

    def sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps * std + mu

class SensorFusion(torch.nn.Module):
    def __init__(self, z_dim=128, use_skip_connection=True, modalities=['frontview', 'eye_in_hand', 'force', 'proprio'], use_gmm=False, proprio_dim=5):
        super().__init__()

        self.use_gmm = use_gmm

        self.z_dim = z_dim

        self.modalities = modalities

        if "frontview" in modalities:
            self.encoder_frontview = ImageEncoder(z_dim=z_dim)
            self.decoder_frontview = ImageDecoder(z_dim=z_dim, use_skip_connection=use_skip_connection)

        self.encoder_agentview = ImageEncoder(z_dim=z_dim)
        self.decoder_agentview = ImageDecoder(z_dim=z_dim, use_skip_connection=use_skip_connection)
        
        self.encoder_eye_in_hand = ImageEncoder(z_dim=z_dim)
        self.decoder_eye_in_hand = ImageDecoder(z_dim=z_dim, use_skip_connection=use_skip_connection)

        self.encoder_force = ForceEncoder(force_dim=6, z_dim=z_dim)
        self.decoder_contact = ContactDecoder(z_dim=z_dim)

        self.encoder_proprio = EEDeltaEncoder(z_dim=z_dim, ee_dim=proprio_dim)
        self.decoder_proprio = EEDeltaDecoder(z_dim=z_dim, ee_dim=proprio_dim)

        self.z_prior_m = torch.nn.Parameter(
            torch.zeros(1, self.z_dim), requires_grad=False
        )
        self.z_prior_v = torch.nn.Parameter(
            torch.ones(1, self.z_dim), requires_grad=False
        )

    def forward(self, x, encoder_only=False):

        batch_dim = x[0].size(0)
        
        z_prior_m = (self.z_prior_m.expand(batch_dim, *self.z_prior_m.shape).reshape(-1, *self.z_prior_m.shape[1:])).unsqueeze(2)
        z_prior_v = (self.z_prior_v.expand(batch_dim, *self.z_prior_v.shape).reshape(-1, *self.z_prior_v.shape[1:])).unsqueeze(2)
        m_vect_list = [z_prior_m]
        v_vect_list = [z_prior_v]
        if 'agentview' in self.modalities:
            x_agentview = x.agentview
            h_agentview, convs_agentview = self.encoder_agentview(x_agentview)
            mu_agentview, var_h_agentview = torch.split(h_agentview, self.z_dim, dim=1)
            var_agentview = torch.nn.Softplus()(var_h_agentview) + 1e-6
            m_vect_list.append(mu_agentview)
            v_vect_list.append(var_agentview)

        if 'frontview' in self.modalities:
            x_frontview = x.frontview
            h_frontview, convs_frontview = self.encoder_frontview(x_frontview)
            mu_frontview, var_h_frontview = torch.split(h_frontview, self.z_dim, dim=1)
            var_frontview = torch.nn.Softplus()(var_h_frontview) + 1e-6
            m_vect_list.append(mu_frontview)
            v_vect_list.append(var_frontview)
            
        if 'eye_in_hand' in self.modalities:
            x_eye_in_hand = x.eye_in_hand
            h_eye_in_hand, convs_eye_in_hand = self.encoder_eye_in_hand(x_eye_in_hand)
            mu_eye_in_hand, var_h_eye_in_hand = torch.split(h_eye_in_hand, self.z_dim, dim=1)
            var_eye_in_hand = torch.nn.Softplus()(var_h_eye_in_hand) + 1e-6

        if 'force' in self.modalities:
            force = x.force
            mu_force, var_h_force = self.encoder_force(force)
            var_force = torch.nn.Softplus()(var_h_force) + 1e-6

            m_vect_list.append(mu_force)
            v_vect_list.append(var_force)

        if 'proprio' in self.modalities:
            proprio = x.proprio
            mu_proprio, var_h_proprio = self.encoder_proprio(proprio)
            var_proprio = torch.nn.Softplus()(var_h_proprio) + 1e-6

            m_vect_list.append(mu_proprio)
            v_vect_list.append(var_proprio)

        m_vect = torch.cat(m_vect_list, dim=2)
        v_vect = torch.cat(v_vect_list, dim=2)

        mu_z, var_z = product_of_experts(m_vect, v_vect)

        if encoder_only:
            return mu_z


        z = self.sampling(mu_z, var_z)
        # print(z.shape)
        # print(mu_force.shape)


        out_frontview = None
        out_agentview = None
        out_eye_in_hand = None
        out_contact = None
        out_proprio = None

        reshaped_z = z.view(batch_dim, self.z_dim, 1, 1).expand(-1, -1, 2, 2)
        if 'agentview' in self.modalities:
            out_agentview = self.decoder_agentview(reshaped_z, convs_agentview)
        
        if 'frontview' in self.modalities:
            out_frontview = self.decoder_frontview(reshaped_z, convs_frontview)

        if 'eye_in_hand' in self.modalities:
            out_eye_in_hand = self.decoder_eye_in_hand(reshaped_z, convs_eye_in_hand)

        if 'force' in self.modalities:
            out_contact = self.decoder_contact(z)

        if 'proprio' in self.modalities:
            out_proprio = self.decoder_proprio(z)
        

        output = Modality_output(frontview_recon=out_frontview,
                                 agentview_recon=out_agentview,
                                 eye_in_hand_recon=out_eye_in_hand,
                                 contact=out_contact,
                                 proprio=out_proprio)

        return output, mu_z, var_z, self.z_prior_m, self.z_prior_v

    # def forward_encoder(self, x_frontview, x_eye_in_hand):

    #     z_prior_m = (self.z_prior_m.expand(x_frontview.size(0), *self.z_prior_m.shape).reshape(-1, *self.z_prior_m.shape[1:])).unsqueeze(2)
    #     z_prior_v = (self.z_prior_v.expand(x_frontview.size(0), *self.z_prior_v.shape).reshape(-1, *self.z_prior_v.shape[1:])).unsqueeze(2)

    #     h_frontview, convs_frontview = self.encoder_frontview(x_frontview)
    #     h_eye_in_hand, convs_eye_in_hand = self.encoder_eye_in_hand(x_eye_in_hand)

    #     mu_frontview, var_h_frontview = torch.split(h_frontview, self.z_dim, dim=1)
    #     mu_eye_in_hand, var_h_eye_in_hand = torch.split(h_eye_in_hand, self.z_dim, dim=1)

    #     var_frontview = torch.nn.Softplus()(var_h_frontview)
    #     var_eye_in_hand = torch.nn.Softplus()(var_h_eye_in_hand)

    #     m_vect = torch.cat([mu_frontview, mu_eye_in_hand, z_prior_m], dim=2)
    #     v_vect = torch.cat([var_frontview, var_eye_in_hand, z_prior_v], dim=2)

    #     mu_z, var_z = product_of_experts(m_vect, v_vect)

    #     z = self.sampling(mu_z, var_z)
    #     return z

    def sampling(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(mu)
        return eps * std + mu

class GMM():
    def __init__(self, z_dim, num_components=10):
        self.num_components = num_components
        self.mean_list = []
        self.var_list = []
        for i in range(self.num_components):
            self.mean_list.append(i * 0.1)
            self.logvar_list.append(0.1)

        self.mu = safe_cuda(torch.stack(self.mean_list, dim=0))
        self.var = safe_cuda(torch.stack(self.var_list, dim=0))
    

class SensorFusionGMM(torch.nn.Module):
    def __init__(self, z_dim=128, use_skip_connection=True, modalities=['frontview', 'eye_in_hand', 'force', 'proprio'], num_components=10):
        super().__init__()

        self.use_gmm = use_gmm
        self.gmm = GMM(z_dim=z_dim,
                       num_components=num_components)

        self.z_dim = z_dim

        self.modalities = modalities

        if "frontview" in modalities:
            self.encoder_frontview = ImageEncoder(z_dim=z_dim)
            self.decoder_frontview = ImageDecoder(z_dim=z_dim, use_skip_connection=use_skip_connection)

        self.encoder_agentview = ImageEncoder(z_dim=z_dim)
        self.decoder_agentview = ImageDecoder(z_dim=z_dim, use_skip_connection=use_skip_connection)
        
        self.encoder_eye_in_hand = ImageEncoder(z_dim=z_dim)
        self.decoder_eye_in_hand = ImageDecoder(z_dim=z_dim, use_skip_connection=use_skip_connection)

        self.encoder_force = ForceEncoder(force_dim=6, z_dim=z_dim)
        self.decoder_contact = ContactDecoder(z_dim=z_dim)

        self.encoder_proprio = EEDeltaEncoder(z_dim=z_dim, ee_dim=6)
        self.decoder_proprio = EEDeltaDecoder(z_dim=z_dim, ee_dim=6)

        self.z_prior_m = torch.nn.Parameter(
            torch.zeros(1, self.z_dim), requires_grad=False
        )
        self.z_prior_v = torch.nn.Parameter(
            torch.ones(1, self.z_dim), requires_grad=False
        )

    def forward(self, x, encoder_only=False):

        batch_dim = x[0].size(0)
        
        z_prior_m = (self.z_prior_m.expand(batch_dim, *self.z_prior_m.shape).reshape(-1, *self.z_prior_m.shape[1:])).unsqueeze(2)
        z_prior_v = (self.z_prior_v.expand(batch_dim, *self.z_prior_v.shape).reshape(-1, *self.z_prior_v.shape[1:])).unsqueeze(2)
        m_vect_list = [z_prior_m]
        v_vect_list = [z_prior_v]
        if 'agentview' in self.modalities:
            x_agentview = x.agentview
            h_agentview, convs_agentview = self.encoder_agentview(x_agentview)
            mu_agentview, var_h_agentview = torch.split(h_agentview, self.z_dim, dim=1)
            var_agentview = torch.nn.Softplus()(var_h_agentview) + 1e-6
            m_vect_list.append(mu_agentview)
            v_vect_list.append(var_agentview)

        if 'frontview' in self.modalities:
            x_frontview = x.frontview
            h_frontview, convs_frontview = self.encoder_frontview(x_frontview)
            mu_frontview, var_h_frontview = torch.split(h_frontview, self.z_dim, dim=1)
            var_frontview = torch.nn.Softplus()(var_h_frontview) + 1e-6
            m_vect_list.append(mu_frontview)
            v_vect_list.append(var_frontview)
            
        if 'eye_in_hand' in self.modalities:
            x_eye_in_hand = x.eye_in_hand
            h_eye_in_hand, convs_eye_in_hand = self.encoder_eye_in_hand(x_eye_in_hand)
            mu_eye_in_hand, var_h_eye_in_hand = torch.split(h_eye_in_hand, self.z_dim, dim=1)
            var_eye_in_hand = torch.nn.Softplus()(var_h_eye_in_hand) + 1e-6

        if 'force' in self.modalities:
            force = x.force
            mu_force, var_h_force = self.encoder_force(force)
            var_force = torch.nn.Softplus()(var_h_force) + 1e-6

            m_vect_list.append(mu_force)
            v_vect_list.append(var_force)

        if 'proprio' in self.modalities:
            proprio = x.proprio
            mu_proprio, var_h_proprio = self.encoder_proprio(proprio)
            var_proprio = torch.nn.Softplus()(var_h_proprio) + 1e-6

            m_vect_list.append(mu_proprio)
            v_vect_list.append(var_proprio)

        m_vect = torch.cat(m_vect_list, dim=2)
        v_vect = torch.cat(v_vect_list, dim=2)

        mu_z, var_z = product_of_experts(m_vect, v_vect)

        if encoder_only:
            return mu_z


        z = self.sampling(mu_z, var_z)

        out_frontview = None
        out_agentview = None
        out_eye_in_hand = None
        out_contact = None
        out_proprio = None

        reshaped_z = z.view(batch_dim, self.z_dim, 1, 1).expand(-1, -1, 2, 2)
        if 'agentview' in self.modalities:
            out_agentview = self.decoder_agentview(reshaped_z, convs_agentview)
        
        if 'frontview' in self.modalities:
            out_frontview = self.decoder_frontview(reshaped_z, convs_frontview)

        if 'eye_in_hand' in self.modalities:
            out_eye_in_hand = self.decoder_eye_in_hand(reshaped_z, convs_eye_in_hand)

        if 'force' in self.modalities:
            out_contact = self.decoder_contact(z)

        if 'proprio' in self.modalities:
            out_proprio = self.decoder_proprio(z)
        

        output = Modality_output(frontview_recon=out_frontview,
                                 agentview_recon=out_agentview,
                                 eye_in_hand_recon=out_eye_in_hand,
                                 contact=out_contact,
                                 proprio=out_proprio)

        return output, z, mu_z, var_z, self.z_prior_m, self.z_prior_v

    def sampling(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(mu)
        return eps * std + mu
    
    
if __name__ == "__main__":
    force_encoder = ForceEncoder(force_dim=6, z_dim=128)

    inp = torch.randn(1, 6, 32)    
    y = force_encoder(inp)
    print(y.shape)

    img_encoder = ImageEncoder(z_dim=128)
    img_decoder = ImageDecoder(z_dim=128)
    inp = torch.randn(1, 3, 128, 128)
    img_out, img_out_convs = img_encoder(inp)

    embedding = torch.split(img_out, 128, dim=1)[0].view(img_out.size(0), 128, 1, 1).expand(-1, -1, 2, 2)
    reconstruct_inp = img_decoder(embedding, img_out_convs)
    print(reconstruct_inp.shape)

    # TODO Depth encoder + decoder

    # Contact
