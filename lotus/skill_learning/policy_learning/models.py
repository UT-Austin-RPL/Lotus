import torch
import torch.nn as nn
import torchvision
import numpy as np
from models.model_utils import safe_cuda
from models.torch_utils import to_onehot
from models.resnet_model_utils import resnet18 as no_pool_resnet18
import torch.nn.functional as F
import torch.distributions as D
from enum import Enum

class PolicyType(Enum):
    NO_SUBGOAL=1
    NORMAL_SUBGOAL=2
    VAE_SUBGOAL=3

def get_activate_fn(activation):
    if activation == 'relu':
        activate_fn = torch.nn.ReLU
    elif activation == 'leaky-relu':
        activate_fn = torch.nn.LeakyReLU
    return activate_fn

class PerceptionEmbedding(torch.nn.Module):
    def __init__(self,
                 use_eye_in_hand=True,
                 pretrained=False,
                 no_training=False,
                 activation='relu',
                 remove_layer_num=2,
                 img_c=3):

        super().__init__()
        if use_eye_in_hand:
            print("Using Eye in Hand !!!!!")            
            img_c = img_c + 3

        # For training policy
        layers = list(torchvision.models.resnet18(pretrained=pretrained).children())[:-remove_layer_num]
        if img_c != 3:
            # If use eye_in_hand, we need to increase the channel size
            conv0 = torch.nn.Conv2d(img_c, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            layers[0] = conv0

        self.resnet18_embeddings = torch.nn.Sequential(*layers)

        if no_training:
            for param in self.resnet18_embeddings.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.resnet18_embeddings(x)
        return h

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, in_c, in_h, in_w, num_kp=None):
        super().__init__()
        self._spatial_conv = torch.nn.Conv2d(in_c, num_kp, kernel_size=1)

        # pos_x, pos_y = torch.meshgrid(torch.from_numpy(np.linspace(-1, 1, in_w)).float(),
        #                               torch.from_numpy(np.linspace(-1, 1, in_h)).float())

        pos_x, pos_y = torch.meshgrid(torch.from_numpy(np.linspace(-1, 1, in_w)).float(),
                                      torch.from_numpy(np.linspace(-1, 1, in_h)).float(),
                                      indexing='xy')

        pos_x = pos_x.reshape(1, in_w * in_h)
        pos_y = pos_y.reshape(1, in_w * in_h)
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        if num_kp is None:
            self._num_kp = in_c
        else:
            self._num_kp = num_kp

        self._in_c = in_c
        self._in_w = in_w
        self._in_h = in_h
        
    def forward(self, x):
        # assert(x.shape[1] == self._in_c)
        # assert(x.shape[2] == self._in_h)
        # assert(x.shape[3] == self._in_w)

        h = x
        if self._num_kp != self._in_c:
            h = self._spatial_conv(h)
        h = h.view(-1, self._in_h * self._in_w)
        attention = F.softmax(h, dim=-1)

        keypoint_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True).view(-1, self._num_kp)
        keypoint_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True).view(-1, self._num_kp)

        keypoints = torch.cat([keypoint_x, keypoint_y], dim=1)
        return keypoints

class PolicyLayer(torch.nn.Module):
    def __init__(self, input_dim, action_dim, activation='relu', action_scale=1., action_squash=True, layer_dims=[256, 256]):
        super().__init__()
        self.action_dim = action_dim
        self.action_scale = 1.0
        self.action_squash = action_squash
        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU

        self._layers = [torch.nn.Linear(input_dim, layer_dims[0]),
                        activate_fn()]
        for i in range(1, len(layer_dims)):
            self._layers += [torch.nn.Linear(layer_dims[i-1], layer_dims[i]),
                            activate_fn()]

        self._layers += [torch.nn.Linear(layer_dims[-1], action_dim)]
        self.layers = torch.nn.Sequential(*self._layers)

    def forward(self, x):
        h = self.layers(x)
        if self.action_squash:
            h = torch.tanh(h) * self.action_scale
        return h

class PolicyGMMLayer(torch.nn.Module):
    def __init__(
        self,
        # network_kwargs
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        min_std=0.0001,
        num_modes=5,
        activation="softplus",
        low_eval_noise=False,
        # loss_kwargs
        loss_coef=1.0,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.output_size = output_size
        self.min_std = min_std

        if num_layers > 0:
            sizes = [input_size] + [hidden_size] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        self.mean_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logstd_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logits_layer = nn.Linear(hidden_size, num_modes)

        self.low_eval_noise = low_eval_noise
        self.loss_coef = loss_coef

        if activation == "softplus":
            self.actv = F.softplus
        else:
            self.actv = torch.exp

    def forward_fn(self, x):
        # x: (B, input_size)
        share = self.share(x)
        means = self.mean_layer(share).view(-1, self.num_modes, self.output_size)
        means = torch.tanh(means)
        logits = self.logits_layer(share)

        if self.training or not self.low_eval_noise:
            logstds = self.logstd_layer(share).view(
                -1, self.num_modes, self.output_size
            )
            stds = self.actv(logstds) + self.min_std
        else:
            stds = torch.ones_like(means) * 1e-4
        return means, stds, logits

    def forward(self, x):
        if x.ndim == 3:
            means, scales, logits = TensorUtils.time_distributed(x, self.forward_fn)
        elif x.ndim < 3:
            means, scales, logits = self.forward_fn(x)

        compo = D.Normal(loc=means, scale=scales)
        compo = D.Independent(compo, 1)
        mix = D.Categorical(logits=logits)
        gmm = D.MixtureSameFamily(
            mixture_distribution=mix, component_distribution=compo
        )
        return gmm

    def loss_fn(self, gmm, target, reduction="mean"):
        log_probs = gmm.log_prob(target)
        loss = -log_probs
        if reduction == "mean":
            return loss.mean() * self.loss_coef
        elif reduction == "none":
            return loss * self.loss_coef
        elif reduction == "sum":
            return loss.sum() * self.loss_coef
        else:
            raise NotImplementedError
    
class ImgEncoder(torch.nn.Module):
    def __init__(self,
                 visual_embedding,
                 img_h,
                 img_w,
                 num_kp,
                 visual_feature_dimension=64,
                 use_eye_in_hand=True,
                 img_c=3,                 
                 pretrained=False,
                 activation='relu'):
        super().__init__()
        
        self._encoder = PerceptionEmbedding(use_eye_in_hand=use_eye_in_hand,
                                            img_c=img_c,
                                            pretrained=pretrained,
                                            activation=activation,
                                            )
        in_c = visual_embedding
        # pooling case
        in_h = img_h // 32
        in_w = img_w // 32
        self._spatial_softmax = SpatialSoftmax(in_c=in_c,
                                               in_h=in_h,
                                               in_w=in_w,
                                               num_kp=num_kp)
        self._fc = torch.nn.Sequential(torch.nn.Linear(num_kp * 2, visual_feature_dimension))

    def forward(self, x):
        h = self._encoder(x)
        h = self._spatial_softmax(h)
        h = self._fc(h)
        return h

class SubgoalPerceptionEmbedding(torch.nn.Module):
    def __init__(self,
                 pretrained=False,
                 no_training=False,
                 use_eye_in_hand=False,
                 use_spatial_softmax=True,
                 activation='relu',
                 img_c=3):

        super().__init__()

        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU
            
        layers = list(torchvision.models.resnet18(pretrained=pretrained).children())[:-3]

        if not use_spatial_softmax:
            layers += [torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                       torch.nn.Flatten(start_dim=1)]

        if use_eye_in_hand:
            conv0 = torch.nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            layers[0] = conv0
            print("Subgoal encoder using eye_in_hand")
            
        
        self.resnet18_embeddings = torch.nn.Sequential(*layers)

    def forward(self, x):
        h = self.resnet18_embeddings(x)
        return h
    
class SubgoalEncoder(torch.nn.Module):
    def __init__(self,
                 visual_embedding,
                 img_h,
                 img_w,
                 num_kp,
                 visual_feature_dimension=32,
                 img_c=3,                 
                 pretrained=False,
                 use_eye_in_hand=False,
                 use_spatial_softmax=True,
                 subgoal_type="sigmoid",
                 activation='relu'):
        super().__init__()

        self.use_eye_in_hand = use_eye_in_hand
        self.use_spatial_softmax = use_spatial_softmax

        self._encoder = SubgoalPerceptionEmbedding(img_c=img_c,
                                                   pretrained=pretrained,
                                                   activation=activation,
                                                   use_eye_in_hand=self.use_eye_in_hand,
                                                   use_spatial_softmax=use_spatial_softmax,
                                            )
        in_c = visual_embedding
        # pooling case
        in_h = img_h // 16
        in_w = img_w // 16

        self._spatial_softmax = SpatialSoftmax(in_c=in_c,
                                               in_h=in_h,
                                               in_w=in_w,
                                               num_kp=num_kp)

        
        activate_fn = get_activate_fn(activation)

        if self.use_spatial_softmax:
            fc_layers = [torch.nn.Linear(num_kp * 2, visual_feature_dimension)]
        else:
            fc_layers = [torch.nn.Linear(visual_embedding, visual_embedding // 2),
                                                 activate_fn(),
                                                 torch.nn.Linear(visual_embedding // 2, visual_feature_dimension)]

        if subgoal_type == "sigmoid":
            print("Using Sigmoid")
            fc_layers.append(torch.nn.Sigmoid())
        elif subgoal_type == "linear":
            print("Using Linear!")
        
        else:
            raise NotImplementedError
        self._fc = torch.nn.Sequential(*fc_layers)

    def forward(self, x):
        h = self._encoder(x)
        if self.use_spatial_softmax:
            h = self._spatial_softmax(h)
        h = self._fc(h)
        return h
    
    
class BCPolicy(torch.nn.Module):
    def __init__(self,
                 action_dim,
                 state_dim=None,
                 proprio_dim=5,
                 data_modality=["image", "proprio"],
                 use_eye_in_hand=True,
                 use_subgoal_eye_in_hand=False,
                 use_subgoal_spatial_softmax=True,
                 use_goal=False,
                 activation='relu',
                 action_squash=True,
                 z_dim=128,
                 num_kp=64,
                 img_h=128,
                 img_w=128,
                 subgoal_type="linear",
                 visual_feature_dimension=64,
                 subgoal_visual_feature_dimension=0,
                 policy_layer_dims=[256, 256],
                 policy_type=PolicyType.NORMAL_SUBGOAL):
        super().__init__()
        self._resnet_embedding_dim = 512

        self.action_dim = action_dim
        self.state_dim = state_dim

        self.data_modality = data_modality
        self.use_goal = use_goal
        self.action_squash = action_squash

        self.subgoal_type = subgoal_type
        self.use_subgoal_noise = True
        self.subgoal_noise_std = 0.1 * (subgoal_visual_feature_dimension / 32)

        self.policy_type = policy_type
        
        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU

        assert((subgoal_visual_feature_dimension == 0 and policy_type == PolicyType.NO_SUBGOAL) or (subgoal_visual_feature_dimension > 0 and policy_type!=PolicyType.NO_SUBGOAL))

        if "proprio" not in data_modality:
            proprio_dim = 0
        self.num_kp = num_kp
        if "image" in self.data_modality:
            self._img_encoder = ImgEncoder(visual_embedding=self._resnet_embedding_dim,
                                           img_h=img_h,
                                           img_w=img_w,
                                           num_kp=num_kp,
                                           visual_feature_dimension=visual_feature_dimension,
                                           use_eye_in_hand=use_eye_in_hand
                                           )
            self._policy_layers = PolicyLayer(input_dim=visual_feature_dimension + proprio_dim + subgoal_visual_feature_dimension,
                                              layer_dims=policy_layer_dims,
                                              action_dim=action_dim,
                                              activation=activation,
                                              action_squash=action_squash)
            if self.policy_type == PolicyType.NORMAL_SUBGOAL:
                self._subgoal_encoder = SubgoalEncoder(visual_embedding=256,
                                                       img_h=img_h,
                                                       img_w=img_w,
                                                       num_kp=num_kp,
                                                       visual_feature_dimension=subgoal_visual_feature_dimension,
                                                       use_spatial_softmax=use_subgoal_spatial_softmax,
                                                       use_eye_in_hand=use_subgoal_eye_in_hand,
                                                       subgoal_type=self.subgoal_type)
            
    def forward(self, x, is_training=True):

        if "image" in self.data_modality:
            embedding = self._img_encoder(x["state_image"])

            if self.policy_type == PolicyType.NORMAL_SUBGOAL:
                subgoal_embedding = self._subgoal_encoder(x["subgoal"])
                if self.use_subgoal_noise:
                    subgoal_embedding = self.subgoal_noise(subgoal_embedding)
                embedding = torch.cat([embedding, subgoal_embedding], dim=1)

            elif self.policy_type == PolicyType.VAE_SUBGOAL:
                assert("vae_embedding" in x)
                embedding = torch.cat([embedding, x["vae_embedding"]], dim=1)

            if "proprio" in self.data_modality:
                embedding = torch.cat([embedding, x["proprio"]], dim=1)
                
            action = self._policy_layers(embedding)
        else:
            action = self._policy_layers(x["state"])
        return action

    def subgoal_noise(self, x):
        eps = torch.randn_like(x)
        return eps * self.subgoal_noise_std + x
        

    def get_action(self, x):
        st_embedding = self._img_encoder(x["state_image"])
        embedding = torch.cat([st_embedding, x["subgoal_embedding"]], dim=1)
        if "proprio" in self.data_modality:
            embedding = torch.cat([embedding, x["proprio"]], dim=1)
        action = self._policy_layers(embedding)
        return action

    def get_embedding(self, subgoal_x):
        assert(self.policy_type==PolicyType.NORMAL_SUBGOAL)
        return self._subgoal_encoder(subgoal_x)

class BCGMMPolicy(torch.nn.Module):
    def __init__(self,
                 action_dim,
                 state_dim=None,
                 proprio_dim=5,
                 data_modality=["image", "proprio"],
                 use_eye_in_hand=True,
                 use_subgoal_eye_in_hand=False,
                 use_subgoal_spatial_softmax=True,
                 use_goal=False,
                 activation='relu',
                 action_squash=True,
                 z_dim=128,
                 num_kp=64,
                 img_h=128,
                 img_w=128,
                 subgoal_type="linear",
                 visual_feature_dimension=64,
                 subgoal_visual_feature_dimension=0,
                 policy_layer_dims=[256, 256],
                 policy_type=PolicyType.NORMAL_SUBGOAL):
        super().__init__()
        self._resnet_embedding_dim = 512

        self.action_dim = action_dim
        self.state_dim = state_dim

        self.data_modality = data_modality
        self.use_goal = use_goal
        self.action_squash = action_squash

        self.subgoal_type = subgoal_type
        self.use_subgoal_noise = True
        self.subgoal_noise_std = 0.1 * (subgoal_visual_feature_dimension / 32)

        self.policy_type = policy_type
        
        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU

        assert((subgoal_visual_feature_dimension == 0 and policy_type == PolicyType.NO_SUBGOAL) or (subgoal_visual_feature_dimension > 0 and policy_type!=PolicyType.NO_SUBGOAL))

        if "proprio" not in data_modality:
            proprio_dim = 0
        self.num_kp = num_kp
        if "image" in self.data_modality:
            self._img_encoder = ImgEncoder(visual_embedding=self._resnet_embedding_dim,
                                           img_h=img_h,
                                           img_w=img_w,
                                           num_kp=num_kp,
                                           visual_feature_dimension=visual_feature_dimension,
                                           use_eye_in_hand=use_eye_in_hand
                                           )
            # self._policy_layers = PolicyLayer(input_dim=visual_feature_dimension + proprio_dim + subgoal_visual_feature_dimension,
            #                                   layer_dims=policy_layer_dims,
            #                                   action_dim=action_dim,
            #                                   activation=activation,
            #                                   action_squash=action_squash)
            self._policy_layers = PolicyGMMLayer(input_size=visual_feature_dimension + proprio_dim + subgoal_visual_feature_dimension,
                                                output_size=action_dim,
                                                hidden_size=512,
                                                num_layers=2,
                                                min_std=0.0001,
                                                num_modes=5,
                                                low_eval_noise=False,
                                                activation="softplus",
                                                loss_coef=1.0)
            if self.policy_type == PolicyType.NORMAL_SUBGOAL:
                self._subgoal_encoder = SubgoalEncoder(visual_embedding=256,
                                                       img_h=img_h,
                                                       img_w=img_w,
                                                       num_kp=num_kp,
                                                       visual_feature_dimension=subgoal_visual_feature_dimension,
                                                       use_spatial_softmax=use_subgoal_spatial_softmax,
                                                       use_eye_in_hand=use_subgoal_eye_in_hand,
                                                       subgoal_type=self.subgoal_type)
            
    def forward(self, x, is_training=True):

        if "image" in self.data_modality:
            embedding = self._img_encoder(x["state_image"])

            if self.policy_type == PolicyType.NORMAL_SUBGOAL:
                subgoal_embedding = self._subgoal_encoder(x["subgoal"])
                if self.use_subgoal_noise:
                    subgoal_embedding = self.subgoal_noise(subgoal_embedding)
                embedding = torch.cat([embedding, subgoal_embedding], dim=1)

            elif self.policy_type == PolicyType.VAE_SUBGOAL:
                assert("vae_embedding" in x)
                embedding = torch.cat([embedding, x["vae_embedding"]], dim=1)

            if "proprio" in self.data_modality:
                embedding = torch.cat([embedding, x["proprio"]], dim=1)
                
            # action = self._policy_layers(embedding)
            dist = self._policy_layers(embedding)
        else:
            # action = self._policy_layers(x["state"])
            dist = self._policy_layers(x["state"])
        return dist

    def subgoal_noise(self, x):
        eps = torch.randn_like(x)
        return eps * self.subgoal_noise_std + x
        

    def get_action(self, x):
        st_embedding = self._img_encoder(x["state_image"])
        embedding = torch.cat([st_embedding, x["subgoal_embedding"]], dim=1)
        if "proprio" in self.data_modality:
            embedding = torch.cat([embedding, x["proprio"]], dim=1)
        # action = self._policy_layers(embedding)
        dist = self._policy_layers(embedding)
        action = dist.sample()
        return action

    def get_embedding(self, subgoal_x):
        assert(self.policy_type==PolicyType.NORMAL_SUBGOAL)
        return self._subgoal_encoder(subgoal_x)


class BaselineBCPolicy(torch.nn.Module):
    def __init__(self,
                 action_dim,
                 state_dim=None,
                 proprio_dim=5,
                 data_modality=["image", "proprio"],
                 use_eye_in_hand=True,
                 use_subgoal_eye_in_hand=False,
                 use_subgoal_spatial_softmax=True,
                 use_goal=False,
                 activation='relu',
                 action_squash=True,
                 z_dim=128,
                 num_kp=64,
                 img_h=128,
                 img_w=128,
                 visual_feature_dimension=64,
                 subgoal_visual_feature_dimension=0,
                 policy_layer_dims=[256, 256]):
        super().__init__()
        self._resnet_embedding_dim = 512

        self.action_dim = action_dim
        self.state_dim = state_dim

        self.data_modality = data_modality
        self.action_squash = action_squash
        
        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU

        if "proprio" not in data_modality:
            proprio_dim = 0
        self.num_kp = num_kp
        if "image" in self.data_modality:
            self._img_encoder = ImgEncoder(visual_embedding=self._resnet_embedding_dim,
                                           img_h=img_h,
                                           img_w=img_w,
                                           num_kp=num_kp,
                                           visual_feature_dimension=visual_feature_dimension,
                                           use_eye_in_hand=use_eye_in_hand
                                           )
            self._policy_layers = PolicyLayer(input_dim=visual_feature_dimension + proprio_dim,
                                              layer_dims=policy_layer_dims,
                                              action_dim=action_dim,
                                              activation=activation,
                                              action_squash=action_squash)

    def forward(self, x, is_training=True):

        if "image" in self.data_modality:
            embedding = self._img_encoder(x["state_image"])
            if "proprio" in self.data_modality:
                embedding = torch.cat([embedding, x["proprio"]], dim=1)
            action = self._policy_layers(embedding)
        return action


    def get_action(self, x):
        embedding = self._img_encoder(x["state_image"])
        if "proprio" in self.data_modality:
            embedding = torch.cat([embedding, x["proprio"]], dim=1)
        action = self._policy_layers(embedding)
        return action


class SubgoalVAE(torch.nn.Module):
    def __init__(self,
                 z_dim=32,
                 activation="relu",
                 decoder_layer_dims=[300, 400],
                 output_size=64*64):
        super().__init__()
        self.z_dim = z_dim

        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU
        
        # resnet18
        self.state_encoder = VAEPerceptionEmbedding(
            img_c=3,
            visual_feature_dimension=64)

        self.subgoal_encoder = VAEPerceptionEmbedding(
            img_c=3,
            visual_feature_dimension=64)
        
        # fully connected layer

        layer_dims = decoder_layer_dims + [output_size]

        self.mlp_layers = torch.nn.Sequential(*[torch.nn.Linear(64 + 64, 128),
                                                activate_fn(),
                                                torch.nn.Linear(128, z_dim*2)])
    
        decoder_layers = [torch.nn.Linear(self.z_dim, layer_dims[0])]
        for i in range(len(layer_dims) - 1):
            decoder_layers += [activate_fn(), torch.nn.Linear(layer_dims[i], layer_dims[i+1])]
        decoder_layers.append(torch.nn.Sigmoid())
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, x):
        h_1 = self.state_encoder(x["subgoal"])
        h_2 = self.subgoal_encoder(x["state_image"][:, :3, ...])

        h = self.mlp_layers(torch.cat([h_1, h_2], dim=1))
        mu, logvar = torch.split(h, self.z_dim, dim=1)

        z = self.sampling(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar, z

    def sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps * std + mu

    def get_embedding(self, x):
        h = self.encoder(x["subgoal"])
        mu, logvar = torch.split(h, self.z_dim, dim=1)
        return mu

class MetaDecisionLayer(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_subtasks,
                 subgoal_embedding_dim,
                 id_layer_dims,
                 embedding_layer_dims,
                 policy_type,
                 subgoal_type="sigmoid",
                 activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.num_subtasks = num_subtasks
        self.subgoal_embedding_dim = subgoal_embedding_dim
        self.policy_type = policy_type

        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU

        self._id_layers = [torch.nn.Linear(input_dim, id_layer_dims[0]),
                        activate_fn()]
        for i in range(1, len(id_layer_dims)):
            self._id_layers += [torch.nn.Linear(id_layer_dims[i-1], id_layer_dims[i]),
                            activate_fn()]

        self._id_layers += [torch.nn.Linear(id_layer_dims[-1], num_subtasks),
                         torch.nn.Softmax(dim=1)]
        self.id_layers = torch.nn.Sequential(*self._id_layers)

        self._embedding_layers = [torch.nn.Linear(input_dim, embedding_layer_dims[0]),
                        activate_fn()]
        for i in range(1, len(embedding_layer_dims)):
            self._embedding_layers += [torch.nn.Linear(embedding_layer_dims[i-1], embedding_layer_dims[i]),
                            activate_fn()]

        self._embedding_layers += [torch.nn.Linear(embedding_layer_dims[-1], subgoal_embedding_dim)]
        if self.policy_type == "normal_subgoal":
            if subgoal_type == "sigmoid":
                print("Using sigmoid")
                self._embedding_layers.append(torch.nn.Sigmoid())
        self.embedding_layers = torch.nn.Sequential(*self._embedding_layers)

    def forward(self, x):
        return self.id_layers(x), self.embedding_layers(x)


class MetaIdLayer(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_subtasks,
                 id_layer_dims,
                 policy_type,
                 subgoal_type="sigmoid",
                 activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.num_subtasks = num_subtasks
        self.policy_type = policy_type

        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU

        self._id_layers = [torch.nn.Linear(input_dim, id_layer_dims[0]),
                        activate_fn()]
        for i in range(1, len(id_layer_dims)):
            self._id_layers += [torch.nn.Linear(id_layer_dims[i-1], id_layer_dims[i]),
                            activate_fn()]

        self._id_layers += [torch.nn.Linear(id_layer_dims[-1], num_subtasks),
                         torch.nn.Softmax(dim=1)]
        self.id_layers = torch.nn.Sequential(*self._id_layers)

    def forward(self, x):
        return self.id_layers(x)

class MetaEmbeddingLayer(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_subtasks,
                 subgoal_embedding_dim,
                 embedding_layer_dims,
                 policy_type,
                 subgoal_type="sigmoid",
                 activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.num_subtasks = num_subtasks
        self.subgoal_embedding_dim = subgoal_embedding_dim
        self.policy_type = policy_type

        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU

        self._embedding_layers = [torch.nn.Linear(input_dim, embedding_layer_dims[0]),
                        activate_fn()]
        for i in range(1, len(embedding_layer_dims)):
            self._embedding_layers += [torch.nn.Linear(embedding_layer_dims[i-1], embedding_layer_dims[i]),
                            activate_fn()]

        self._embedding_layers += [torch.nn.Linear(embedding_layer_dims[-1], subgoal_embedding_dim)]
        if self.policy_type == "normal_subgoal":
            if subgoal_type == "sigmoid":
                print("Using sigmoid")
                self._embedding_layers.append(torch.nn.Sigmoid())
        self.embedding_layers = torch.nn.Sequential(*self._embedding_layers)

    def forward(self, x):
        return self.embedding_layers(x)
    

class MetaPolicy(torch.nn.Module):
    def __init__(self,
                 num_subtasks,
                 subgoal_embedding_dim,
                 id_layer_dims,
                 embedding_layer_dims,
                 use_eye_in_hand,
                 policy_type,
                 activation='relu',
                 subgoal_type="sigmoid",
                 latent_dim=32,                 
                 use_spatial_softmax=False,
                 num_kp=64,
                 visual_feature_dimension=64,
    ):
        super().__init__()
        self.num_subtasks = num_subtasks
        self.subgoal_embedding_dim = subgoal_embedding_dim

        self.use_spatial_softmax = use_spatial_softmax
        if use_spatial_softmax:
            remove_layer_num = 2
        else:
            remove_layer_num = 1
            
        self.perception_embedding_layer = PerceptionEmbedding(use_eye_in_hand=use_eye_in_hand,
                                                              activation=activation,
                                                              remove_layer_num=1)

        if use_spatial_softmax:
            in_c = 512
            in_h = 4
            in_w = 4
            self._spatial_softmax_layers = torch.nn.Sequential(*[SpatialSoftmax(in_c=in_c,
                                                                                in_h=in_h,
                                                                                in_w=in_w,
                                                                                num_kp=num_kp),
                                                                 torch.nn.Linear(num_kp * 2, visual_feature_dimension)])

        if use_spatial_softmax:
            print("Using Spatial softmax for meta policy")
            input_dim = visual_feature_dimension
        else:
            input_dim = 512

        self.meta_decision_layer = MetaDecisionLayer(input_dim,
                                                     num_subtasks,
                                                     subgoal_embedding_dim,
                                                     id_layer_dims=id_layer_dims,
                                                     embedding_layer_dims=embedding_layer_dims,
                                                     policy_type=policy_type,
                                                     activation=activation,
                                                     subgoal_type=subgoal_type)

    def forward(self, x):
        h = self.perception_embedding_layer(x["state_image"])
        if self.use_spatial_softmax:
            h = self._spatial_softmax_layers(h)
        else:
            h = torch.flatten(h, start_dim=1)
        return self.meta_decision_layer(h)

    def predict(self, x):
        subtask_vector, subgoal_embedding = self.forward(x)
        subtask_id = torch.argmax(subtask_vector, dim=1).cpu().detach().numpy()
        return {"subtask_id": subtask_id,
                "subtask_vector": subtask_vector,
                "embedding": subgoal_embedding}


class MetaCVAEPolicy(torch.nn.Module):
    def __init__(self,
                 num_subtasks,
                 subgoal_embedding_dim,
                 id_layer_dims,
                 embedding_layer_dims,
                 use_eye_in_hand,
                 policy_type,
                 activation='relu',
                 subgoal_type="sigmoid",
                 latent_dim=32,
                 use_spatial_softmax=False,
                 num_kp=64,
                 visual_feature_dimension=64,
                 separate_id_prediction=False,
    ):
        super().__init__()
        self.num_subtasks = num_subtasks
        self.subgoal_embedding_dim = subgoal_embedding_dim

        self.use_spatial_softmax = use_spatial_softmax
        if use_spatial_softmax:
            remove_layer_num = 2
        else:
            remove_layer_num = 1
        
        self.perception_encoder_layer = PerceptionEmbedding(use_eye_in_hand=use_eye_in_hand,
                                                              activation=activation,
                                                              remove_layer_num=remove_layer_num)

        if use_spatial_softmax:
            in_c = 512
            in_h = 4
            in_w = 4
            self._spatial_softmax_layers = torch.nn.Sequential(*[SpatialSoftmax(in_c=in_c,
                                                                                in_h=in_h,
                                                                                in_w=in_w,
                                                                                num_kp=num_kp),
                                                                 torch.nn.Linear(num_kp * 2, visual_feature_dimension)])

        if use_spatial_softmax:
            print("Using Spatial softmax for meta policy")
            input_dim = visual_feature_dimension
        else:
            input_dim = 512

        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU
        
        self.latent_dim = latent_dim
        id_dim = 0

        intermediate_state_dim = 256
        self.mlp_encoder_layer = torch.nn.Sequential(*[torch.nn.Linear(input_dim + subgoal_embedding_dim + id_dim, intermediate_state_dim),
                                                              activate_fn(),
                                                              torch.nn.Linear(intermediate_state_dim, intermediate_state_dim),
                                                       activate_fn(),
                                                       torch.nn.Linear(intermediate_state_dim, self.latent_dim * 2)])

        # self.meta_decision_layer = MetaDecisionLayer(input_dim + self.latent_dim,
        #                                              num_subtasks,
        #                                              subgoal_embedding_dim,
        #                                              id_layer_dims=id_layer_dims,

        #                                              embedding_layer_dims=embedding_layer_dims,
        #                                              policy_type=policy_type,
        #                                              subgoal_type=subgoal_type,
        #                                              activation=activation)

        id_input_dim = input_dim
        embedding_input_dim = input_dim + self.latent_dim + id_dim

        self.meta_id_layer = MetaIdLayer(id_input_dim,
                                         num_subtasks,
                                         id_layer_dims=id_layer_dims,
                                         policy_type=policy_type,
                                         subgoal_type=subgoal_type,
                                         activation=activation)
        self.meta_embedding_layer = MetaEmbeddingLayer(embedding_input_dim,
                                                       num_subtasks,
                                                       subgoal_embedding_dim,
                                                       embedding_layer_dims=embedding_layer_dims,
                                                       policy_type=policy_type,
                                                       subgoal_type=subgoal_type,
                                                       activation=activation)
        

    def forward(self, x):
        # state_image:(100, 3, 128, 128); embedding:(100, 32); id_vector:(100, 6)
        z_state = self.perception_encoder_layer(x["state_image"])
        if self.use_spatial_softmax:
            z_state = self._spatial_softmax_layers(z_state)
        else:
            z_state = torch.flatten(z_state, start_dim=1)
        h = torch.cat([z_state, x["embedding"]], dim=1) # (Batch, 512 + 32)
        h = self.mlp_encoder_layer(h) # (Batch, 128)

        mu, logvar = torch.split(h, self.latent_dim, dim=1)
        z = self.sampling(mu, logvar) # (Batch, 64)
        z_concat = torch.cat([z_state, z], dim=1) # (Batch, 512 + 64)
        # z_concat = torch.cat([z_concat, x["id_vector"]], dim=1) # (Batch, 512 + 64 + 6)
        skill_id = self.meta_id_layer(z_state) # (Batch, 6)
        embedding = self.meta_embedding_layer(z_concat)
        return skill_id, embedding, mu, logvar


    def predict(self, x):
        z_state = self.perception_encoder_layer(x["state_image"])
        if self.use_spatial_softmax:
            z_state = self._spatial_softmax_layers(z_state)
        else:
            z_state = torch.flatten(z_state, start_dim=1)
        
        z = safe_cuda(torch.randn(x["state_image"].shape[0], self.latent_dim))
        # subtask_vector, subgoal_embedding = self.meta_decision_layer(torch.cat([z_state, z], dim=1))
        subtask_vector = self.meta_id_layer(z_state)
        subtask_id = torch.argmax(subtask_vector, dim=1)
        z_concat = [z_state, z]

        subgoal_embedding = self.meta_embedding_layer(torch.cat(z_concat, dim=1))

        subtask_id = subtask_id.cpu().detach().numpy()
        return {"subtask_id": subtask_id,
                "subtask_vector": subtask_vector,
                "embedding": subgoal_embedding}

    def sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps * std + mu
