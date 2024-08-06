import torch.nn.functional as F
import torch.nn as nn

# --------------------------- decoder -----------------------------

# predict the sdf (opposite sign to the actual sdf)
# unit is already m
def sdf(self, features):
    
    for k, l in enumerate(self.layers):
        if k == 0:
            if self.use_leaky_relu:
                h = F.leaky_relu(l(features))
            else:
                h = F.relu(l(features))
        else:
            if self.use_leaky_relu:
                h = F.leaky_relu(l(h))
            else:
                h = F.relu(l(h))

    out = self.lout(h).squeeze(1)
    out *= self.sdf_scale
    # linear (feature_dim -> hidden_dim)
    # relu
    # linear (hidden_dim -> hidden_dim)
    # relu
    # linear (hidden_dim -> 1)

    return out


# _init_
position_dim = config.pos_input_dim * (2 * config.pos_encoding_band + 1) # 3
            
feature_dim = config.feature_dim # 8
input_layer_count = feature_dim + position_dim # 8+3

# predict sdf (now it anyway only predict sdf without further sigmoid
# Initializa the structure of shared MLP
hidden_dim = 64
hidden_level = 1
out_dim = 1
layers = []
for i in range(hidden_level):
    if i == 0:
        # layers.append(nn.Linear(input_layer_count, hidden_dim, bias_on)) # 11, 64
        layers.extend([
            nn.Linear(input_layer_count, hidden_dim, bias_on),
            nn.ReLU(inplace=True)
        ])
    else:
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias_on))
self.layers = nn.ModuleList(layers) # hidden_level = 1
self.lout = nn.Linear(hidden_dim, out_dim, bias_on) # 64, 1


# --------------------------- mapper ---------------------
# in tool.py
def setup_optimizer(config: Config, neural_point_feat, mlp_geo_param = None, 
                    mlp_sem_param = None, mlp_color_param = None, poses = None, lr_ratio = 1.0) -> Optimizer:
    lr_cur = config.lr * lr_ratio
    lr_pose = config.lr_pose
    weight_decay = config.weight_decay
    opt_setting = []
    # weight_decay is for L2 regularization, only applied to MLP
    if mlp_geo_param is not None: 
        mlp_geo_param_opt_dict = {'params': mlp_geo_param, 'lr': lr_cur, 'weight_decay': weight_decay} 
        opt_setting.append(mlp_geo_param_opt_dict)

    # feature octree
    feat_opt_dict = {'params': neural_point_feat, 'lr': lr_cur, 'weight_decay': weight_decay} 
    opt_setting.append(feat_opt_dict)
    if config.opt_adam:
        opt = optim.Adam(opt_setting, betas=(0.9,0.99), eps = config.adam_eps) 
    else:
        opt = optim.SGD(opt_setting, momentum=0.9)
    
    return opt    

# PIN map online training (mapping) given the fixed pose
def mapping(self, iter_count): 

    if self.train_less:
        iter_count = max(1, iter_count-5)

    neural_point_feat = list(self.neural_points.parameters()) # geo_features 8
    geo_mlp_param = list(self.geo_mlp.parameters()) # decoder trainable params 11


    # -tools
    opt = setup_optimizer(self.config, neural_point_feat, geo_mlp_param, sem_mlp_param, color_mlp_param)

    elif self.config.numerical_grad: # use for mapping # do not use this for the tracking, still analytical grad for tracking   
        g = self.get_numerical_gradient(coord[::self.config.gradient_decimation], 
                                        sdf_pred[::self.config.gradient_decimation],
                                        self.config.voxel_size_m*self.config.num_grad_step_ratio) 
    
    # calculate the loss            
    cur_loss = 0.0
    weight = torch.abs(weight).detach() # weight's sign indicate the sample is around the surface or in the free space
    if self.config.main_loss_type == 'bce': # [used]
        sdf_loss = sdf_bce_loss(sdf_pred, sdf_label, self.sdf_scale, weight, self.config.loss_weight_on) 
    # ekional loss
    eikonal_loss = ((g_used.norm(2, dim=-1) - 1.0) ** 2).mean() # both the surface and the freespace
    

    opt.zero_grad(set_to_none=True)
    cur_loss.backward(retain_graph=False)
    opt.step()

# ---------------------------------neuralPoints -----------------------
# _init_
self.local_geo_features = nn.Parameter() # learnable parameters, will be optimized during training.

# --------------------------------pinslam
# tool
def freeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
def freeze_decoders(geo_decoder, sem_decoder, color_decoder, config):
    if not config.silence:
        print("Freeze the decoder")
    freeze_model(geo_decoder) # fixed the geo decoder
#
if used_frame_id == config.freeze_after_frame: # freeze the decoder after certain frame 
    freeze_decoders(geo_mlp, sem_mlp, color_mlp, config)