import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import torch
import torch.nn as nn
class ModelWithUncertainty(nn.Module):
    def __init__(self, baseModel, last_layer, in_train_loss_fn, in_nested_sets_from_output_fn, params):
        super(ModelWithUncertainty, self).__init__()
        self.baseModel = baseModel
        self.last_layer = last_layer
        self.register_buffer('lhat',None)
        self.in_train_loss_fn = in_train_loss_fn
        self.in_nested_sets_from_output_fn = in_nested_sets_from_output_fn
        self.params = params
    def forward(self, x):
        x = self.baseModel(x)
        return self.last_layer(x)
    def loss_fn(self, pred, target):
        return self.in_train_loss_fn(pred, target, self.params)
    def nested_sets_from_output(self, output, lam=None):
        lower_edge, prediction, upper_edge = self.in_nested_sets_from_output_fn(self, output, lam)
        upper_edge = torch.maximum(upper_edge, prediction + 1e-6) # set a lower bound on the size.
        lower_edge = torch.minimum(lower_edge, prediction - 1e-6)
        return lower_edge, prediction, upper_edge
    def nested_sets(self, x, lam=None):
        if lam == None:
            if self.lhat == None:
                raise Exception("You have to specify lambda unless your model is already calibrated.")
            lam = self.lhat
        output = self(*x)
        return self.nested_sets_from_output(output, lam=lam)
    def set_lhat(self,lhat):
        self.lhat = lhat
class PinballLoss():
    def __init__(self, quantile=0.10, reduction='mean'):
        self.quantile = quantile
        assert 0 < self.quantile
        assert self.quantile < 1
        self.reduction = reduction
        
    def __call__(self, output, target):
#         print('\n IN UNCERTAINTY LAYER = ', output.shape)
#         print('IN UNCERTAINTY LAYER = ', target.shape)
        assert output.shape == target.shape
        loss = torch.zeros_like(target, dtype=torch.float)
        error = output - target
        smaller_index = error < 0
        bigger_index = 0 < error
        loss[smaller_index] = self.quantile * abs(error).float()[smaller_index]
        loss[bigger_index] = (1-self.quantile) * abs(error).float()[bigger_index]
        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss
class QuantileRegressionLayer(nn.Module):
    def __init__(self, n_channels_middle, n_channels_out, params):
        super(QuantileRegressionLayer, self).__init__()
        self.q_lo = params["q_lo"]
        self.q_hi = params["q_hi"]
        self.params = params
        self.lower = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)
        self.prediction = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)
        self.upper = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)
    def forward(self, x):
        output = torch.cat((self.lower(x).unsqueeze(1), self.prediction(x).unsqueeze(1), self.upper(x).unsqueeze(1)), dim=1)
        return output
def quantile_regression_loss_fn(pred, target, params):
    q_lo_loss = PinballLoss(quantile=params['q_lo'])
    q_hi_loss = PinballLoss(quantile=params['q_hi'])
    mse_loss = nn.MSELoss()
    loss = 1 * q_lo_loss(pred[:,0,:,:,:].squeeze(), target.squeeze()) + \
            1* q_hi_loss(pred[:,2,:,:,:].squeeze(), target.squeeze()) + \
            1 * mse_loss(pred[:,1,:,:,:].squeeze(), target.squeeze())
        # params[‘q_lo_weight’] 1 * q_lo_loss(pred[:,0,:,:,:].squeeze(), target.squeeze()) + \
        #  params[‘q_hi_weight’] 1* q_hi_loss(pred[:,2,:,:,:].squeeze(), target.squeeze()) + \
        #  params[‘mse_weight’] 1 * mse_loss(pred[:,1,:,:,:].squeeze(), target.squeeze())
    return loss
def quantile_regression_nested_sets_from_output(model, output, lam=None):
    if lam == None:
        if model.lhat == None:
            raise Exception('You have to specify lambda unless your model is already calibrated.')
        lam = model.lhat
    output[:,0,:,:,:] = torch.minimum(output[:,0,:,:,:], output[:,1,:,:,:]-1e-6)
    output[:,2,:,:,:] = torch.maximum(output[:,2,:,:,:], output[:,1,:,:,:]+1e-6)
    upper_edge = lam * (output[:,2,:,:,:] - output[:,1,:,:,:]) + output[:,1,:,:,:]
    lower_edge = output[:,1,:,:,:] - lam * (output[:,1,:,:,:] - output[:,0,:,:,:])
    return lower_edge, output[:,1,:,:,:], upper_edge
def add_uncertainty(model):
    last_layer = None
    train_loss_fn = None
    nested_sets_from_output_fn = None
    params = {
        'q_lo': 0.05,
        'q_hi': 0.95,
    }
    last_layer = QuantileRegressionLayer(model.n_channels_middle, model.n_channels_out, params)
    train_loss_fn = quantile_regression_loss_fn
    nested_sets_from_output_fn = quantile_regression_nested_sets_from_output
    return ModelWithUncertainty(model, last_layer, train_loss_fn, nested_sets_from_output_fn, params)
