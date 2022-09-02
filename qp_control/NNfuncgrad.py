import itertools
from typing import Tuple, List, Optional
from collections import OrderedDict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np 


class CBF(nn.Module):

    def __init__(self, dynamics, n_state, m_control, preprocess_func=None):
        super().__init__()
        self.n_state = n_state
        self.m_control = m_control
        self.dynamics = dynamics
        self.preprocess_func = preprocess_func

        self.n_dims_extended = self.n_state
        self.cbf_hidden_layers = 3
        self.cbf_hidden_size = 128

        self.V_layers: OrderedDict[str, nn.Module] = OrderedDict()

        self.V_layers["input_linear"] = nn.Linear(
            self.n_dims_extended, self.cbf_hidden_size
        )
        self.V_layers["input_activation"] = nn.Tanh()
        for i in range(self.cbf_hidden_layers):
            self.V_layers[f"layer_{i}_linear"] = nn.Linear(
                self.cbf_hidden_size, self.cbf_hidden_size
            )
            if i < self.cbf_hidden_layers - 1:
                self.V_layers[f"layer_{i}_activation"] = nn.Tanh()
        self.V_layers["output_linear"] = nn.Linear(self.cbf_hidden_size, 1)
        self.V_nn = nn.Sequential(self.V_layers)



    def forward(self, state):
        """
        args:
            state (bs, n_state)
            obstacle (bs, k_obstacle, n_state)
        returns:
            h (bs, k_obstacle)
        """
        # state = torch.unsqueeze(state, 2)    # (bs, n_state, 1)
        # state_diff = state
        
        # if self.preprocess_func is not None:
        #     state_diff = self.preprocess_func(state_diff)
        
        # x = self.activation(self.conv0(state_diff))
        # x = self.activation(self.conv1(x))
        # x = self.activation(self.conv2(x))   # (bs, 128, k_obstacle)
        # x = self.activation(self.conv3(x))
        # x = self.conv4(x)
        # h = torch.squeeze(x, dim=1)          # (bs, k_obstacle)

        h, Jh = self.V_with_jacobian(state)
        # H = torch.tensor(h).reshape(1,1)
        # JH = torch.tensor(Jh).reshape(1,self.n_state)
        HJH = torch.hstack((h.reshape(1,1), Jh.reshape(1,self.n_state)))
        # dh1 = F.conv1d(h,x)
        return HJH

    def V_with_jacobian(self, x: torch.Tensor):
        """Computes the CLBF value and its Jacobian
        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLBF
        returns:
            V: bs tensor of CLBF values
            JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
        """
        x_norm = torch.unsqueeze(x, 2)    # (bs, n_state, 1)
        bs = x_norm.shape[0]
        x_norm = x_norm.reshape(bs,self.n_state,1)

        # print(x_norm)
        # print(x_norm.shape)
        x_norm, x_range = self.normalize(x_norm)
        
        # print(x_norm)
        x_range = x_range.reshape(self.dynamics.n_dims)

        # print(x_range.shape)
        
        # print(x_norm.shape)
        x_norm = x_norm.reshape(bs, self.n_state)

        JV = torch.zeros(
            (bs, self.dynamics.n_dims, self.dynamics.n_dims)).type_as(x)

        # print(JV.shape)

        for dim in range(self.dynamics.n_dims):
            JV[:, dim, dim] = 1.0 / x_range[dim].type_as(x)
        # print(JV.shape)
       
        # Now step through each layer in V
        V = x_norm

        # print(self.V_nn)
        for layer in self.V_nn:
            V = layer(V)

            if isinstance(layer, nn.Linear):
                JV = torch.matmul(layer.weight, JV)
                # print("Linear")
                # print(JV.shape)
            elif isinstance(layer, nn.Tanh):
                JV = torch.matmul(torch.diag_embed(1 - V ** 2), JV)
                # print("Tanh")
                # print(JV.shape)
            elif isinstance(layer, nn.ReLU):
                # print("Relu")
                JV = torch.matmul(torch.diag_embed(torch.sign(V)), JV)
                # print(JV.shape)
            # print(V.shape)
        return V, JV

    def normalize(self, x: torch.Tensor, k: float = 1.0):
        """Normalize the state input to [-k, k]

        args:
            dynamics_model: the dynamics model matching the provided states
            x: bs x self.dynamics_model.n_dims the points to normalize
            k: normalize non-angle dimensions to [-k, k]
        """
        shape_x = x.shape

        # print(shape_x)

        x_max, x_min = self.dynamics.state_limits()

        x_center = torch.tensor(x_max + x_min).type_as(x) / 2
        # x_center.to(torch.device('cuda'))

        x_center = x_center.reshape(1,self.n_state,1)
        # print(x_center.shape)
        x_range = (x_max - x_min) / 2.0
        # Scale to get the input between (-k, k), centered at 0
        x_range = x_range / k
        # x_range.to(torch.device('cuda'))
        x_norm = x - x_center # .type_as(x) #.reshape(shape_x)
        x_range = x_range.reshape(1,self.n_state,1)
        # print(x_norm.shape)
        # print(x_center.shape)
        x_norm = x_norm / x_range.type_as(x)
        # x_norm = torch.div(x_norm, x_range.type_as(x))
        # We shouldn't scale or offset any angle dimensions
        # print(x_norm.shape)

        # Do the normalization
        return x_norm, x_range



class alpha_param(nn.Module):

    def __init__(self, n_state, preprocess_func=None):
        super().__init__()
        self.n_state = n_state

        self.preprocess_func = preprocess_func

        self.conv0 = nn.Conv1d(n_state, 64, 1)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 1, 1)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()


    def forward(self, state):
        """
        args:
            state (bs, n_state)
            obstacle (bs, k_obstacle, n_state)
        returns:
            h (bs, k_obstacle)
        """
        state = torch.unsqueeze(state, 2)    # (bs, n_state, 1)
        state_diff = state

        if self.preprocess_func is not None:
            state_diff = self.preprocess_func(state_diff)
        
        x = self.activation(self.conv0(state_diff))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))   # (bs, 128, k_obstacle)
        x = self.activation(self.conv3(x))
        x = self.conv4(x)
        alpha = torch.squeeze(x, dim=1)          # (bs, k_obstacle)
        return alpha

class NNController(nn.Module):

    def __init__(self, n_state, m_control, preprocess_func=None, output_scale=1.0):
        super().__init__()
        self.n_state = n_state
        self.k_obstacle = k_obstacle
        self.m_control = m_control
        self.preprocess_func = preprocess_func

        self.conv0 = nn.Conv1d(n_state, 64, 1)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.fc0 = nn.Linear(128 + m_control + n_state, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, m_control)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()
        self.output_scale = output_scale

    def forward(self, state, obstacle, u_nominal, state_error):
        """
        args:
            state (bs, n_state)
            obstacle (bs, k_obstacle, n_state)
            u_nominal (bs, m_control)
            state_error (bs, n_state)
        returns:
            u (bs, m_control)
        """
        state = torch.unsqueeze(state, 2)    # (bs, n_state, 1)
        # print(state)
        # print(len(state))
        obstacle = obstacle.permute(0, 2, 1) # (bs, n_state, k_obstacle)
        state_diff = state - obstacle

        if self.preprocess_func is not None:
            state_diff = self.preprocess_func(state_diff)
            state_error = self.preprocess_func(state_error)
        
        x = self.activation(self.conv0(state_diff))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))   # (bs, 128, k_obstacle)
        x, _ = torch.max(x, dim=2)              # (bs, 128)
        x = torch.cat([x, u_nominal, state_error], dim=1) # (bs, 128 + m_control)
        x = self.activation(self.fc0(x))
        x = self.activation(self.fc1(x))
        x = self.output_activation(self.fc2(x)) * self.output_scale
        u = x + u_nominal
        return u


class NNController_new(nn.Module):

    def __init__(self, n_state, m_control, preprocess_func=None, output_scale=1.0):
        super().__init__()
        self.n_state = n_state
        # self.k_obstacle = k_obstacle
        self.m_control = m_control
        self.preprocess_func = preprocess_func

        self.conv0 = nn.Conv1d(n_state, 64, 1)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.fc0 = nn.Linear(128 + m_control, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, m_control)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()
        self.output_scale = output_scale

    def forward(self, state, u_nominal):
        """
        args:
            state (bs, n_state)
            obstacle (bs, k_obstacle, n_state)
            u_nominal (bs, m_control)
            state_error (bs, n_state)
        returns:
            u (bs, m_control)
        """
        state = torch.unsqueeze(state, 2)    # (bs, n_state, 1)
        # print(state)
        # print(len(state))
        # obstacle = obstacle.permute(0, 2, 1) # (bs, n_state, k_obstacle)
        # state_diff = state - obstacle

        if self.preprocess_func is not None:
            state_diff = self.preprocess_func(state)
            # state_error = self.preprocess_func(state_error)
        
        x = self.activation(self.conv0(state))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))   # (bs, 128, k_obstacle)
        x, _ = torch.max(x, dim=2)              # (bs, 128)
        x = torch.cat([x, u_nominal], dim=1) # (bs, 128 + m_control)
        x = self.activation(self.fc0(x))
        x = self.activation(self.fc1(x))
        x = self.output_activation(self.fc2(x)) * self.output_scale
        u = x + u_nominal
        return u