import torch
import torch.nn as nn
import numpy as np

class BCPolicy(nn.Module):
    """
    Behavior Cloning Policy Network
    """
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256], activation='relu'):
        super(BCPolicy, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(self.activation)
            input_dim = hidden_size
        
        # Output layer for actions
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """
        Forward pass through the network
        
        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)
        
        Returns:
            action: Action tensor of shape (batch_size, action_dim) or (action_dim,)
        """
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        action = self.network(state)
        
        if squeeze_output:
            action = action.squeeze(0)
        
        return action
    
    def get_action(self, state, deterministic=True):
        """
        Get action for given state (compatible with SAC interface)
        
        Args:
            state: State as numpy array
            deterministic: Whether to return deterministic action (unused for BC)
        
        Returns:
            action: Action as numpy array
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action = self.forward(state)
        
        return action.cpu().numpy().flatten()
    
    def save(self, filepath):
        """Save model to file"""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath):
        """Load model from file"""
        self.load_state_dict(torch.load(filepath, map_location='cpu'))
