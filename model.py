import torch
import torch.nn as nn
import torch.nn.functional as F

class TreeCNN(nn.Module):
    """
    Tree Convolutional Neural Network as described in the Bao paper.
    Processes tree nodes bottom-up to predict query execution latency.
    """
    def __init__(self, in_channels, out_channels=128, max_children=2):
        super(TreeCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_children = max_children
        
        # Tree Convolution layer:
        # It takes the concatenated features of the current node and its padded children.
        # Dimension = in_channels (node) + max_children * out_channels (children embeddings)
        conv_input_dim = in_channels + (max_children * out_channels)
        self.conv = nn.Linear(conv_input_dim, out_channels)
        
        # Fully connected layers to map the root embedding to a single scalar (latency)
        # Dropout is included to support Monte Carlo Dropout for Thompson Sampling.
        self.fc = nn.Sequential(
            nn.Linear(out_channels, 64),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def _forward_node(self, node):
        """
        Recursively process the node and its children bottom-up.
        """
        # Extract current node's base features
        node_features = torch.tensor(node.get_feature_vector(), dtype=torch.float32)
        
        # Process children
        child_embeddings = []
        for child in node.children:
            child_emb = self._forward_node(child)
            child_embeddings.append(child_emb)
            
        # Pad or truncate children to the max_children arity
        # Default zero-embedding for missing children
        zero_pad = torch.zeros(self.out_channels, dtype=torch.float32)
        
        padded_children = []
        for i in range(self.max_children):
            if i < len(child_embeddings):
                padded_children.append(child_embeddings[i])
            else:
                padded_children.append(zero_pad)
                
        # Concatenate current node features + all padded children embeddings
        combined_features = torch.cat([node_features] + padded_children)
        
        # Apply Tree Convolution and activation
        node_embedding = F.relu(self.conv(combined_features))
        return node_embedding

    def forward(self, root_node):
        """
        Forward pass starting from the root of the parsed query plan tree.
        """
        # Propagate bottom-up to get the root embedding
        root_embedding = self._forward_node(root_node)
        
        # Pass root embedding through Multi-Layer Perceptron to predict latency
        predicted_latency = self.fc(root_embedding)
        return predicted_latency
