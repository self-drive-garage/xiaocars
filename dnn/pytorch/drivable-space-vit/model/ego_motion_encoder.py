import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def get_default_config():
    """Return default configuration parameters for EgoMotionEncoder"""
    return {
        'ego_motion_dim': 9,  # 3 velocity + 3 acceleration + 3 angular velocity
        'embed_dim': 768,
        'dropout': 0.1,
    }

class EgoMotionEncoder(nn.Module):
    """Enhanced encoder for ego motion data with explicit separation of components and early integration support"""
    def __init__(
        self,
        ego_motion_dim=None,
        embed_dim=None,
        dropout=None,
        config=None,
    ):
        super().__init__()
        
        # Get default configuration
        default_config = get_default_config()
        
        # Use provided config if available
        model_config = {}
        if config is not None and 'model' in config:
            model_config = config['model']
        
        # Use provided parameters if given, otherwise use config, fallback to defaults
        self.ego_motion_dim = ego_motion_dim if ego_motion_dim is not None else model_config.get('ego_motion_dim', default_config['ego_motion_dim'])
        self.embed_dim = embed_dim if embed_dim is not None else model_config.get('embed_dim', default_config['embed_dim'])
        dropout_value = dropout if dropout is not None else model_config.get('dropout', default_config['dropout'])
        
        # Component-specific feature dim - each component gets 1/3 of the embedding dimension
        # since we have 3 components (velocity, acceleration, angular velocity)
        feature_dim = self.embed_dim // 3

        # Linear acceleration encoder (x, y)
        self.accel_encoder = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
        )
        
        # Linear velocity encoder (x, y)
        self.velocity_encoder = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
        )
        
        # Angular velocity encoder (roll, yaw)
        self.angular_vel_encoder = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
        )
        
        # Full dynamics encoder for 3 feature groups
        self.dynamics_encoder = nn.Sequential(
            nn.Linear(3 * feature_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(self.embed_dim // 2, self.embed_dim),
            nn.Dropout(dropout_value),
        )
        
    def forward(self, x):
        # x shape: (batch_size, ego_motion_dim)
        B, D = x.shape
        
        # Split into components based on new structure
        velocity = x[:, 0:3]        # x, y, z velocity - shape: (B, 3)
        acceleration = x[:, 3:6]    # x, y, z acceleration - shape: (B, 3)
        angular_vel = x[:, 6:9]     # x, y, z angular velocity - shape: (B, 3)
        
        # Process each component directly - nn.Linear can handle 3D inputs
        # The linear transformation is applied only to the last dimension
        vel_features = self.velocity_encoder(velocity)  # shape: (B, feature_dim)
        accel_features = self.accel_encoder(acceleration)  # shape: (B, feature_dim)
        ang_vel_features = self.angular_vel_encoder(angular_vel)  # shape: (B, feature_dim)
        
        # Concatenate only the real features
        combined = torch.cat([
            accel_features, 
            vel_features, 
            ang_vel_features
        ], dim=-1)  # shape: (B, 3*feature_dim)
        
        # Final projection
        output = self.dynamics_encoder(combined)  # shape: (B, embed_dim)
        
        return output

    def encode_frame(self, x):
        """
        Process a single frame or batch of frames (without sequence dimension)
        
        Args:
            x: Ego motion data with shape (batch_size, ego_motion_dim)
            
        Returns:
            Encoded features with shape (batch_size, embed_dim)
        """
        # Add debugging for input tensor
        logger.debug(f"encode_frame input shape: {x.shape}, dtype: {x.dtype}")
        
        # Process with forward
        features = self.forward(x)  # (B, embed_dim)
        
        return features
    
class MotionGuidedAttention(nn.Module):
    """Attention module that uses ego motion to guide visual attention"""
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Motion feature projection
        self.motion_proj = nn.Linear(embed_dim, embed_dim)
        
        # Visual feature projection
        self.visual_proj_q = nn.Linear(embed_dim, embed_dim)
        self.visual_proj_k = nn.Linear(embed_dim, embed_dim)
        self.visual_proj_v = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, visual_features, motion_features):
        """
        Apply motion-guided attention to visual features
        
        Args:
            visual_features: (B, T, N, E) - visual features
            motion_features: (B, T, E) - motion features
        
        Returns:
            Attention weights (B, T, N, 1)
        """
        B, T, N, E = visual_features.shape
        
        # Project motion features
        motion_proj = self.motion_proj(motion_features)  # (B, T, E)
        motion_proj = motion_proj.unsqueeze(2)  # (B, T, 1, E)
        
        # Project visual features
        q = self.visual_proj_q(visual_features)  # (B, T, N, E)
        k = self.visual_proj_k(visual_features)  # (B, T, N, E)
        v = self.visual_proj_v(visual_features)  # (B, T, N, E)
        
        # Reshape for multi-head attention
        q = q.reshape(B, T, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)  # (B, T, H, N, D)
        k = k.reshape(B, T, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)  # (B, T, H, N, D)
        v = v.reshape(B, T, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)  # (B, T, H, N, D)
        
        # Get motion query
        motion_q = motion_proj.reshape(B, T, 1, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)  # (B, T, H, 1, D)
        
        # Compute attention scores
        attn_scores = torch.matmul(motion_q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, T, H, 1, N)
        
        # Apply softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, T, H, 1, N)
        
        # Reshape to (B, T, N, 1)
        attn_weights = attn_weights.permute(0, 1, 3, 4, 2).mean(dim=-1)  # (B, T, 1, N)
        attn_weights = attn_weights.transpose(-2, -1)  # (B, T, N, 1)
        
        return attn_weights
    
    def apply_attention(self, visual_features, motion_features):
        """
        Apply motion-guided attention to visual features and return the attended features
        
        Args:
            visual_features: (B, T, N, E) - visual features
            motion_features: (B, T, E) - motion features
            
        Returns:
            Attended features (B, T, N, E)
        """
        # Get attention weights
        attn_weights = self.forward(visual_features, motion_features)  # (B, T, N, 1)
        
        # Apply attention with residual connection
        attended_features = visual_features * attn_weights + visual_features
        
        return attended_features

class MotionGuidedFramePredictor(nn.Module):
    """Predicts future frames based on current frame and ego motion"""
    def __init__(self, embed_dim, ego_motion_dim=9):
        super().__init__()
        
        self.motion_transform = nn.Sequential(
            nn.Linear(ego_motion_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.frame_predictor = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim*4
        )
    
    def forward(self, current_features, ego_motion):
        """Predict future features based on current features and ego motion"""
        # Transform ego motion to feature space
        motion_features = self.motion_transform(ego_motion)
        
        # Predict future features with motion guidance
        future_features = self.frame_predictor(
            current_features, 
            motion_features
        )
        
        return future_features
