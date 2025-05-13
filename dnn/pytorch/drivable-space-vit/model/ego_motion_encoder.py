import torch
import torch.nn as nn

def get_default_config():
    """Return default configuration parameters for EgoMotionEncoder"""
    return {
        'ego_motion_dim': 12,  # 3 position + 3 orientation + 2 accel + 2 velocity + 2 angular velocity
        'embed_dim': 768,
        'dropout': 0.1,
    }

class EgoMotionEncoder(nn.Module):
    """Enhanced encoder for ego motion data with explicit separation of components"""
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
        
        # Component-specific feature dim
        feature_dim = self.embed_dim // 6
        
        # Position encoder (x, y, z)
        self.position_encoder = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
        )
        
        # Orientation encoder (roll, pitch, yaw)
        self.orientation_encoder = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
        )
        
        # Linear acceleration encoder (x, y)
        self.accel_encoder = nn.Sequential(
            nn.Linear(2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
        )
        
        # Linear velocity encoder (x, y)
        self.velocity_encoder = nn.Sequential(
            nn.Linear(2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
        )
        
        # Angular velocity encoder (roll, yaw)
        self.angular_vel_encoder = nn.Sequential(
            nn.Linear(2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
        )
        
        # Full dynamics encoder for fallback
        self.dynamics_encoder = nn.Sequential(
            nn.Linear(self.ego_motion_dim, self.embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_value),
        )
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(5 * feature_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(self.embed_dim // 2, self.embed_dim),
            nn.Dropout(dropout_value),
        )
        
        # Legacy encoder for backward compatibility
        self.encoder = nn.Sequential(
            nn.Linear(self.ego_motion_dim, self.embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(self.embed_dim // 4, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(self.embed_dim // 2, self.embed_dim),
            nn.Dropout(dropout_value),
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, ego_motion_dim)
        B, T, D = x.shape
        
        # Use enhanced processing if we have the expected dimensions (12+)
        if D >= 12:
            # Split into components
            position = x[:, :, 0:3]         # x, y, z position
            orientation = x[:, :, 3:6]      # roll, pitch, yaw
            acceleration = x[:, :, 6:8]     # x, y acceleration
            velocity = x[:, :, 8:10]        # x, y velocity
            angular_vel = x[:, :, 10:12]    # roll, yaw angular velocity
            
            # Process separately
            pos_features = self.position_encoder(position)
            ori_features = self.orientation_encoder(orientation)
            accel_features = self.accel_encoder(acceleration)
            vel_features = self.velocity_encoder(velocity)
            ang_vel_features = self.angular_vel_encoder(angular_vel)
            
            # Concatenate features
            combined = torch.cat([
                pos_features, 
                ori_features, 
                accel_features, 
                vel_features, 
                ang_vel_features
            ], dim=-1)
            
            # Final projection
            output = self.projection(combined)
            
            return output  # (batch_size, seq_len, embed_dim)
        # Fallback for original implementation (6+ dimensions)
        elif D >= 6:
            # Split into components
            position = x[:, :, 0:3]  # x, y, z position
            orientation = x[:, :, 3:6]  # roll, pitch, yaw
            
            # Process separately
            pos_features = self.position_encoder(position)
            ori_features = self.orientation_encoder(orientation)
            dyn_features = self.dynamics_encoder(x)
            
            # Concatenate features
            combined = torch.cat([pos_features, ori_features, dyn_features], dim=-1)
            
            # Final projection
            output = self.projection(combined)
            
            return output  # (batch_size, seq_len, embed_dim)
        else:
            # Fallback to legacy encoder
            return self.encoder(x)

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

class MotionGuidedFramePredictor(nn.Module):
    """Predicts future frames based on current frame and ego motion"""
    def __init__(self, embed_dim, ego_motion_dim):
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
