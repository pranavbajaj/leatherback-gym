import torch
import math

# Option 1: Beta Distribution - Natural bell curve centered at 0.5
def spawn_position_beta(left_pts, right_pts, num_envs, device, beta_param=3.0):
    """
    Uses Beta distribution to favor center positions.
    beta_param: Higher values = stronger center bias (3-5 recommended)
    """
    # Beta distribution with equal alpha and beta parameters creates symmetric bell curve
    alpha = torch.distributions.Beta(beta_param, beta_param).sample((num_envs,)).to(device)
    spawn_positions = left_pts * (1 - alpha[:, None]) + right_pts * alpha[:, None]
    return spawn_positions, alpha


# Option 2: Gaussian/Normal Distribution - Classic bell curve
def spawn_position_gaussian(left_pts, right_pts, num_envs, device, std=0.15):
    """
    Uses Gaussian distribution centered at 0.5 (center of track).
    std: Standard deviation (0.1-0.2 recommended)
    """
    # Sample from normal distribution centered at 0.5
    alpha = torch.normal(mean=0.5, std=std, size=(num_envs,), device=device)
    # Clamp to valid range [0, 1]
    alpha = torch.clamp(alpha, 0.0, 1.0)
    spawn_positions = left_pts * (1 - alpha[:, None]) + right_pts * alpha[:, None]
    return spawn_positions, alpha


# Option 3: Truncated Gaussian with threshold
def spawn_position_truncated_gaussian(left_pts, right_pts, num_envs, device, std=0.15, threshold=0.1):
    """
    Gaussian distribution with minimum distance from edges.
    threshold: Minimum distance from track edges (0-0.5)
    """
    # Sample from normal distribution
    alpha = torch.normal(mean=0.5, std=std, size=(num_envs,), device=device)
    # Clamp to range [threshold, 1-threshold]
    alpha = torch.clamp(alpha, threshold, 1.0 - threshold)
    spawn_positions = left_pts * (1 - alpha[:, None]) + right_pts * alpha[:, None]
    return spawn_positions, alpha


# Option 4: Triangular Distribution - Linear peak at center
def spawn_position_triangular(left_pts, right_pts, num_envs, device, threshold=0.1):
    """
    Triangular distribution peaked at center.
    Simple and efficient, creates linear bias toward center.
    """
    # Generate two uniform random numbers
    u1 = torch.rand(num_envs, device=device)
    u2 = torch.rand(num_envs, device=device)
    
    # Average them to get triangular distribution centered at 0.5
    alpha = (u1 + u2) / 2.0
    
    # Apply threshold if needed
    alpha = alpha * (1 - 2*threshold) + threshold
    spawn_positions = left_pts * (1 - alpha[:, None]) + right_pts * alpha[:, None]
    return spawn_positions, alpha


# Option 5: Power Function - Adjustable center bias
def spawn_position_power(left_pts, right_pts, num_envs, device, power=2.0, threshold=0.1):
    """
    Uses power function to push values toward center.
    power: Higher values = stronger center bias
    """
    # Generate uniform random values
    u = torch.rand(num_envs, device=device)
    
    # Apply power function to bias toward 0.5
    # When u < 0.5: push up toward 0.5
    # When u > 0.5: push down toward 0.5
    alpha = torch.where(
        u < 0.5,
        0.5 * (2 * u) ** (1/power),
        1 - 0.5 * (2 * (1 - u)) ** (1/power)
    )
    
    # Apply threshold
    alpha = alpha * (1 - 2*threshold) + threshold
    spawn_positions = left_pts * (1 - alpha[:, None]) + right_pts * alpha[:, None]
    return spawn_positions, alpha


# Option 6: Mixed Distribution - Most realistic
def spawn_position_mixed(left_pts, right_pts, num_envs, device, 
                         center_weight=0.7, std=0.12, threshold=0.05):
    """
    Mix of center-biased and uniform distribution for variety.
    center_weight: Probability of spawning near center (0-1)
    """
    # Decide which robots spawn near center vs randomly
    use_center = torch.rand(num_envs, device=device) < center_weight
    
    # Generate center-biased positions
    center_alpha = torch.normal(mean=0.5, std=std, size=(num_envs,), device=device)
    center_alpha = torch.clamp(center_alpha, threshold, 1.0 - threshold)
    
    # Generate uniform positions
    uniform_alpha = torch.rand(num_envs, device=device) * (1 - 2*threshold) + threshold
    
    # Mix the two distributions
    alpha = torch.where(use_center, center_alpha, uniform_alpha)
    spawn_positions = left_pts * (1 - alpha[:, None]) + right_pts * alpha[:, None]
    return spawn_positions, alpha


# RECOMMENDED: Clean implementation for SpawnOnTrackEvent
def spawn_position_recommended(left_pts, right_pts, num_envs, device, 
                               method='gaussian', std=0.15, threshold=0.1):
    """
    Recommended implementation with configurable method.
    
    Args:
        method: 'gaussian', 'beta', 'triangular', or 'uniform'
        std: Standard deviation for gaussian (0.1-0.2)
        threshold: Minimum distance from edges (0-0.3)
    """
    if method == 'gaussian':
        # Most natural center bias
        alpha = torch.normal(mean=0.5, std=std, size=(num_envs,), device=device)
        alpha = torch.clamp(alpha, threshold, 1.0 - threshold)
    
    elif method == 'beta':
        # Smooth bell curve
        beta_param = 3.0
        alpha = torch.distributions.Beta(beta_param, beta_param).sample((num_envs,)).to(device)
        alpha = alpha * (1 - 2*threshold) + threshold
    
    elif method == 'triangular':
        # Simple and efficient
        u1 = torch.rand(num_envs, device=device)
        u2 = torch.rand(num_envs, device=device)
        alpha = (u1 + u2) / 2.0
        alpha = alpha * (1 - 2*threshold) + threshold
    
    else:  # uniform
        # Original uniform distribution
        alpha = torch.rand(num_envs, device=device) * (1 - 2*threshold) + threshold
    
    spawn_positions = left_pts * (1 - alpha[:, None]) + right_pts * alpha[:, None]
    return spawn_positions, alpha


# Example usage in SpawnOnTrackEvent:
"""
# Replace the original lines:
# alpha = torch.rand(num_envs, device=device) * (1 - 2*self.threshold) + self.threshold
# spawn_positions = left_pts * (1 - alpha[:, None]) + right_pts * alpha[:, None]

# With one of these options:

# Option 1: Gaussian (RECOMMENDED - most natural)
alpha = torch.normal(mean=0.5, std=0.15, size=(num_envs,), device=device)
alpha = torch.clamp(alpha, self.threshold, 1.0 - self.threshold)
spawn_positions = left_pts * (1 - alpha[:, None]) + right_pts * alpha[:, None]

# Option 2: Beta distribution (smooth bell curve)
alpha = torch.distributions.Beta(3.0, 3.0).sample((num_envs,)).to(device)
alpha = alpha * (1 - 2*self.threshold) + self.threshold
spawn_positions = left_pts * (1 - alpha[:, None]) + right_pts * alpha[:, None]

# Option 3: Triangular (simple, efficient)
u1 = torch.rand(num_envs, device=device)
u2 = torch.rand(num_envs, device=device)
alpha = (u1 + u2) / 2.0
alpha = alpha * (1 - 2*self.threshold) + self.threshold
spawn_positions = left_pts * (1 - alpha[:, None]) + right_pts * alpha[:, None]
"""