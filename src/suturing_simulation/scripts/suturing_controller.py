#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose, Twist, WrenchStamped
from std_msgs.msg import Bool, Float64MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import threading
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import os
import pickle

@dataclass
class SuturingState:
    """Complete state representation for suturing task"""
    # Visual features from multiple views
    endoscope_image: np.ndarray
    overhead_image: np.ndarray  
    left_view_image: np.ndarray
    right_view_image: np.ndarray
    
    # Robot kinematics
    joint_positions: np.ndarray  # 12 DOF (2 arms × 6 joints)
    joint_velocities: np.ndarray
    
    # Tool poses
    left_tool_pose: np.ndarray   # 6D pose (position + orientation)
    right_tool_pose: np.ndarray
    
    # Force/torque feedback
    left_tool_wrench: np.ndarray  # 6D wrench
    right_tool_wrench: np.ndarray
    
    # Task-specific features
    needle_pose: np.ndarray      # Estimated needle pose
    tissue_deformation: float    # Estimated tissue deformation
    suture_tension: float        # Suture thread tension
    
    # Temporal context
    timestamp: float

class MultiModalEncoder(nn.Module):
    """Multi-modal encoder for visual and proprioceptive features"""
    
    def __init__(self, visual_dim=512, proprio_dim=64, fusion_dim=256):
        super().__init__()
        
        # Visual encoder (CNN for images)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, visual_dim)
        )
        
        # Proprioceptive encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(42, 128),  # 12 joints + 12 poses + 12 wrenches + 6 task features
            nn.ReLU(),
            nn.Linear(128, proprio_dim)
        )
        
        # Multi-view attention
        self.view_attention = nn.MultiheadAttention(
            embed_dim=visual_dim, num_heads=8, batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + proprio_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, images, proprioception):
        # Process each view
        batch_size = images.shape[0]
        num_views = images.shape[1]
        
        # Encode visual features for each view
        visual_features = []
        for i in range(num_views):
            view_features = self.visual_encoder(images[:, i])
            visual_features.append(view_features)
        
        # Stack and apply attention across views
        visual_stack = torch.stack(visual_features, dim=1)  # [batch, views, features]
        attended_visual, _ = self.view_attention(visual_stack, visual_stack, visual_stack)
        
        # Aggregate across views (mean pooling)
        visual_agg = attended_visual.mean(dim=1)
        
        # Encode proprioceptive features
        proprio_features = self.proprio_encoder(proprioception)
        
        # Fuse modalities
        fused = torch.cat([visual_agg, proprio_features], dim=1)
        return self.fusion(fused)

class SuturingPolicy(nn.Module):
    """Deep learning policy for suturing demonstrations"""
    
    def __init__(self, state_dim=256, action_dim=12, hidden_dim=512):
        super().__init__()
        
        self.encoder = MultiModalEncoder()
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Action prediction heads
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_dim)
        )
        
        # Value estimation (for RL fine-tuning)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, images, proprioception, hidden_state=None):
        # Encode current state
        encoded_state = self.encoder(images, proprioception)
        
        # Add sequence dimension if needed
        if len(encoded_state.shape) == 2:
            encoded_state = encoded_state.unsqueeze(1)
        
        # LSTM forward pass
        lstm_out, new_hidden = self.lstm(encoded_state, hidden_state)
        
        # Predictions
        actions = self.action_head(lstm_out)
        values = self.value_head(lstm_out)
        uncertainties = F.softplus(self.uncertainty_head(lstm_out))
        
        return actions, values, uncertainties, new_hidden

class DeepLfDController(Node):
    """Deep Learning from Demonstrations controller for surgical suturing"""
    
    def __init__(self):
        super().__init__('deep_lfd_controller')
        
        # Initialize components
        self.bridge = CvBridge()
        self.policy = SuturingPolicy()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)
        
        # State buffers
        self.state_buffer = deque(maxlen=100)
        self.current_state = None
        self.lstm_hidden = None
        
        # Demo collection
        self.collecting_demo = False
        self.demo_data = []
        self.demo_count = 0
        
        # Image subscribers
        self.image_subs = {
            'endoscope': self.create_subscription(
                Image, '/endoscope/image', self.endoscope_callback, 10),
            'overhead': self.create_subscription(
                Image, '/overhead/image', self.overhead_callback, 10),
            'left_view': self.create_subscription(
                Image, '/left_view/image', self.left_view_callback, 10),
            'right_view': self.create_subscription(
                Image, '/right_view/image', self.right_view_callback, 10)
        }
        
        # Robot state subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.wrench_sub = self.create_subscription(
            WrenchStamped, '/ft_sensor/raw', self.wrench_callback, 10)
        
        # Demo control subscriber
        self.demo_control_sub = self.create_subscription(
            Bool, '/demo_control', self.demo_control_callback, 10)
        
        # Action publisher
        self.action_pub = self.create_publisher(
            Float64MultiArray, '/davinci/joint_commands', 10)
        
        # Status publisher
        self.status_pub = self.create_publisher(
            Bool, '/controller_status', 10)
        
        # Control parameters
        self.control_frequency = 50  # Hz
        self.action_scale = 0.1
        self.safety_limits = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14] * 2)
        
        # Initialize state
        self.latest_images = {
            'endoscope': None,
            'overhead': None, 
            'left_view': None,
            'right_view': None
        }
        self.latest_joints = None
        self.latest_wrenches = np.zeros(12)  # 2 tools × 6 DOF
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency, self.control_loop)
        
        # Load pretrained model if available
        self.load_model()
        
        self.get_logger().info("Deep-LfD Controller initialized")
    
    def endoscope_callback(self, msg):
        self.latest_images['endoscope'] = self.bridge.imgmsg_to_cv2(msg, "rgb8")
    
    def overhead_callback(self, msg):
        self.latest_images['overhead'] = self.bridge.imgmsg_to_cv2(msg, "rgb8")
    
    def left_view_callback(self, msg):
        self.latest_images['left_view'] = self.bridge.imgmsg_to_cv2(msg, "rgb8")
    
    def right_view_callback(self, msg):
        self.latest_images['right_view'] = self.bridge.imgmsg_to_cv2(msg, "rgb8")
    
    def joint_callback(self, msg):
        self.latest_joints = {
            'positions': np.array(msg.position),
            'velocities': np.array(msg.velocity) if msg.velocity else np.zeros(len(msg.position)),
            'names': msg.name
        }
    
    def wrench_callback(self, msg):
        # Assuming alternating left/right tool measurements
        wrench = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ])
        # Update appropriate tool (simplified)
        self.latest_wrenches[:6] = wrench
    
    def demo_control_callback(self, msg):
        """Control demonstration recording"""
        if msg.data and not self.collecting_demo:
            self.start_demonstration()
        elif not msg.data and self.collecting_demo:
            self.stop_demonstration()
    
    def get_current_state(self) -> Optional[SuturingState]:
        """Extract current state from sensors"""
        if not all(img is not None for img in self.latest_images.values()):
            return None
        if self.latest_joints is None:
            return None
        
        # Ensure we have enough joint data
        if len(self.latest_joints['positions']) < 12:
            # Pad with zeros if needed
            positions = np.zeros(12)
            velocities = np.zeros(12)
            positions[:len(self.latest_joints['positions'])] = self.latest_joints['positions']
            velocities[:len(self.latest_joints['velocities'])] = self.latest_joints['velocities']
        else:
            positions = self.latest_joints['positions'][:12]
            velocities = self.latest_joints['velocities'][:12]
        
        # Estimate tool poses from joint states (simplified)
        left_tool_pose = self.forward_kinematics(positions[:6])
        right_tool_pose = self.forward_kinematics(positions[6:])
        
        # Extract task-specific features (simplified)
        needle_pose = self.estimate_needle_pose()
        tissue_deformation = self.estimate_tissue_deformation()
        suture_tension = self.estimate_suture_tension()
        
        return SuturingState(
            endoscope_image=self.latest_images['endoscope'],
            overhead_image=self.latest_images['overhead'],
            left_view_image=self.latest_images['left_view'],
            right_view_image=self.latest_images['right_view'],
            joint_positions=positions,
            joint_velocities=velocities,
            left_tool_pose=left_tool_pose,
            right_tool_pose=right_tool_pose,
            left_tool_wrench=self.latest_wrenches[:6],
            right_tool_wrench=self.latest_wrenches[6:],
            needle_pose=needle_pose,
            tissue_deformation=tissue_deformation,
            suture_tension=suture_tension,
            timestamp=time.time()
        )
    
    def forward_kinematics(self, joint_angles):
        """Simplified forward kinematics (replace with actual FK)"""
        # Placeholder - implement actual forward kinematics using DH parameters
        # For now, use simplified transformation
        x = np.sum(joint_angles[:3] * 0.1)
        y = np.sum(joint_angles[1:4] * 0.1) 
        z = np.sum(joint_angles[2:5] * 0.1)
        rx, ry, rz = joint_angles[3:6]
        return np.array([x, y, z, rx, ry, rz])
    
    def estimate_needle_pose(self):
        """Estimate needle pose from visual features"""
        # Placeholder - implement computer vision needle tracking
        # Use template matching or deep learning detection
        if self.latest_images['endoscope'] is not None:
            # Simplified needle detection
            gray = cv2.cvtColor(self.latest_images['endoscope'], cv2.COLOR_RGB2GRAY)
            # Find bright objects (needle tip)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Convert pixel coordinates to world coordinates (simplified)
                    x = (cx - 320) * 0.001  # Assuming 640x480 image
                    y = (cy - 240) * 0.001
                    return np.array([x, y, 0.05, 0, 0, 0])
        
        return np.array([0, 0, 0.05, 0, 0, 0])  # Default pose
    
    def estimate_tissue_deformation(self):
        """Estimate tissue deformation"""
        # Use force feedback magnitude as proxy
        force_magnitude = np.linalg.norm(self.latest_wrenches[:3])
        return min(force_magnitude / 10.0, 1.0)  # Normalize to [0,1]
    
    def estimate_suture_tension(self):
        """Estimate suture thread tension"""
        # Combine force feedback and visual cues
        tension = np.linalg.norm(self.latest_wrenches[6:9]) / 5.0
        return min(tension, 1.0)  # Normalize to [0,1]
    
    def state_to_tensors(self, state: SuturingState):
        """Convert state to neural network inputs"""
        # Prepare images
        images = []
        for img_name in ['endoscope_image', 'overhead_image', 'left_view_image', 'right_view_image']:
            img = getattr(state, img_name)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (224, 224))
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            images.append(img)
        
        images = torch.stack(images).unsqueeze(0).to(self.device)
        
        # Prepare proprioception
        proprio = np.concatenate([
            state.joint_positions,
            state.joint_velocities,
            state.left_tool_pose,
            state.right_tool_pose,
            state.left_tool_wrench,
            state.right_tool_wrench,
            state.needle_pose[:3],  # Only position
            [state.tissue_deformation, state.suture_tension, 0]  # Padding
        ])
        
        # Ensure correct dimensionality
        if len(proprio) < 42:
            proprio = np.pad(proprio, (0, 42 - len(proprio)))
        else:
            proprio = proprio[:42]
        
        proprio = torch.from_numpy(proprio).float().unsqueeze(0).to(self.device)
        
        return images, proprio
    
    def control_loop(self):
        """Main control loop"""
        current_state = self.get_current_state()
        if current_state is None:
            return
        
        self.current_state = current_state
        self.state_buffer.append(current_state)
        
        if self.collecting_demo:
            self.record_demonstration_step(current_state)
        else:
            self.execute_policy(current_state)
        
        # Publish status
        status_msg = Bool()
        status_msg.data = True
        self.status_pub.publish(status_msg)
    
    def execute_policy(self, state: SuturingState):
        """Execute learned policy"""
        try:
            images, proprio = self.state_to_tensors(state)
            
            with torch.no_grad():
                actions, values, uncertainties, self.lstm_hidden = self.policy(
                    images, proprio, self.lstm_hidden
                )
            
            # Extract action
            action = actions.squeeze().cpu().numpy()
            uncertainty = uncertainties.squeeze().cpu().numpy()
            
            # Apply safety checks
            action = np.clip(action, -self.action_scale, self.action_scale)
            
            # High uncertainty -> reduce action magnitude
            uncertainty_threshold = 0.5
            uncertainty_mask = uncertainty > uncertainty_threshold
            action[uncertainty_mask] *= 0.3
            
            # Additional safety: limit rapid changes
            if hasattr(self, 'last_action'):
                action_diff = np.abs(action - self.last_action)
                large_change_mask = action_diff > 0.05
                action[large_change_mask] = self.last_action[large_change_mask] + \
                                          0.05 * np.sign(action[large_change_mask] - self.last_action[large_change_mask])
            
            self.last_action = action.copy()
            
            # Publish action
            self.publish_action(action)
            
        except Exception as e:
            self.get_logger().error(f"Policy execution error: {e}")
            # Publish zero action as safety fallback
            self.publish_action(np.zeros(12))
    
    def publish_action(self, action):
        """Publish joint commands"""
        msg = Float64MultiArray()
        msg.data = action.tolist()
        self.action_pub.publish(msg)
    
    def start_demonstration(self):
        """Start collecting demonstration"""
        self.collecting_demo = True
        self.demo_data = []
        self.demo_count += 1
        self.lstm_hidden = None  # Reset LSTM state
        self.get_logger().info(f"Started demonstration {self.demo_count}")
    
    def stop_demonstration(self):
        """Stop collecting demonstration and save"""
        if not self.collecting_demo:
            return
        
        self.collecting_demo = False
        self.save_demonstration()
        self.get_logger().info(f"Saved demonstration {self.demo_count} with {len(self.demo_data)} steps")
    
    def record_demonstration_step(self, state: SuturingState):
        """Record demonstration step"""
        # Also record the current action being executed (from human teleoperation)
        step_data = {
            'state': state,
            'timestamp': state.timestamp,
            'joint_positions': state.joint_positions.copy(),
            'joint_velocities': state.joint_velocities.copy()
        }
        self.demo_data.append(step_data)
    
    def save_demonstration(self):
        """Save demonstration to file"""
        if not self.demo_data:
            self.get_logger().warn("No demonstration data to save")
            return
        
        # Create demonstrations directory if it doesn't exist
        demo_dir = os.path.expanduser("~/suturing_ws/demonstrations")
        os.makedirs(demo_dir, exist_ok=True)
        
        filename = os.path.join(demo_dir, f"demo_{self.demo_count:04d}.pkl")
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.demo_data, f)
            self.get_logger().info(f"Demonstration saved to {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save demonstration: {e}")
    
    def load_model(self):
        """Load pretrained model"""
        model_path = os.path.expanduser("~/suturing_ws/models/suturing_policy.pth")
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.policy.load_state_dict(checkpoint['model_state_dict'])
                self.get_logger().info(f"Loaded pretrained model from {model_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load model: {e}")
        else:
            self.get_logger().info("No pretrained model found, using random initialization")
    
    def save_model(self):
        """Save current model"""
        model_dir = os.path.expanduser("~/suturing_ws/models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "suturing_policy.pth")
        try:
            torch.save({
                'model_state_dict': self.policy.state_dict(),
                'demo_count': self.demo_count
            }, model_path)
            self.get_logger().info(f"Model saved to {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save model: {e}")
    
    def shutdown(self):
        """Clean shutdown"""
        if self.collecting_demo:
            self.stop_demonstration()
        self.save_model()
        self.get_logger().info("Controller shutdown complete")

def main(args=None):
    rclpy.init(args=args)
    
    controller = DeepLfDController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("Keyboard interrupt received")
    finally:
        controller.shutdown()
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
