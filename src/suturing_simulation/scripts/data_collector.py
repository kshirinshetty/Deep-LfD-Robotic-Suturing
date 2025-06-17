#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose, Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import json
from datetime import datetime
import threading

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Data storage
        self.data_dir = os.path.expanduser("~/suturing_dataset")
        self.create_directories()
        
        # Current episode data
        self.current_episode = 0
        self.episode_data = {
            'images': {},
            'joint_states': [],
            'poses': [],
            'actions': [],
            'timestamps': []
        }
        
        # Subscribers for all 6 cameras
        self.camera_subs = {}
        self.latest_images = {}
        
        for i in range(1, 7):
            topic = f'/camera{i}/image'
            self.camera_subs[f'camera{i}'] = self.create_subscription(
                Image, topic, 
                lambda msg, cam=f'camera{i}': self.image_callback(msg, cam),
                10
            )
            self.latest_images[f'camera{i}'] = None
        
        # Joint state subscriber
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        
        # Data recording flag
        self.recording = False
        self.frame_count = 0
        
        # Timer for data collection
        self.timer = self.create_timer(0.1, self.collect_data)  # 10 Hz
        
        self.get_logger().info("Data Collector Node Started")
        self.get_logger().info(f"Data will be saved to: {self.data_dir}")
    
    def create_directories(self):
        """Create necessary directories for data storage"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "episodes"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "images"), exist_ok=True)
        
    def image_callback(self, msg, camera_name):
        """Callback for camera images"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_images[camera_name] = cv_image
        except Exception as e:
            self.get_logger().error(f"Error converting image from {camera_name}: {e}")
    
    def joint_state_callback(self, msg):
        """Callback for joint states"""
        if self.recording:
            joint_data = {
                'names': msg.name,
                'positions': list(msg.position),
                'velocities': list(msg.velocity) if msg.velocity else [],
                'efforts': list(msg.effort) if msg.effort else [],
                'timestamp': self.get_clock().now().to_msg()
            }
            self.episode_data['joint_states'].append(joint_data)
    
    def start_recording(self):
        """Start recording a new episode"""
        self.recording = True
        self.frame_count = 0
        self.current_episode += 1
        
        # Reset episode data
        self.episode_data = {
            'images': {},
            'joint_states': [],
            'poses': [],
            'actions': [],
            'timestamps': []
        }
        
        self.get_logger().info(f"Started recording episode {self.current_episode}")
    
    def stop_recording(self):
        """Stop recording and save data"""
        if not self.recording:
            return
        
        self.recording = False
        self.save_episode_data()
        self.get_logger().info(f"Stopped recording episode {self.current_episode}")
    
    def collect_data(self):
        """Main data collection function"""
        if not self.recording:
            return
        
        # Check if we have images from all cameras
        if not all(img is not None for img in self.latest_images.values()):
            return
        
        timestamp = self.get_clock().now()
        
        # Save images for this frame
        frame_images = {}
        for cam_name, image in self.latest_images.items():
            if image is not None:
                # Create episode image directory
                episode_img_dir = os.path.join(
                    self.data_dir, "images", f"episode_{self.current_episode:04d}"
                )
                os.makedirs(episode_img_dir, exist_ok=True)
                
                # Save image
                img_filename = f"{cam_name}_frame_{self.frame_count:06d}.png"
                img_path = os.path.join(episode_img_dir, img_filename)
                cv2.imwrite(img_path, image)
                
                frame_images[cam_name] = img_filename
        
        # Store frame data
        frame_data = {
            'frame_id': self.frame_count,
            'timestamp': timestamp.to_msg(),
            'images': frame_images
        }
        
        if self.frame_count not in self.episode_data['images']:
            self.episode_data['images'][self.frame_count] = frame_data
        
        self.frame_count += 1
    
    def save_episode_data(self):
        """Save episode data to JSON file"""
        episode_file = os.path.join(
            self.data_dir, "episodes", f"episode_{self.current_episode:04d}.json"
        )
        
        # Add metadata
        metadata = {
            'episode_id': self.current_episode,
            'total_frames': self.frame_count,
            'recording_start': datetime.now().isoformat(),
            'cameras': list(self.latest_images.keys()),
            'frame_rate': 10  # Hz
        }
        
        episode_data_with_metadata = {
            'metadata': metadata,
            'data': self.episode_data
        }
        
        with open(episode_file, 'w') as f:
            json.dump(episode_data_with_metadata, f, indent=2, default=str)
        
        self.get_logger().info(f"Saved episode data to {episode_file}")

def main(args=None):
    rclpy.init(args=args)
    
    data_collector = DataCollector()
    
    try:
        rclpy.spin(data_collector)
    except KeyboardInterrupt:
        data_collector.get_logger().info("Shutting down data collector...")
        if data_collector.recording:
            data_collector.stop_recording()
    finally:
        data_collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
