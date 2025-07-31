import rclpy
from rclpy.node import Node
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import json
from typing import Dict, List, Any
import argparse

class BagAnalyzer(Node):
    # analyze rosbag files
    
    def __init__(self, bag_path: str, output_dir: str):
        super().__init__('bag_analyzer')
        
        self.bag_path = bag_path
        self.output_dir = output_dir
        self.bridge = CvBridge()
        
        os.makedirs(output_dir, exist_ok=True)
        s
        self.analysis = {
            'bag_info': {},
            'image_quality': {},
            'stereo_quality': {},
            'temporal_analysis': {},
            'recommendations': []
        }
    
    def analyze_bag(self):
        # the main analysis function
        try:
            storage_options = StorageOptions(uri=self.bag_path, storage_id='sqlite3')
            converter_options = ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr'
            )
            
            reader = SequentialReader()
            reader.open(storage_options, converter_options)
            
            self.analyze_bag_metadata(reader)
            self.analyze_messages(reader)
            
            reader.close()

            self.generate_recommendations()
            self.save_analysis()

            self.generate_plots()
            
            self.get_logger().info('Bag analysis completed')
            
        except Exception as e:
            self.get_logger().error(f'Analysis failed: {str(e)}')
    
    def analyze_bag_metadata(self, reader):
        topic_types = reader.get_all_topics_and_types()
        
        self.analysis['bag_info'] = {
            'topics': [{'name': t.name, 'type': t.type} for t in topic_types],
            'total_topics': len(topic_types),
            'stereo_topics': [t.name for t in topic_types if 'image' in t.name]
        }
        
        self.get_logger().info(f'Found {len(topic_types)} topics')
    
    def analyze_messages(self, reader):
        left_images = []
        right_images = []
        timestamps = {'left': [], 'right': []}
        
        message_count = 0
        
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            message_count += 1
            
            if topic == '/camera/left/image_raw':
                msg = self.deserialize_image(data)
                if msg:
                    img = self.bridge.imgmsg_to_cv2(msg, 'mono8')
                    left_images.append(img)
                    timestamps['left'].append(timestamp)
                    
            elif topic == '/camera/right/image_raw':
                msg = self.deserialize_image(data)
                if msg:
                    img = self.bridge.imgmsg_to_cv2(msg, 'mono8')
                    right_images.append(img)
                    timestamps['right'].append(timestamp)
            
            if message_count % 1000 == 0:
                self.get_logger().info(f'Processed {message_count} messages...')
        
        self.analyze_image_quality(left_images, right_images)
        
        self.analyze_temporal_properties(timestamps)
        
        self.analyze_stereo_quality(left_images, right_images)
    
    def deserialize_image(self, data):
        try:
            from rclpy.serialization import deserialize_message
            return deserialize_message(data, Image)
        except:
            return None
    
    def analyze_image_quality(self, left_images: List[np.ndarray], right_images: List[np.ndarray]):
        def compute_image_stats(images, name):
            if not images:
                return {}
            
            sample_size = min(100, len(images))
            sampled = images[::len(images)//sample_size] if len(images) > sample_size else images
            
            stats = {
                'total_frames': len(images),
                'analyzed_frames': len(sampled),
                'mean_intensity': [],
                'std_intensity': [],
                'sharpness': [],
                'contrast': []
            }
            
            for img in sampled:
                stats['mean_intensity'].append(float(np.mean(img)))
                stats['std_intensity'].append(float(np.std(img)))
                
                # sharpness
                laplacian = cv2.Laplacian(img, cv2.CV_64F)
                stats['sharpness'].append(float(laplacian.var()))
                
                # contrast
                rms_contrast = np.sqrt(np.mean((img - np.mean(img)) ** 2))
                stats['contrast'].append(float(rms_contrast))
            
            for metric in ['mean_intensity', 'std_intensity', 'sharpness', 'contrast']:
                values = stats[metric]
                stats[f'{metric}_mean'] = float(np.mean(values))
                stats[f'{metric}_std'] = float(np.std(values))
                stats[f'{metric}_min'] = float(np.min(values))
                stats[f'{metric}_max'] = float(np.max(values))
            
            return stats
        
        self.analysis['image_quality'] = {
            'left': compute_image_stats(left_images, 'left'),
            'right': compute_image_stats(right_images, 'right')
        }
    
    def analyze_temporal_properties(self, timestamps: Dict[str, List[int]]):
        def analyze_timing(ts_list, name):
            if len(ts_list) < 2:
                return {}
            
            ts_sec = np.array(ts_list) / 1e9
            
            intervals = np.diff(ts_sec)
            
            return {
                'total_frames': len(ts_list),
                'duration_seconds': float(ts_sec[-1] - ts_sec[0]),
                'average_fps': float(len(ts_list) / (ts_sec[-1] - ts_sec[0])) if len(ts_list) > 1 else 0,
                'frame_interval_mean': float(np.mean(intervals)),
                'frame_interval_std': float(np.std(intervals)),
                'frame_interval_min': float(np.min(intervals)),
                'frame_interval_max': float(np.max(intervals)),
                'timing_consistency': float(1.0 - (np.std(intervals) / np.mean(intervals))) if np.mean(intervals) > 0 else 0
            }
        
        self.analysis['temporal_analysis'] = {
            'left': analyze_timing(timestamps['left'], 'left'),
            'right': analyze_timing(timestamps['right'], 'right')
        }
        
        if timestamps['left'] and timestamps['right']:
            left_ts = np.array(timestamps['left']) / 1e9
            right_ts = np.array(timestamps['right']) / 1e9
            
            sync_errors = []
            for l_ts in left_ts[:min(100, len(left_ts))]:  # Sample for efficiency
                closest_r_idx = np.argmin(np.abs(right_ts - l_ts))
                sync_error = abs(l_ts - right_ts[closest_r_idx])
                sync_errors.append(sync_error)
            
            self.analysis['temporal_analysis']['synchronization'] = {
                'mean_sync_error': float(np.mean(sync_errors)),
                'max_sync_error': float(np.max(sync_errors)),
                'sync_quality': 'good' if np.mean(sync_errors) < 0.033 else 'poor'  # 33ms threshold
            }
    
    def analyze_stereo_quality(self, left_images: List[np.ndarray], right_images: List[np.ndarray]):
        if not left_images or not right_images:
            return
        
        sample_size = min(20, len(left_images), len(right_images))
        
        disparity_stats = []
        feature_matches = []
        
        for i in range(0, min(len(left_images), len(right_images)), max(1, len(left_images)//sample_size)):
            left_img = left_images[i]
            right_img = right_images[i] if i < len(right_images) else right_images[-1]
            
            stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
            disparity = stereo.compute(left_img, right_img)
            
            valid_disparity = disparity[disparity > 0]
            if len(valid_disparity) > 0:
                disparity_stats.append({
                    'valid_pixel_ratio': len(valid_disparity) / disparity.size,
                    'mean_disparity': float(np.mean(valid_disparity)),
                    'std_disparity': float(np.std(valid_disparity))
                })
            
            orb = cv2.ORB_create(nfeatures=500)
            kp1, desc1 = orb.detectAndCompute(left_img, None)
            kp2, desc2 = orb.detectAndCompute(right_img, None)
            
            if desc1 is not None and desc2 is not None:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(desc1, desc2)
                feature_matches.append(len(matches))
        
        self.analysis['stereo_quality'] = {
            'disparity_stats': {
                'mean_valid_pixel_ratio': float(np.mean([s['valid_pixel_ratio'] for s in disparity_stats])) if disparity_stats else 0,
                'mean_disparity': float(np.mean([s['mean_disparity'] for s in disparity_stats])) if disparity_stats else 0,
                'disparity_consistency': float(1.0 - np.std([s['std_disparity'] for s in disparity_stats])) if disparity_stats else 0
            },
            'feature_matching': {
                'mean_matches': float(np.mean(feature_matches)) if feature_matches else 0,
                'min_matches': float(np.min(feature_matches)) if feature_matches else 0,
                'max_matches': float(np.max(feature_matches)) if feature_matches else 0
            }
        }
    
    def generate_recommendations(self):
        recommendations = []
        
        left_fps = self.analysis['temporal_analysis'].get('left', {}).get('average_fps', 0)
        right_fps = self.analysis['temporal_analysis'].get('right', {}).get('average_fps', 0)
        
        if left_fps < 10 or right_fps < 10:
            recommendations.append("Low frame rate detected - consider increasing camera FPS")
        
        sync_info = self.analysis['temporal_analysis'].get('synchronization', {})
        if sync_info.get('sync_quality') == 'poor':
            recommendations.append("Poor stereo synchronization - check camera timing configuration")
        
        left_sharpness = self.analysis['image_quality'].get('left', {}).get('sharpness_mean', 0)
        if left_sharpness < 100:
            recommendations.append("Low image sharpness - check camera focus and lighting")
        
        valid_pixels = self.analysis['stereo_quality'].get('disparity_stats', {}).get('mean_valid_pixel_ratio', 0)
        if valid_pixels < 0.3:
            recommendations.append("Low disparity coverage - check stereo calibration and scene texture")
        
        if not recommendations:
            recommendations.append("Data quality appears good for SLAM processing")
        
        self.analysis['recommendations'] = recommendations
    
    def save_analysis(self):
        json_path = os.path.join(self.output_dir, 'bag_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(self.analysis, f, indent=2)
        
        yaml_path = os.path.join(self.output_dir, 'bag_analysis_summary.yaml')
        summary = {
            'bag_quality_score': self.compute_quality_score(),
            'key_metrics': {
                'left_fps': self.analysis['temporal_analysis'].get('left', {}).get('average_fps', 0),
                'right_fps': self.analysis['temporal_analysis'].get('right', {}).get('average_fps', 0),
                'sync_quality': self.analysis['temporal_analysis'].get('synchronization', {}).get('sync_quality', 'unknown'),
                'mean_feature_matches': self.analysis['stereo_quality'].get('feature_matching', {}).get('mean_matches', 0)
            },
            'recommendations': self.analysis['recommendations']
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        self.get_logger().info(f'Analysis saved to {json_path} and {yaml_path}')
    
    def compute_quality_score(self) -> float:
        score = 100.0
        
        left_fps = self.analysis['temporal_analysis'].get('left', {}).get('average_fps', 0)
        if left_fps < 15:
            score -= 20
        elif left_fps < 25:
            score -= 10
        
        sync_quality = self.analysis['temporal_analysis'].get('synchronization', {}).get('sync_quality', 'unknown')
        if sync_quality == 'poor':
            score -= 25
        
        sharpness = self.analysis['image_quality'].get('left', {}).get('sharpness_mean', 1000)
        if sharpness < 50:
            score -= 20
        elif sharpness < 100:
            score -= 10
        
        valid_pixels = self.analysis['stereo_quality'].get('disparity_stats', {}).get('mean_valid_pixel_ratio', 1.0)
        if valid_pixels < 0.2:
            score -= 20
        elif valid_pixels < 0.4:
            score -= 10
        
        return max(0.0, score)
    
    def generate_plots(self):
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Bag Analysis Results')
            
            left_fps = self.analysis['temporal_analysis'].get('left', {}).get('average_fps', 0)
            right_fps = self.analysis['temporal_analysis'].get('right', {}).get('average_fps', 0)
            
            axes[0, 0].bar(['Left Camera', 'Right Camera'], [left_fps, right_fps])
            axes[0, 0].set_title('Average Frame Rates')
            axes[0, 0].set_ylabel('FPS')
            
            left_sharpness = self.analysis['image_quality'].get('left', {}).get('sharpness_mean', 0)
            right_sharpness = self.analysis['image_quality'].get('right', {}).get('sharpness_mean', 0)
            
            axes[0, 1].bar(['Left Camera', 'Right Camera'], [left_sharpness, right_sharpness])
            axes[0, 1].set_title('Image Sharpness')
            axes[0, 1].set_ylabel('Laplacian Variance')
            
            valid_ratio = self.analysis['stereo_quality'].get('disparity_stats', {}).get('mean_valid_pixel_ratio', 0)
            mean_matches = self.analysis['stereo_quality'].get('feature_matching', {}).get('mean_matches', 0)
            
            axes[1, 0].bar(['Valid Disparity Ratio', 'Feature Matches/100'], [valid_ratio, mean_matches/100])
            axes[1, 0].set_title('Stereo Quality Metrics')
            axes[1, 0].set_ylabel('Normalized Values')
            
            quality_score = self.compute_quality_score()
            colors = ['green' if quality_score > 80 else 'orange' if quality_score > 60 else 'red']
            
            axes[1, 1].bar(['Overall Quality'], [quality_score], color=colors)
            axes[1, 1].set_title('Overall Quality Score')
            axes[1, 1].set_ylabel('Score (0-100)')
            axes[1, 1].set_ylim(0, 100)
            
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, 'analysis_plots.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.get_logger().info(f'Plots saved to {plot_path}')
            
        except ImportError:
            self.get_logger().warn('Matplotlib not available, skipping plots')
        except Exception as e:
            self.get_logger().error(f'Failed to generate plots: {str(e)}')


def main():
    # main entry
    parser = argparse.ArgumentParser(description='Analyze rosbag files for SLAM quality')
    parser.add_argument('bag_path', help='Path to the rosbag directory')
    parser.add_argument('--output', '-o', default='./analysis_output', help='Output directory')
    
    args = parser.parse_args()
    
    rclpy.init()
    
    analyzer = BagAnalyzer(args.bag_path, args.output)
    
    try:
        analyzer.analyze_bag()
    except Exception as e:
        analyzer.get_logger().error(f'Analysis failed: {str(e)}')
    finally:
        analyzer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()default_plugins/TF
      Enabled: true
      Filter (blacklist): ""
      Filter (whitelist): ""
      Frame Timeout: 15
      Frames:
        All Enabled: true
        map:
          Value: true
        camera_link:
          Value: true
        base_link:
          Value: true
      Marker Alpha: 1
      Marker Scale: 1
      Name: TF
      Show Arrows: true
      Show Axes: true
      Show Names: true
      Tree:
        map:
          camera_link:
            {}
      Update Interval: 0
      Value: true
    - Class: rviz_default_plugins/Image
      Enabled: true
      Image Topic: /camera/left/image_raw
      Max Value: 1
      Median window: 5
      Min Value: 0
      Name: Left Camera
      Normalize Range: true
      Queue Size: 2
      Transport Hint: raw
      Unreliable: false
      Value: true
    - Class: rviz_default_plugins/Image
      Enabled: true
      Image Topic: /camera/right/image_raw
      Max Value: 1
      Median window: 5
      Min Value: 0
      Name: Right Camera
      Normalize Range: true
      Queue Size: 2
      Transport Hint: raw
      Unreliable: false
      Value: true
    - Alpha: 1
      Buffer Length: 1
      Class: rviz_default_plugins/Path
      Color: 25; 255; 0
      Enabled: true
      Head Diameter: 0.30000001192092896
      Head Length: 0.20000000298023224
      Length: 0.30000001192092896
      Line Style: Lines
      Line Width: 0.029999999329447746
      Name: SLAM Trajectory
      Offset:
        X: 0
        Y: 0
        Z: 0
      Pose Color: 255; 85; 255
      Pose Style: None
      Queue Size: 10
      Radius: 0.029999999329447746
      Shaft Diameter: 0.10000000149011612
      Shaft Length: 0.10000000149011612
      Topic: /slam/trajectory
      Unreliable: false
      Value: true
    - Alpha: 1
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds:
        Max Value: 10
        Min Value: -10
        Value: true
      Axis: Z
      Channel Name: intensity
      Class: rviz_default_plugins/PointCloud2
      Color: 255; 255; 255
      Color Transformer: RGB8
      Decay Time: 0
      Enabled: true
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Min Color: 0; 0; 0
      Name: Point Cloud Map
      Position Transformer: XYZ
      Queue Size: 10
      Selectable: true
      Size (Pixels): 3
      Size (m): 0.009999999776482582
      Style: Flat Squares
      Topic: /slam/map
      Unreliable: false
      Use Fixed Frame: true
      Use rainbow: true
      Value: true
    - Alpha: 1
      Axes Length: 1
      Axes Radius: 0.10000000149011612
      Class: rviz_default_plugins/PoseStamped
      Color: 255; 25; 0
      Enabled: true
      Head Length: 0.30000001192092896
      Head Radius: 0.10000000149011612
      Name: Current Pose
      Queue Size: 10
      Shaft Length: 1
      Shaft Radius: 0.05000000074505806
      Shape: Arrow
      Topic: /slam/pose
      Unreliable: false
      Value: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Default Light: true
    Fixed Frame: map
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
    - Class: rviz_default_plugins/SetInitialPose
      Theta std deviation: 0.2617993950843811
      Topic: /initialpose
      X std deviation: 0.5
      Y std deviation: 0.5
    - Class: rviz_default_plugins/SetGoal
      Topic: /goal_pose
    - Class: rviz_