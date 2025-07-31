#control interface for SLAM system

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from std_msgs.msg import Bool
import threading
import time
import sys
import select
import termios
import tty


class SLAMControlInterface(Node):
    # interactive CLI for SLAM control
    
    def __init__(self):
        super().__init__('slam_control_interface')
        
        # Service clients
        self.reset_client = self.create_client(Empty, '/slam/reset')
        self.save_map_client = self.create_client(Empty, '/slam/save_map')
        
        # Publishers
        self.control_pub = self.create_publisher(Bool, '/slam/control', 10)
        
        # Terminal settings for keyboard input
        self.old_settings = termios.tcgetattr(sys.stdin)
        
        self.get_logger().info('SLAM Control Interface initialized')
        self.print_help()
    
    def print_help(self):
        """Print available commands"""
        print("\n" + "="*50)
        print("SLAM Control Interface")
        print("="*50)
        print("Commands:")
        print("  r - Reset SLAM system")
        print("  s - Save current map")
        print("  p - Pause/Resume SLAM")
        print("  h - Show this help")
        print("  q - Quit")
        print("="*50)
        print("Press any key to execute command...\n")
    
    def get_char(self):
        # get single char from stdin
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        return ch
    
    def reset_slam(self):
        print("Resetting SLAM system...")
        
        if not self.reset_client.wait_for_service(timeout_sec=2.0):
            print("ERROR: Reset service not available")
            return
        
        request = Empty.Request()
        future = self.reset_client.call_async(request)
        
        timeout = 5.0
        start_time = time.time()
        while not future.done() and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if future.done():
            print("✓ SLAM system reset successfully")
        else:
            print("✗ Failed to reset SLAM system (timeout)")
    
    def save_map(self):
        # save curr map
        print("Saving current map...")
        
        if not self.save_map_client.wait_for_service(timeout_sec=2.0):
            print("ERROR: Save map service not available")
            return
        
        request = Empty.Request()
        future = self.save_map_client.call_async(request)
        
        timeout = 10.0
        start_time = time.time()
        while not future.done() and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if future.done():
            print("Map saved successfully")
        else:
            print("Failed to save map (timeout)")
    
    def toggle_pause(self):
        # toggle SLAM pause and resume
        msg = Bool()
        msg.data = True 
        self.control_pub.publish(msg)
        print("✓ Sent pause/resume command")
    
    def run_interface(self):
        print("SLAM Control Interface running. Press 'h' for help.")
        
        try:
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = self.get_char().lower()
                    
                    if char == 'q':
                        print("Quitting...")
                        break
                    elif char == 'r':
                        self.reset_slam()
                    elif char == 's':
                        self.save_map()
                    elif char == 'p':
                        self.toggle_pause()
                    elif char == 'h':
                        self.print_help()
                    else:
                        print(f"Unknown command: '{char}'. Press 'h' for help.")
                
                rclpy.spin_once(self, timeout_sec=0.0)
                
        except KeyboardInterrupt:
            print("\nReceived interrupt, shutting down...")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


def main():
    rclpy.init()
    
    interface = SLAMControlInterface()
    
    try:
        interface.run_interface()
    except Exception as e:
        interface.get_logger().error(f'Interface error: {str(e)}')
    finally:
        interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
