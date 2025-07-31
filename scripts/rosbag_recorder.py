import subprocess
import sys

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Record stereo camera topics to rosbag2')
    parser.add_argument('--output', type=str, default='slam_recording', help='Output bag name')
    parser.add_argument('--topics', nargs='+', default=['/camera/left/image_raw', '/camera/right/image_raw', '/tf', '/tf_static'], help='Topics to record')
    args = parser.parse_args()

    cmd = [
        'ros2', 'bag', 'record', '-o', args.output
    ] + args.topics
    print(f"Recording topics: {args.topics} to bag: {args.output}")
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
