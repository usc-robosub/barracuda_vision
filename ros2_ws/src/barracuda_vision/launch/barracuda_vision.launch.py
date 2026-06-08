from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_path = LaunchConfiguration('model_path')
    model_dir = LaunchConfiguration('model_dir')
    input_topic = LaunchConfiguration('input_topic')

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path', default_value='',
            description='Explicit path to a YOLO .pt model file. '
                        'If empty, model_dir is scanned for a single model.'),
        DeclareLaunchArgument(
            'model_dir', default_value='/models',
            description='Directory scanned for exactly one model file '
                        'when model_path is not set.'),
        DeclareLaunchArgument(
            'input_topic', default_value='yolo_input_image',
            description='Image topic to subscribe to.'),
        Node(
            package='barracuda_vision',
            executable='yolo_image_listener',
            name='yolo_image_listener',
            output='screen',
            parameters=[{
                'model_path': model_path,
                'model_dir': model_dir,
                'input_topic': input_topic,
            }],
        ),
    ])
