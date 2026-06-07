import os
from glob import glob

from setuptools import setup

package_name = 'barracuda_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'images'),
            glob('images/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Matt',
    maintainer_email='jiangmy@usc.edu',
    description='The barracuda_vision package (ROS 2 / YOLO object detection)',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_image_listener = barracuda_vision.yolo_image_listener:main',
            'test_pub_image = barracuda_vision.test_pub_image:main',
        ],
    },
)
