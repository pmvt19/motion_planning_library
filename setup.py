from setuptools import setup, find_packages

setup(
    name='motion_planning',
    version='1.0',
    package_data=find_packages(),
    install_requires=['numpy',
                      'shapely',
                      'matplotlib']
)