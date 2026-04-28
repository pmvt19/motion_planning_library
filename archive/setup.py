from setuptools import setup, find_packages

setup(
    name='motion_planning',
    version='1.1',
    py_modules=['motion_planning'],
    install_requires=['numpy',
                      'shapely',
                      'matplotlib']
)