from setuptools import setup

setup(name='diffsim',
      version='1.0',
      install_requires=['gym>=0.5',
                        'sklearn',
                        'torch',
                        'numpy',
                        'matplotlib',
                        'scipy',
                        'bayesian-optimization',
                        'box2d',
                        'box2d-kengz',
                        'mujoco_py',
                        'lxml',
                        'tqdm',
                        'gym_ergojr>=1.2']
      )
