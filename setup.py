from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='Convolution Bag train on gcloud ml-engine',
      author ='Tasnia Tahsin',
      author_email='ttahsin88@gmail.com',
      install_requires=[
          'tensorflow-gpu',
          'numpy',
          'sklearn',
          'matplotlib',
      ],
      zip_safe=False)