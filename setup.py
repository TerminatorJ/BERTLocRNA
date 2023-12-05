from setuptools import setup, find_packages


with open('./requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='BERTLocRNA',
    version='0.0.4',
    author='TerminatorJ',
    author_email='wangjun19950708@gmail.com',
    description='Predicting RNA localization based on RBP binding information in BERT architecture',
    license='MIT',
    url='https://github.com/TerminatorJ/BERTLocRNA',
    packages=find_packages(),
    install_requires=requirements,
)
