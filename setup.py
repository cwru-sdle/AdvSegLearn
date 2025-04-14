from setuptools import setup, find_packages

setup(
    name='AdvSegLearn',
    version='0.1.0',
    packages=find_packages(),
    description='Package used for a combined supervised and unsupervised learning scheme for segmentation models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Anthony Lino',
    author_email='aml334@case.edu',
    url='https://github.com/cwru-sdle/AdvSegLearn.git',
    install_requires=[
        'torch>=2.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache 2.0 License',
        'Operating System :: OS Independent',
    ],
)