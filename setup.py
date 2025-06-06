from setuptools import setup, find_packages
#set up flashdiv package
setup(
    name='flashdiv',
    version='0.1.0',
    description='Equivariant flow-matching model for Lennard-Jones systems',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'torch>=1.10',
        'einops>=0.6.0',
        'numpy'
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  # or your license
        'Programming Language :: Python :: 3',
    ],
)
