from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
__version__ = "0.0.1"
    
setup(
    name='few-shot-segmenter',
    version=__version__,
    description="Few-shot leaning model for 3D tomographic microstructure segmentaion",
    author='Po-Yen Tung',
    author_email='pyt21@cam.ac.uk',
    license='GNU GPLv3',
    url="https://github.com/poyentung/few-shot-segmenter",
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[
        "few-shot learning",
        "3D electron microscopy",
        "scanning electron microscopy",
        "segmentation",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(),
    install_requires=[
        "scipy",
        "notebook",
        "ipywidgets",
        "jupyterlab", 
        "torch             == 2.0.0",
        "torchmetrics      == 0.10.3",
        "torchvision       == 0.15.1",
        "pytorch-lightning == 1.6.0",  
        "rich              == 12.6.0",
        "hydra-core        == 1.2.0",
        "hydra-colorlog    == 1.2.0",
        "scikit-image      == 0.19.3",
        "scikit-learn      == 1.1.3", 
        "numpy             == 1.23.5",
        "numba             == 0.56.4",
        "matplotlib        == 3.6.2",
        "seaborn           == 0.12.1",
        "wandb             == 0.13.5",
        
    ],
    python_requires=">=3.9",
    package_data={
        "": ["LICENSE", "README.md"],
        "few-shot-segmenter": ["*.py"],
    },
)