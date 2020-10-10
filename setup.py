from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='long_range_convolutions',
    version='0.1.0',
    description='Efficient Long Range Convolutions for Point Clouds',
    long_description=readme,
    author='Leonardo Zepeda-Nunez',
    author_email='zepedanunez@wisc.edu',
    url='https://github.com/Forgotten/BathFitting',
    license=license,
    install_requires=['numpy', 'scipy', 'numba', 'tensorflow'],
    packages=find_packages(),
    classifiers=["Programming Language :: Python :: 3",
                "License :: MIT License",
                "Operating System :: OS Independent",],
)