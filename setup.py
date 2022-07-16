from setuptools import setup, find_packages


with open('requirements.txt', 'r') as f:
    requirements = f.readlines()
    requirements = [r.strip() for r in requirements]


setup(
    name='slowmatch',
    description='A python implementation of minimum-weight embedded matching',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3'
)
