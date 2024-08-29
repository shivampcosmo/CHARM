from setuptools import setup, find_packages

# Function to parse the requirements from the requirements.txt file
def parse_requirements(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line and not line.startswith("#")]


setup(
    name="charm",
    version="0.1",
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),  # Includes dependencies from requirements.txt
    author="Shivam Pandey",
    author_email="shivampcosmo@gmail.com",
    description="python code for obtaining N-body like halo catalog from fast particle mesh simulations",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/shivampcosmo/CHARM.git",  # Replace with your project's URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',  # Specify the Python versions you support
)