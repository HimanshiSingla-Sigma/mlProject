# with the help of setup.py , i will be able to make my entirre machine learning application 
# as a package and then deploy it , and then anybody can use it 

# Purpose of setup.py file
# Package your Python code so it can be installed via pip or shared with others.

# Define metadata about your project: name, version, author, dependencies, etc.

# Make your project pip-installable, either locally or via PyPI (Python Package Index).

# Automate installation of dependencies when someone installs your package.

from setuptools import setup,find_packages
from typing import List

HYPHEN_E = "-e ."
def get_requirements(file_path: str)->List[str]:
    '''
    this will return a list of packages inside requirements.txt
    '''
    with open("requirements.txt") as file_obj:
        requirements = file_obj.readlines()# this will also add \n characters inside the list
        # but we don't want them
        requirements = [requirement.replace("\n","") for requirement in requirements]
        if HYPHEN_E in requirements:
            requirements.remove(HYPHEN_E) # as we do not want .e - in ou requirements list 
    return requirements
setup(
    name="mlProject",                  
    version="0.0.1",                      
    author="Himanshi Singla",          
    author_email="singlahimanshi2@gmail.com",     
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)