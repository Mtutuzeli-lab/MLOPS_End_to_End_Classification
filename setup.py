''' This setup.py file is used to package and distribute the Python project. It specifies the package metadata, dependencies, and other configurations required for installation. 
'''
from setuptools import find_packages, setup
from typing import List

def get_requirements()-> List[str]:
    """
    This function reads the requirements from a given file and returns them as a list of strings.


    """
    requirement_lst: List[str]=[]
    try:
        with open('requirements.txt', 'r') as file:
            # Read lines from the file of requirements
            lines=file.readlines()
            ## Process each line
            for line in lines:
                requirement=line.strip()
                ## ignore empty lines and -e .
                if requirement and requirement!= '-e .' :
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst

setup(
    name='MLOps_End_to_End_Classification',
    version='0.0.1',    
    author="Mthura ",
    author_email="mtutuzeli11@gmail.comm",
    packages=find_packages(),
    install_requires=get_requirements(),
)                     