from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name = 'InfraROBManagementSystem',
    version='0.0.1',
    description='InfraROB Management System is a comprehensive framework to manage and maintain road pavements.',
    keywords=['InfraROB',
              'Road management', 
              'quality control;', 
              'decision-making',
              'performance indicators', 
              'degradation model', 
              'Markov'],
    author = 'Maurício Bonatte',
    author_email='mbonatte@ymail.com',
    url = 'https://github.com/mbonatte/InfraROBManagementSystem',
    license='GPL-3.0',
    long_description=long_description,
    
    # Dependencies
    install_requires=['numpy', 
                      'pandas',
                      'AssetManagementSystem @ git+https://github.com/mbonatte/AssetManagementSystem.git#egg=AssetManagementSystem'],
    
    # Packaging
    packages =['InfraROBManagementSystem',
               'InfraROBManagementSystem.convert',
               'InfraROBManagementSystem.optimization'],
    
)