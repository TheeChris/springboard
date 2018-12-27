from setuptools import find_packages, setup

def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()

setup(
    name='Capstone2',
    packages=find_packages(),
    version='0.1.0',
    description='Predicting hospital readmission using NLP and unstructured clinical notes',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ],
    url='https://github.com/TheeChris/springboard/Capstone_2',
    author='E. Chris Lynch',
    license='MIT',
    install_requires=['pypandoc>=1.4',
                      'pytest>=3.2.3',
                      'pytest-runner>=2.12.1',
                      'click>=6.7'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)