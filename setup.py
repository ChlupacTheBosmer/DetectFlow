import re
from pathlib import Path
from setuptools import setup, find_packages

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / 'README.md').read_text(encoding='utf-8')


def get_version():
    """
    Retrieve the version number from the 'detectflow/__init__.py' file.

    Returns:
        (str): The version number extracted from the '__version__' attribute in the 'detectflow/__init__.py' file.
    """
    file = PARENT / 'detectflow/__init__.py'
    return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding='utf-8'), re.M)[1]


def parse_requirements(file_path: Path):
    """
    Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.

    Args:
        file_path (str | Path): Path to the requirements.txt file.

    Returns:
        (List[str]): List of parsed requirements.
        (List[str]): List of parsed dependency links.
    """

    requirements = []
    dependency_links = []
    for line in Path(file_path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            if line.startswith('git+'):
                dependency_links.append(line)
            else:
                requirements.append(line)
    return requirements, dependency_links


install_requires, dependency_links = parse_requirements(PARENT / 'requirements.txt')

setup(
    name='detectflow',  # name of pypi package
    version=get_version(),  # version of pypi package
    python_requires='>=3.10',
    license='MIT',
    description='DetectFlow package for object detection data processing pipeline.',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/ChlupacTheBosmer/DetectFlow',
    project_urls={
        'Bug Reports': 'https://github.com/ChlupacTheBosmer/DetectFlow/issues',
        'Source': 'https://github.com/ChlupacTheBosmer/DetectFlow',
    },
    author='Petr Chlup',
    author_email='USER@natur.cuni.cz',
    packages=find_packages(exclude=['tests', 'examples']),
    package_data={
        '': ['*.yaml', '*.jpg', '*.json', '*.csv'],  # Add other data file types if needed
    },
    include_package_data=True,
    install_requires=install_requires,
    dependency_links=dependency_links, #TODO: SAHI is not being installed correctly
    extras_require={
        'ai': [
            'transformers==4.41.0',
            'flash_attn==2.5.8',
            'bitsandbytes==0.43.1',
            'openai==1.30.1'
        ],
        'dev': [
            'memory_profiler==0.61.0',
            'psutil==5.9.8'
        ],
        'vision': [
            'google-cloud-vision>=3.4.4'
        ],
        'pdf': [
            'reportlab==4.2.0'
        ],
        'excel': [
            'openpyxl==3.1.2',
            'xlwings==0.31.1'
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
    ],
    keywords='YOLO, video-object-detection, pipeline, object-detection, DetectFlow',
    entry_points={
        'console_scripts': [
            'detectflow = detectflow.main:main',  # Adjust if your main entry point is different
        ],
    },
)
