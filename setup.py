from setuptools import setup

setup(
    name = 'pydis',
    version = '1.0',
    description = 'A simple longslit spectroscopy pipeline in Python',
    author = 'James Davenport',
    author_email = 'jradavenport@gmail.com',
    license = 'MIT',
    url = 'https://github.com/jradavenport/pydis',
    packages = ['pydis'],
    package_data = {
	'': ['resources/*.dat', 'resources/linelists/*.dat'],
    },
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is the project?
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        # should match "license" above
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
