#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Yasser Alemán Gómez",
    author_email='yasseraleman@protonmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="This tool was created to parcellate T1-weighted images using the Lausanne 2018 multi-scale cortical atlas and the thalamic nuclei atlas",
    entry_points={
        'console_scripts': [
            'indivbrainparcel=indivbrainparcel.cli:main',
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='indivbrainparcel',
    name='indivbrainparcel',
    packages=find_packages(include=['indivbrainparcel', 'indivbrainparcel.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/yasseraleman/indivbrainparcel',
    version='0.1.0',
    zip_safe=False,
)
