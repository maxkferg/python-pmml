from setuptools import setup

packages = [
  'lxml',
  'numpy',
  'datetime',
],

setup(name='pmml-python',
      version='1.0',
      description='A PMML translator',
      long_description=read_md('README.md'),
      url='https://github.com/maxkferg/pmml-python',
      author='Max Ferguson',
      test_suite='tests',
      author_email='maxferg@stanford.edu',
      license='MIT',
      packages=['pmml'],
      install_requires=packages,
      tests_require=packages,
      zip_safe=False)
