from setuptools import setup

packages = [
  'lxml',
  'numpy',
  'datetime',
],

setup(name='pmml-scoring-machine',
      version='0.3',
      description='A PMML translator',
      long_description=read_md('README.md'),
      url='https://github.com/maxkferg/pmml-gpr/tree/master/matlab',
      author='EIG',
      test_suite='tests',
      author_email='maxkferg@stanford.edu',
      license='EIG',
      packages=['pmml'],
      install_requires=packages,
      tests_require=packages,
      zip_safe=False)
