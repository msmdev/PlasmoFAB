from setuptools import setup

setup(
   name='pfmptool',
   version='1.0',
   description='Module containing models to be used to predict candidate antigens',
   author='Timo Maier',
   author_email='tmaier96@gmail.com',
   packages=['pfmptool'],
   scripts=[
      'scripts/train.py',
      'scripts/pfal_dataset_creation.py',
      'scripts/evaluate_results.py',
      'scripts/dataset_plots.py',
      'scripts/boxplot_results.py',
      'scripts/export_dataset.py'
   ]
)
