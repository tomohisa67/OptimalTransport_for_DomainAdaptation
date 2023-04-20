# OTDA with Regularization for all types of outliers
This code is for the paper "Optimal Transport with Regularization for all types of outliers; Application to Domain Adaptation". 
## Overview
* demo1.py: OTDA with real data including outliers (simple version)
* demo2.py: OTDA with real data including outliers (completed version)
* demo2_para.py: Parameter search and adopt the one with the highest accuracy
* demo3.py: OTDA with synthetic data including outliers (simple version)
* demo4.py: OTDA with synthetic data including outliers (completed version)
* demo5.py: OTDA with synthetic data including outliers and SVM experiments (simple version)
* demo6.py: OTDA with synthetic data including outliers and SVM experiments (completed version)
* demo6_savefig.py: Image-saved version of demo6.py
* result.ipynb: Visualizing results
* result.csv: Output file
* utils.py: Various useful functions

## Algorithm
* "OT" and "UOT" are OTDA method and OTDA with OT changed to UOT respectively.
* "OTc" and "UOTc" are proposed methods.
* "UOTn" is the method that takes into account the distance of the nearest neighbor samples.

## Requirment
* Python Optimal Transport: https://pythonot.github.io/
```
pip install pot
```

## Dataset
Download Office+Caltech SURF dataset from below and put them in the folder "./data".
* Office+Caltech: https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md

## Examples
Set the parameters in the code and execute the following command. The results are output to the csv file (result.csv).
```
python demo2.py
```

## Author
This code was written for educational purposes and is not guaranteed to be correct. If there are any errors, please do not hesitate to contact me."

Author: T.Tabuchi  
e-mail: tyoge1054a@fuji.waseda.jp

