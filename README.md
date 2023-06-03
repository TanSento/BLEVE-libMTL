# BLEVE-libMTL

``BLEVE-LibMTL`` is an adaptation of LibMTL to perform predictions on Boiling Liquid and Expanding Vapour Explosion (BLEVE).
``LibMTL`` is an open-source library built on [PyTorch](https://pytorch.org/) for Multi-Task Learning (MTL). See the [latest documentation](https://libmtl.readthedocs.io/en/latest/) for detailed introductions and API instructions. The authors' source codes are located in is https://github.com/median-research-group/LibMTL.

# BLEVE problems

``Boiling Liquid and Expanding Vapour Explosion (BLEVE)`` is an extreme blast event which can cause catastrophic consequences, imposing enormous threats to surrounding structures and personnel. Prediction of BLEVE blast loading is not feasible using simple tools. Current practice are often based on computational fluid dynamics (CFD), which requires profound expertise and are computational expensive to run, e.g., the simulation of a single BLEVE case can take days and weeks, if realistic environmental geometry is considered. Machine learning provides effective and efficient alternatives to mitigate this gap. The goal of this study is to develop machine learning approaches for BLEVE loading prediction. In this project, LibMTL has been used to perform multi-task predictions on 8 different targets generated from BLEVE blast waves, namely: **Positive Peak Time, Negative Peak Time, Arrival Time, Positive Duration, Negative Duration, Positive Pressure, Negative Pressure, Positive Impulse.**



