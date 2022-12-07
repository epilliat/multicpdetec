# multicpdetec

This repository provides an implementation of the bottom-up procedure defined in [Optimal multiple change-point detection for high-dimensional data](https://arxiv.org/abs/1809.09602) and compares its performances with the inspect method from [High-dimensional changepoint estimation via sparse projection](https://arxiv.org/abs/1606.06246) and where an implementation can be found in [inspect method](https://github.com/wangtengyao/InspectChangepoint/).

## Repository Structure

The procedure is implemented in `mutichange.py` and the notebook `performances.ipynb` explains how we get to the result of the paper. The csv files are the data obtained to draw the illustrations and they can be redrawn with the notebook `savingPNG.ipynb` without redoing all the computations.
