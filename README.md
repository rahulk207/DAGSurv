# DAGSurv

Survival analysis (SA) is a well-known statistical technique for the study of temporal events. In SA, time-to-an-event data is modeled using a parametric probabilistic function of fully or partially observed covariates.
All the existing technique for survival analysis assume that the covariates are statistically independent.
To integrate the cause-effect relationship between covariates and the time-to-event outcome, we present to you DAGSurv which encodes the causal DAG structure into the analysis of temporal data and eventually leads to better results (higher Concordance Index).

![plot](./model.png)

## Dependencies
This code requires the following key dependencies:
- Python 3.8
- torch==1.6.0
- pycox==0.2.1

There are a number of hyper-parameters present in the script which can be easily changed. 

## Usage
To train the DAGSurv model, please run the *main.py* as `python main.py`

## Experiments
We evaluated our approach on two real-world and two synthetic datasets; and used time-dependent Concordance Index(C-td) as our evaluation metric.

### Real-World Datasets
- METABRIC
- GBSG

### Time-Dependent Concordance Index(C-td)

### Results
Here, we present our results on the two real-world datasets mentioned above - 

## References
