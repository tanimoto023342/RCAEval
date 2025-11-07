#!/usr/bin/env python3

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from RCAEval.utility import download_multi_source_sample
from sklearn.preprocessing import StandardScaler, RobustScaler

from RCAEval.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    preprocess,
    select_useful_cols,
)

from RCAEval.e2e import mmbaro

def main():
    # Download a sample of multi-source data
    download_multi_source_sample()

    # Read data 
    metrics = pd.read_csv("data/multi-source-data/metrics.csv")
    logs = pd.read_csv("data/multi-source-data/logs.csv")
    traces = pd.read_csv("data/multi-source-data/traces.csv")

    # Time series transformed from parsed logs
    log_ts = pd.read_csv("data/multi-source-data/logts.csv")

    # Time series transformed from parsed traces latency and response code
    trace_ts_lat = pd.read_csv("data/multi-source-data/tracets_lat.csv")
    trace_ts_err = pd.read_csv("data/multi-source-data/tracets_err.csv")

    with open("data/multi-source-data/inject_time.txt") as ref:
        inject_time = int(ref.read().strip())

    # Preprocess data
    metrics = metrics.loc[:, ~metrics.columns.str.endswith("_latency-50")]
    metrics = metrics.replace([np.inf, -np.inf], np.nan)

    # Create multi-source data dictionary
    mmdata = {
        "metric": metrics,
        "logs": logs,
        "logts": log_ts,
        "traces": traces,
        "tracets_lat": trace_ts_lat,
        "tracets_err": trace_ts_err,
        "cluster_info": None
    }

    # Perform root cause analysis using MMBARO
    output = mmbaro(mmdata, inject_time)
    ranks = output["ranks"]
    print("Top 5 root cause candidates:")
    print(ranks[:5])

if __name__ == "__main__":
    main()