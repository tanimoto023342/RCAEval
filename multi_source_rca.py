#!/usr/bin/env python3

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import os
import zipfile
import shutil
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

def extract_re3ob(zip_path):
    """
    RE3-OBのzipファイルを解凍する関数
    
    Args:
        zip_path: RE3-OBのzipファイルパス
    """
    # 解凍先のディレクトリを作成
    extract_dir = "RE3-OB"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)  # 既存のディレクトリを削除
    os.makedirs(extract_dir)

    # zipファイルを解凍
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Successfully extracted {zip_path} to {extract_dir}")
        return True
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file")
        return False
    except Exception as e:
        print(f"Error extracting {zip_path}: {str(e)}")
        return False

def analyze_service(service_path):
    """
    特定のサービスのデータを解析する関数
    
    Args:
        service_path: サービスのデータが格納されているパス
    """
    # Read data for the specific service
    metrics = pd.read_csv(os.path.join(service_path, "metrics.csv"))
    logs = pd.read_csv(os.path.join(service_path, "logs.csv"))
    traces = pd.read_csv(os.path.join(service_path, "traces.csv"))

    # Get inject time
    with open(os.path.join(service_path, "inject_time.txt")) as ref:
        inject_time = int(ref.read().strip())

    # Preprocess data
    metrics = metrics.loc[:, ~metrics.columns.str.endswith("_latency-50")]
    metrics = metrics.replace([np.inf, -np.inf], np.nan)

    # Create multi-source data dictionary
    mmdata = {
        "metric": metrics,
        "logs": logs,
        "traces": traces,
        "logts": None,  # RE3-OBデータセットではlogtsは含まれていない
        "tracets_lat": None,  # RE3-OBデータセットではtracets_latは含まれていない
        "tracets_err": None,  # RE3-OBデータセットではtracets_errは含まれていない
        "cluster_info": None
    }

    # Perform root cause analysis using MMBARO
    output = mmbaro(mmdata, inject_time)
    ranks = output["ranks"]
    return ranks

def main():
    # RE3-OBのzipファイルパスを確認
    zip_path = "RE3-OB.zip"
    re3ob_base = "RE3-OB"

    # zipファイルが存在する場合は解凍
    if os.path.exists(zip_path):
        if not extract_re3ob(zip_path):
            print("Failed to extract RE3-OB.zip. Exiting...")
            return
    elif not os.path.exists(re3ob_base):
        print(f"Error: Neither {zip_path} nor {re3ob_base} directory found")
        return

    # サービスごとに解析を実行
    for service_dir in os.listdir(re3ob_base):
        service_path = os.path.join(re3ob_base, service_dir)
        if os.path.isdir(service_path):
            # 各フォルダ内の実験データに対して解析
            for exp_dir in os.listdir(service_path):
                exp_path = os.path.join(service_path, exp_dir, "1")  # "1"はサンプル番号
                if os.path.isdir(exp_path):
                    print(f"\nAnalyzing {service_dir}/{exp_dir}")
                    try:
                        ranks = analyze_service(exp_path)
                        print("Top 5 root cause candidates:")
                        print(ranks[:5])
                    except Exception as e:
                        print(f"Error analyzing {exp_path}: {str(e)}")

if __name__ == "__main__":
    main()