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
    Returns:
        ranks: 解析結果のランキング
    Raises:
        FileNotFoundError: 必要なファイルが見つからない場合
        ValueError: データの読み込みや処理に問題がある場合
    """
    # ファイルパスの確認
    metrics_path = os.path.join(service_path, "metrics.csv")
    logs_path = os.path.join(service_path, "logs.csv")
    traces_path = os.path.join(service_path, "traces.csv")
    inject_time_path = os.path.join(service_path, "inject_time.txt")

    # 必要なファイルの存在確認
    required_files = [metrics_path, logs_path, traces_path, inject_time_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    try:
        # データの読み込み
        metrics = pd.read_csv(metrics_path)
        if metrics.empty:
            raise ValueError(f"Empty metrics data in {metrics_path}")
            
        logs = pd.read_csv(logs_path)
        if logs.empty:
            raise ValueError(f"Empty logs data in {logs_path}")
            
        traces = pd.read_csv(traces_path)
        if traces.empty:
            raise ValueError(f"Empty traces data in {traces_path}")

        # inject timeの読み込み
        with open(inject_time_path) as ref:
            inject_time_str = ref.read().strip()
            if not inject_time_str:
                raise ValueError(f"Empty inject_time in {inject_time_path}")
            inject_time = int(inject_time_str)

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
            
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Empty CSV file: {str(e)}")
    except pd.errors.ParserError as e:
        raise ValueError(f"CSV parsing error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing data: {str(e)}")



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
        if not os.path.isdir(service_path) or service_dir.startswith('.'):
            continue

        # 実験データのパスを正しく構築
        if os.path.exists(os.path.join(service_path, "1")):
            # データが直接service_pathの下にある場合
            exp_path = os.path.join(service_path, "1")
            print(f"\nAnalyzing {service_dir}")
            try:
                ranks = analyze_service(exp_path)
                print("Top 5 root cause candidates:")
                print(ranks[:5])
            except FileNotFoundError as e:
                print(f"Missing file in {exp_path}: {str(e)}")
            except ValueError as e:
                print(f"Data error in {exp_path}: {str(e)}")
            except Exception as e:
                print(f"Error analyzing {exp_path}: {str(e)}")
        else:
            # 実験タイプのサブディレクトリがある場合
            for exp_dir in os.listdir(service_path):
                if not os.path.isdir(os.path.join(service_path, exp_dir)) or exp_dir.startswith('.'):
                    continue
                    
                exp_path = os.path.join(service_path, exp_dir, "1")  # "1"はサンプル番号
                if os.path.isdir(exp_path):
                    print(f"\nAnalyzing {service_dir}/{exp_dir}")
                    try:
                        ranks = analyze_service(exp_path)
                        print("Top 5 root cause candidates:")
                        print(ranks[:5])
                    except FileNotFoundError as e:
                        print(f"Missing file in {exp_path}: {str(e)}")
                    except ValueError as e:
                        print(f"Data error in {exp_path}: {str(e)}")
                    except Exception as e:
                        print(f"Error analyzing {exp_path}: {str(e)}")

if __name__ == "__main__":
    main()