import argparse
import glob
import json
import os
import shutil
import warnings
from datetime import datetime, timedelta
from multiprocessing import Pool
from os.path import abspath, basename, dirname, exists, join
import zipfile
import logging

# turn off all warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from RCAEval.benchmark.evaluation import Evaluator
from RCAEval.classes.graph import Node

from RCAEval.io.time_series import drop_constant, drop_time, preprocess
from RCAEval.utility import (
    dump_json,
    is_py312,
    is_py310,
    is_py38,
    load_json,
    download_online_boutique_dataset,
    download_sock_shop_1_dataset,
    download_sock_shop_2_dataset,
    download_train_ticket_dataset,
    download_re1_dataset,
    download_re2_dataset,
    download_re3_dataset, 
)

# Configure logging
logging.basicConfig(level=logging.INFO)


if is_py312():
    from RCAEval.e2e import (
        baro,
        causalrca,
        circa,
        cloudranger,
        cmlp_pagerank,
        dummy,
        e_diagnosis,
        easyrca,
        fci_pagerank,
        fci_randomwalk,
        ges_pagerank,
        granger_pagerank,
        granger_randomwalk,
        lingam_pagerank,
        lingam_randomwalk,
        micro_diag,
        microcause,
        microrank,
        mscred,
        nsigma,
        ntlr_pagerank,
        ntlr_randomwalk,
        pc_pagerank,
        pc_randomwalk,
        run,
        tracerca,
    )

elif is_py38():
    from RCAEval.e2e import dummy, e_diagnosis, ht, rcd, mmrcd
else:
    print(f"Please use Python 3.8 or 3.12 to run this script.")
    exit(1)

try:
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from RCAEval.e2e.causalrca import causalrca
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="RCAEval evaluation")
    parser.add_argument("--method", type=str, help="Choose a method.")
    parser.add_argument("--dataset", type=str, help="Choose a dataset.", choices=[
        "online-boutique", "sock-shop-1", "sock-shop-2", "train-ticket",
        "re1-ob", "re1-ss", "re1-tt", "re2-ob", "re2-ss", "re2-tt", "re3-ob", "re3-ss", "re3-tt"
    ])
    parser.add_argument("--length", type=int, default=None, help="Time series length (RQ4)")
    parser.add_argument("--tdelta", type=int, default=0, help="Specify $t_delta$ to simulate delay in anomaly detection")
    parser.add_argument("--test", action="store_true", help="Perform smoke test on certain methods without fully run on all data")
    args = parser.parse_args()

    if args.method not in globals():
        raise ValueError(f"{args.method=} not defined. Please check imported methods.")

    return args


args = parse_args()

def _extract_zip_if_present(zip_name: str, extract_to: str, extract_base: str = "data"):
    """If a zip file named `zip_name` exists in cwd, extract it into `extract_base` and remove the zip.
    If the target extraction path already exists, do nothing.
    """
    target_path = os.path.join(extract_base, extract_to)
    if os.path.exists(target_path):
        return
    if os.path.exists(zip_name):
        logging.info(f'Extracting {zip_name} to {extract_base}')
        with zipfile.ZipFile(zip_name, "r") as zf:
            zf.extractall(extract_base)
        try:
            os.remove(zip_name)
        except Exception:
            pass

# download dataset (if extracted folder exists we skip; if zip exists we extract only; otherwise download)
if "online-boutique" in args.dataset or "re1-ob" in args.dataset:
    logging.info("Processing online-boutique dataset...")
    zip_name = "online-boutique.zip"
    target_path = os.path.join("data", "online-boutique")
    if os.path.exists(target_path):
        logging.info("Online-boutique dataset already extracted at %s", target_path)
    elif os.path.exists(zip_name):
        _extract_zip_if_present(zip_name, "online-boutique")
        logging.info("Online-boutique dataset extracted from %s", zip_name)
    else:
        logging.info("Downloading online-boutique dataset...")
        download_online_boutique_dataset()
elif "sock-shop-1" in args.dataset:
    logging.info("Processing sock-shop-1 dataset...")
    zip_name = "sock-shop-1.zip"
    target_path = os.path.join("data", "sock-shop-1")
    if os.path.exists(target_path):
        logging.info("Sock-shop-1 dataset already extracted at %s", target_path)
    elif os.path.exists(zip_name):
        _extract_zip_if_present(zip_name, "sock-shop-1")
        logging.info("Sock-shop-1 dataset extracted from %s", zip_name)
    else:
        logging.info("Downloading sock-shop-1 dataset...")
        download_sock_shop_1_dataset()
elif "sock-shop-2" in args.dataset or "re1-ss" in args.dataset:
    logging.info("Processing sock-shop-2 dataset...")
    zip_name = "sock-shop-2.zip"
    target_path = os.path.join("data", "sock-shop-2")
    if os.path.exists(target_path):
        logging.info("Sock-shop-2 dataset already extracted at %s", target_path)
    elif os.path.exists(zip_name):
        _extract_zip_if_present(zip_name, "sock-shop-2")
        logging.info("Sock-shop-2 dataset extracted from %s", zip_name)
    else:
        logging.info("Downloading sock-shop-2 dataset...")
        download_sock_shop_2_dataset()
elif "train-ticket" in args.dataset or "re1-tt" in args.dataset:
    logging.info("Processing train-ticket dataset...")
    zip_name = "train-ticket.zip"
    target_path = os.path.join("data", "train-ticket")
    if os.path.exists(target_path):
        logging.info("Train-ticket dataset already extracted at %s", target_path)
    elif os.path.exists(zip_name):
        _extract_zip_if_present(zip_name, "train-ticket")
        logging.info("Train-ticket dataset extracted from %s", zip_name)
    else:
        logging.info("Downloading train-ticket dataset...")
        download_train_ticket_dataset()
elif "re2" in args.dataset:
    logging.info("Processing RE2 datasets...")
    re2_base = os.path.join("data", "RE2")
    ob_path = os.path.join(re2_base, "RE2-OB")
    ss_path = os.path.join(re2_base, "RE2-SS")
    tt_path = os.path.join(re2_base, "RE2-TT")
    if os.path.exists(ob_path) and os.path.exists(ss_path) and os.path.exists(tt_path):
        logging.info("RE2 datasets already extracted at %s", re2_base)
    else:
        extracted_any = False
        if os.path.exists("RE2-OB.zip"):
            _extract_zip_if_present("RE2-OB.zip", os.path.join("RE2", "RE2-OB"), extract_base=re2_base)
            logging.info("Extracted RE2-OB.zip")
            extracted_any = True
        if os.path.exists("RE2-SS.zip"):
            _extract_zip_if_present("RE2-SS.zip", os.path.join("RE2", "RE2-SS"), extract_base=re2_base)
            logging.info("Extracted RE2-SS.zip")
            extracted_any = True
        if os.path.exists("RE2-TT.zip"):
            _extract_zip_if_present("RE2-TT.zip", os.path.join("RE2", "RE2-TT"), extract_base=re2_base)
            logging.info("Extracted RE2-TT.zip")
            extracted_any = True
        if not extracted_any:
            logging.info("Downloading RE2 datasets...")
            download_re2_dataset()
elif "re3" in args.dataset:
    logging.info("Processing RE3 datasets...")
    re3_base = os.path.join("data", "RE3")
    ob_path = os.path.join(re3_base, "RE3-OB")
    ss_path = os.path.join(re3_base, "RE3-SS")
    tt_path = os.path.join(re3_base, "RE3-TT")
    if os.path.exists(ob_path) and os.path.exists(ss_path) and os.path.exists(tt_path):
        logging.info("RE3 datasets already extracted at %s", re3_base)
    else:
        extracted_any = False
        if os.path.exists("RE3-OB.zip"):
            _extract_zip_if_present("RE3-OB.zip", os.path.join("RE3", "RE3-OB"), extract_base=re3_base)
            logging.info("Extracted RE3-OB.zip")
            extracted_any = True
        if os.path.exists("RE3-SS.zip"):
            _extract_zip_if_present("RE3-SS.zip", os.path.join("RE3", "RE3-SS"), extract_base=re3_base)
            logging.info("Extracted RE3-SS.zip")
            extracted_any = True
        if os.path.exists("RE3-TT.zip"):
            _extract_zip_if_present("RE3-TT.zip", os.path.join("RE3", "RE3-TT"), extract_base=re3_base)
            logging.info("Extracted RE3-TT.zip")
            extracted_any = True
        if not extracted_any:
            logging.info("Downloading RE3 datasets...")
            download_re3_dataset()
else:
    raise Exception(f"{args.dataset} is not defined!")

DATASET_MAP = {
    "online-boutique": "data/online-boutique",
    "sock-shop-1": "data/sock-shop-1",
    "sock-shop-2": "data/sock-shop-2",
    "train-ticket": "data/train-ticket",
    "re1-ob": "data/online-boutique",
    "re1-ss": "data/sock-shop-2",
    "re1_tt": "data/train-ticket",
    "re2-ob": "data/RE2/RE2-OB",
    "re2-ss": "data/RE2/RE2-SS",
    "re2-tt": "data/RE2/RE2-TT",
    "re3-ob": "data/RE3/RE3-OB",
    "re3-ss": "data/RE3/RE3-SS",
    "re3-tt": "data/RE3/RE3-TT"
}
dataset = DATASET_MAP[args.dataset]


# prepare input paths
logging.info("Looking for data files in: %s", dataset)
data_paths = list(glob.glob(os.path.join(dataset, "**/data.csv"), recursive=True))
if not data_paths:
    # Try looking for simple_metrics.csv directly
    data_paths = list(glob.glob(os.path.join(dataset, "**/simple_metrics.csv"), recursive=True))
    logging.info("No data.csv files found, found %d simple_metrics.csv files", len(data_paths))
else:
    logging.info("Found %d data.csv files", len(data_paths))

new_data_paths = []
for p in data_paths:
    simple_data = p.replace("data.csv", "simple_data.csv")
    simple_metrics = p.replace("data.csv", "simple_metrics.csv")
    if os.path.exists(simple_data):
        logging.info("Using simple_data.csv instead of %s", p)
        new_data_paths.append(simple_data)
    elif os.path.exists(simple_metrics):
        logging.info("Using simple_metrics.csv instead of %s", p)
        new_data_paths.append(simple_metrics)
    else:
        new_data_paths.append(p)
        logging.info("Using original file: %s", p)
data_paths = new_data_paths
if args.test is True:
    data_paths = sorted(data_paths)[:2]


# prepare output paths
from tempfile import TemporaryDirectory
# output_path = TemporaryDirectory().name
output_path = "output"
report_path = join(output_path, f"report.xlsx")
result_path = join(output_path, "results")
os.makedirs(result_path, exist_ok=True)




def process(data_path):
    run_args = argparse.Namespace()
    run_args.root_path = os.getcwd()
    run_args.data_path = data_path
    
    # convert length from minutes to seconds
    if args.length is None:
        args.length = 10
    data_length = args.length * 60 // 2

    data_dir = dirname(data_path)

    service, metric = basename(dirname(dirname(data_path))).split("_")
    case = basename(dirname(data_path))

    rp = join(result_path, f"{service}_{metric}_{case}.json")

    # == Load and Preprocess data ==
    data = pd.read_csv(data_path)
    if "time.1" in data:
        data = data.drop(columns=["time.1"])

    if "rca_" in data_path:
        data.columns = ["SIM_" + c for c in data.columns]

    if "time" not in data:
        data["time"] = data.index

    if "sock-shop" in data_path:
        data = data.loc[:, ~data.columns.str.endswith("_lat_50")]
        data = data.loc[:, ~data.columns.str.endswith("_lat_99")]

    if "train-ticket" in data_path:
        time_col = data["time"]
        data = data.loc[:, data.columns.str.startswith("ts-")]
        data["time"] = time_col

    # handle inf
    data = data.replace([np.inf, -np.inf], np.nan)

    # handle na
    data = data.fillna(method="ffill")
    data = data.fillna(0)

    cut_length = 0

    with open(join(data_dir, "inject_time.txt")) as f:
        inject_time = int(f.readlines()[0].strip()) + args.tdelta
    normal_df = data[data["time"] < inject_time].tail(data_length)
    anomal_df = data[data["time"] >= inject_time].head(data_length)
    cut_length = min(normal_df.time) - min(data.time)
    data = pd.concat([normal_df, anomal_df], ignore_index=True)

    # num column, exclude time
    num_node = len(data.columns) - 1

    # select sli for certain methods
    sli = None
    if "my-sock-shop" in data_path:
        sli = "front-end_cpu"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
    elif "sock-shop" in data_path:
        sli = "front-end_cpu"
        if f"{service}_lat_90" in data:
            sli = f"{service}_lat_90"
    elif "train-ticket" in data_path:
        sli = "ts-ui-dashboard_latency-90"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
    elif "online-boutique" in data_path or "re2-ob" in data_path:  # Added re2-ob
        sli = "frontend_latency-90"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
    
    # Default SLI selection if none matched
    if sli is None:
        # Try service-specific latency
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
            logging.info("Using service-specific latency as SLI: %s", sli)
        # Try front-end latency
        elif "frontend_latency" in data:
            sli = "frontend_latency"
            logging.info("Using frontend latency as SLI: %s", sli)
        else:
            # Default to the first metric for this service
            service_metrics = [col for col in data.columns if col.startswith(f"{service}_")]
            if service_metrics:
                sli = service_metrics[0]
                logging.info("Using first service metric as SLI: %s", sli)
            else:
                sli = next((col for col in data.columns if "latency" in col.lower()), data.columns[0])
                logging.info("Using fallback SLI: %s", sli)

    # == PROCESS ==
    func = globals()[args.method]

    try:
        st = datetime.now()
        
        out = func(
            data,
            inject_time,
            dataset=args.dataset,
            anomalies=None,
            dk_select_useful=False,
            sli=sli,
            verbose=False,
            n_iter=num_node,
            args=run_args,
        )
        root_causes = out.get("ranks")
        dump_json(filename=rp, data={0: root_causes})
    except Exception as e:
        raise e
        print(f"{args.method=} failed on {data_path=}")
        print(e)
        rp = join(result_path, f"{service}_{metric}_{case}_failed.json")
        with open(rp, "w") as f:
            json.dump({"error": str(e)}, f)


start_time = datetime.now()

for data_path in tqdm(sorted(data_paths)):
    process(data_path)

end_time = datetime.now()
time_taken = end_time - start_time
#avg_speed = round(time_taken.total_seconds() / len(data_paths), 2)
avg_speed =0


# ======== EVALUTION ===========
rps = glob.glob(join(result_path, "*.json"))
services = sorted(list(set([basename(x).split("_")[0] for x in rps])))
faults = sorted(list(set([basename(x).split("_")[1] for x in rps])))

eval_data = {
    "service-fault": [],
    "top_1_service": [],
    "top_3_service": [],
    "top_5_service": [],
    "avg@5_service": [],
    "top_1_metric": [],
    "top_3_metric": [],
    "top_5_metric": [],
    "avg@5_metric": [],
}

s_evaluator_all = Evaluator()
f_evaluator_all = Evaluator()
s_evaluator_cpu = Evaluator()
f_evaluator_cpu = Evaluator()
s_evaluator_mem = Evaluator()
f_evaluator_mem = Evaluator()
s_evaluator_lat = Evaluator()
f_evaluator_lat = Evaluator()
s_evaluator_loss = Evaluator()
f_evaluator_loss = Evaluator()
s_evaluator_io = Evaluator()
f_evaluator_io = Evaluator()

for service in services:
    for fault in faults:
        s_evaluator = Evaluator()
        f_evaluator = Evaluator()

        for rp in rps:
            s, m = basename(rp).split("_")[:2]
            if s != service or m != fault:
                continue  # ignore

            data = load_json(rp)
            if "error" in data:
                continue  # ignore

            for i, ranks in data.items():
                s_ranks = [Node(x.split("_")[0].replace("-db", ""), "unknown") for x in ranks]
                # remove duplication
                old_s_ranks = s_ranks.copy()
                s_ranks = (
                    [old_s_ranks[0]]
                    + [
                        old_s_ranks[i]
                        for i in range(1, len(old_s_ranks))
                        if old_s_ranks[i] not in old_s_ranks[:i]
                    ]
                    if old_s_ranks
                    else []
                )

                f_ranks = [Node(x.split("_")[0], x.split("_")[1]) for x in ranks]

                s_evaluator.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                f_evaluator.add_case(ranks=f_ranks, answer=Node(service, fault))

                if fault == "cpu":
                    s_evaluator_cpu.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_cpu.add_case(ranks=f_ranks, answer=Node(service, fault))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, fault))

                elif fault == "mem":
                    s_evaluator_mem.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_mem.add_case(ranks=f_ranks, answer=Node(service, fault))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, fault))

                elif fault == "delay":
                    s_evaluator_lat.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_lat.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                elif fault == "loss":
                    s_evaluator_loss.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_loss.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                elif fault == "disk":
                    s_evaluator_io.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_io.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, "latency"))

        eval_data["service-fault"].append(f"{service}_{fault}")
        eval_data["top_1_service"].append(s_evaluator.accuracy(1))
        eval_data["top_3_service"].append(s_evaluator.accuracy(3))
        eval_data["top_5_service"].append(s_evaluator.accuracy(5))
        eval_data["avg@5_service"].append(s_evaluator.average(5))
        eval_data["top_1_metric"].append(f_evaluator.accuracy(1))
        eval_data["top_3_metric"].append(f_evaluator.accuracy(3))
        eval_data["top_5_metric"].append(f_evaluator.accuracy(5))
        eval_data["avg@5_metric"].append(f_evaluator.average(5))


print("--- Evaluation results ---")
for name, s_evaluator, f_evaluator in [
    ("cpu", s_evaluator_cpu, f_evaluator_cpu),
    ("mem", s_evaluator_mem, f_evaluator_mem),
    ("io", s_evaluator_io, f_evaluator_io),
    ("delay", s_evaluator_lat, f_evaluator_lat),
    ("loss", s_evaluator_loss, f_evaluator_loss),
]:
    eval_data["service-fault"].append(f"overall_{name}")
    eval_data["top_1_service"].append(s_evaluator.accuracy(1))
    eval_data["top_3_service"].append(s_evaluator.accuracy(3))
    eval_data["top_5_service"].append(s_evaluator.accuracy(5))
    eval_data["avg@5_service"].append(s_evaluator.average(5))
    eval_data["top_1_metric"].append(f_evaluator.accuracy(1))
    eval_data["top_3_metric"].append(f_evaluator.accuracy(3))
    eval_data["top_5_metric"].append(f_evaluator.accuracy(5))
    eval_data["avg@5_metric"].append(f_evaluator.average(5))

    if name == "io":
        name = "disk"

    if s_evaluator.average(5) is not None:
        print( f"Avg@5-{name.upper()}:".ljust(12), round(s_evaluator.average(5), 2))


print("---")
print("Avg speed:", avg_speed)

