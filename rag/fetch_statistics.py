import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("log_file_path", help="path of log file")
args = parser.parse_args()
log_file_path = args.log_file_path
filename = log_file_path.split("/")[-1]

with open(log_file_path, "r", encoding="utf-8") as file:
    log_content = file.read()

rag_avg_time_match = re.search(r'rag_time_avg:\s*([\d.]+)', log_content)

average_f1_match = re.search(r"Overall F1: ([0-9.]+) \((\d+)\)", log_content)

if rag_avg_time_match and average_f1_match:
    rag_avg_time = rag_avg_time_match.group(1)
    average_f1 = float(average_f1_match.group(1))
    total_num = int(average_f1_match.group(2))
    print(f"{filename}\t rag_time_avg: {rag_avg_time}, \tavg_f1: {average_f1} ({total_num})")
else:
    print("No matching data found")
