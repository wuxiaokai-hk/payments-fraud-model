"""Clone the simulated transaction data of the Fraud Detection Handbook (Le Borgne, Siblini, Lebichot and
Bontempi, 2022, ULB Machine Learning Group and Worldline) into data/simulated-data-raw."""
import os
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dest = os.path.join(HERE, "data", "simulated-data-raw")
if not os.path.isdir(dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/Fraud-Detection-Handbook/simulated-data-raw", dest], check=True)
print("data in", dest)
