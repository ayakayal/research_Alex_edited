import os

exps = os.listdir("data/")

for exp in exps:
    print(exp)
    os.system(f"python plot_results.py {exp}")
