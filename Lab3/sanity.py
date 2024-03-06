import subprocess
import sys
import os

os.system("nvcc lab3.cu")

wrong = 0
for i in range(int(sys.argv[1])):
    out = subprocess.getoutput("./a.out")
    print(out)
    if int(out) == 32651:
        continue
    else:
        wrong += 1

print(f"{wrong} were wrong out of {sys.argv[1]}")

