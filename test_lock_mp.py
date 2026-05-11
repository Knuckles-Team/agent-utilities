import sys
import time

import ladybug

db = ladybug.Database("test_db")
print(f"Process {sys.argv[1]} opened write")
time.sleep(3)
print(f"Process {sys.argv[1]} done")
