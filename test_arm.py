#!/usr/bin/env python3
import subprocess
import time

INITIAL_POSITIONS = [0.0, -1.6, 1.3, 0.0, -0.9, 0.0]

for i, pos in enumerate(INITIAL_POSITIONS, start=1):
    topic = f"/model/hop/arm/joint{i}/cmd_pos"
    print(f"Sending {pos} to {topic}...")
    subprocess.Popen([
        'gz', 'topic', '-t', topic, '-m', 'gz.msgs.Double', '-p', f'data: {pos}'
    ])
    time.sleep(5)

time.sleep(5)
print("Done.")
