"""Replace the "Now" section of the monitor notes with now.md, stamp the edit time, and publish the file."""
import re
import shutil
import time

T = "/home/eric-luo/.claude/jobs/8dde3c62/tmp/monitor/"
OUT = "/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/monitor/RUNS.md"
text = open(T + "RUNS.md").read()
now = open(T + "now.md").read()
head, rest = text.split("## Now", 1)
tail = rest.split("## How runs are organised", 1)[1]
text = head + now.rstrip("\n") + "\n\n## How runs are organised" + tail
text = re.sub(r"Last edited: .*", "Last edited: " + time.strftime("%Y-%m-%d %H:%M %Z") + ".", text, count=1)
open(T + "RUNS.md", "w").write(text)
shutil.copy(T + "RUNS.md", OUT)
shutil.copy(T + "now.md", OUT.replace("RUNS.md", "now_section_source.md"))
print("published", OUT)
