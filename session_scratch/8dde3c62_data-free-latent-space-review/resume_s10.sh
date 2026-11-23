#!/bin/bash
# Resume S10 after the machine crash at about 00:10 CST 18 Nov: from its checkpoint at 18,700 to 20,000, then the evaluation.
T=/home/eric-luo/.claude/jobs/8dde3c62/tmp
S10=S10_S9_continued_to_20k
setsid -f nohup bash $T/snapshot_at.sh $S10 20000 > $T/S10_snap2.out 2>&1 < /dev/null
bash $T/resume_run.sh $S10 20000 --set lr=6.364e-5 --set diag_every=250
echo "training done $(date)"
bash $T/eval_s10.sh > $T/eval_S10.out 2>&1
echo "evaluation done $(date)"
