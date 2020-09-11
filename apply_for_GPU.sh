srun -t1:30:00 -c 4 --mem=3000 --gres=gpu:1 --pty /bin/bash

scancel 100001 #用来取消job
