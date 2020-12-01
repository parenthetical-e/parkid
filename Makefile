SHELL=/bin/bash -O expand_aliases
DATA_PATH=/Users/qualia/Code/parkid/data
# DATA_PATH=/home/stitch/Code/parkid/data/

# --------------------------------------------------------------------------
# 11-30-20
# 9c1a244
#
# Intial test of parkid versus par on the change bandits. Number of 
# interactions is conserved and learning rate is shared.
# 
# RESULT: 

exp1: 
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp1.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=2420  --change=1210 --par_boredom=0.01 --par_boredom=0.01 --set_point=None --lr_R=0.1 --log_dir=$(DATA_PATH)/exp1/run{1} --master_seed={1}' ::: {0..10} 
		
exp2: 
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp2.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py par --num_episodes=4840  --change=2420 --par_boredom=0.01 --lr_R=0.1 --log_dir=$(DATA_PATH)/exp2/run{1} --master_seed={1}' ::: {0..10} 
			