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
# RESULT: parkid does better w/ these arb params. Thanks to the kid. The adult
# does not adapt.

exp1: 
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp1.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=120  --change=60 --par_boredom=0.01 --kid_boredom=0.0 --set_point=None --lr_R=0.1 --log_dir=$(DATA_PATH)/exp1/run{1} --master_seed={1}' ::: {0..20} 
		
exp2: 
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp2.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py twopar --num_episodes=120  --change=60 --par_boredom=0.001 --lr_R=0.1 --log_dir=$(DATA_PATH)/exp2/run{1} --master_seed={1}' ::: {0..20} 
			

# --------------------------------------------------------------------------
# 11-4-20
# 
#
# exp1-2 are cheap to run. Do a random hp search. Let's see how good they can 
# really do. (Keep lr fixed.
# 
# RESULT: 

tune1: 
	python parkid/run/tune_change.py random $(DATA_PATH)/tune1 \
		--model_name='parkid' \
		--env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=25 \
		--num_processes=4 \
		--log_space=True \
		--par_boredom='(1e-4, 1e-1)' \
		--kid_boredom='(1e-4, 1e-1)'

tune2: 
	python parkid/run/tune_change.py random $(DATA_PATH)/tune2 \
		--model_name='twopar' \
		 --env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=25 \
		--num_processes=4 \
		--log_space=True \
		--par_boredom='(1e-4, 1e-1)' 
