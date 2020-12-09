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
# 12-4-20
# 49c98a8
#
# exp1-2 are cheap to run. Do a random hp search. Let's see how good they can 
# really do. (Keep lr fixed.
# 
# RESULT: yay parkid

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
		--par_boredom='(loguniform, 1e-4, 1e-1)' \
		--kid_boredom='(loguniform, 1e-4, 1e-1)'

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
		--par_boredom='(loguniform, 1e-4, 1e-1)' 

# --------------------------------------------------------------------------
# 12-4-20
# 49c98a8
#
# Share updates (but not rewards)
# 
# RESULT: yay parkid (but a little less yay then tune1/2)

tune3: 
	python parkid/run/tune_change.py random $(DATA_PATH)/tune3 \
		--model_name='parkid' \
		--env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--share_update=True \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=25 \
		--num_processes=4 \
		--par_boredom='(loguniform, 1e-4, 1e-1)' \
		--kid_boredom='(loguniform, 1e-4, 1e-1)'

tune4: 
	python parkid/run/tune_change.py random $(DATA_PATH)/tune4 \
		--model_name='twopar' \
		 --env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--share_update=True \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=25 \
		--num_processes=4 \
		--par_boredom='(loguniform, 1e-4, 1e-1)' 

# --------------------------------------------------------------------------
# 12-8-20
# 4a782d0
#
# Kid homeostasis now iss turned on. In above it was off.
# 
# RESULT: yay parkid

tune5: 
	python parkid/run/tune_change.py random $(DATA_PATH)/tune5 \
		--model_name='parkid' \
		--env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=25 \
		--num_processes=4 \
		--par_boredom='(loguniform, 1e-4, 1e-1)' \
		--kid_boredom='(loguniform, 1e-4, 1e-1)' \
		--set_point='(uniform, 1, 120)'  

tune6: 
	python parkid/run/tune_change.py random $(DATA_PATH)/tune6 \
		--model_name='parkid' \
		--env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=25 \
		--num_processes=4 \
		--par_boredom='(loguniform, 1e-4, 1e-1)' \
		--kid_boredom='(loguniform, 1e-4, 1e-1)' \
		--set_point='(uniform, 1, 120)' \
		--share='(uniform, 1e-6, 1)'

tune7: 
	python parkid/run/tune_change.py random $(DATA_PATH)/tune7 \
		--model_name='twopar' \
		--env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=25 \
		--num_processes=4 \
		--par_boredom='(loguniform, 1e-4, 1e-1)' 

# --------------------------------------------------------------------------
# 12-9-20
# 6955692
#
# Tweaked how H is calc. Run tune5-7 again w/ this change
# 
# RESULT: yay parkid

tune8: 
	python parkid/run/tune_change.py random $(DATA_PATH)/tune8 \
		--model_name='parkid' \
		--env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=25 \
		--num_processes=4 \
		--par_boredom='(loguniform, 1e-4, 1e-1)' \
		--kid_boredom='(loguniform, 1e-4, 1e-1)' \
		--set_point='(uniform, 1, 120)'  

tune9: 
	python parkid/run/tune_change.py random $(DATA_PATH)/tune9 \
		--model_name='parkid' \
		--env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=25 \
		--num_processes=4 \
		--par_boredom='(loguniform, 1e-4, 1e-1)' \
		--kid_boredom='(loguniform, 1e-4, 1e-1)' \
		--set_point='(uniform, 1, 120)' \
		--share='(uniform, 1e-6, 1)'

tune10: 
	python parkid/run/tune_change.py random $(DATA_PATH)/tune10 \
		--model_name='twopar' \
		--env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=25 \
		--num_processes=4 \
		--par_boredom='(loguniform, 1e-4, 1e-1)' 

# --------------------------------------------------------------------------
# 12-9-20
# 6955692
#
# Try 121 bandits (matching more or less tune8-10)
# 
# RESULT: 

tune11: 
	python parkid/run/tune_change.py random $(DATA_PATH)/tune11 \
		--model_name='parkid' \
		--env_name1="BanditUniform121" \
		--env_name2="BanditChange121" \
		--change=1210 \
		--num_episodes=2420 \
		--num_samples=100 \
		--num_repeats=25 \
		--num_processes=40 \
		--par_boredom='(loguniform, 1e-6, 1e-1)' \
		--kid_boredom='(loguniform, 1e-6, 1e-1)' \
		--set_point='(uniform, 1, 120)'  

tune12: 
	python parkid/run/tune_change.py random $(DATA_PATH)/tune12 \
		--model_name='parkid' \
		--env_name1="BanditUniform121" \
		--env_name2="BanditChange121" \
		--change=1210 \
		--num_episodes=2420 \
		--num_samples=100 \
		--num_repeats=25 \
		--num_processes=40 \
		--par_boredom='(loguniform, 1e-6, 1e-1)' \
		--kid_boredom='(loguniform, 1e-6, 1e-1)' \
		--set_point='(uniform, 1, 120)' \
		--share='(uniform, 1e-6, 1)'

tune13: 
	python parkid/run/tune_change.py random $(DATA_PATH)/tune13 \
		--model_name='twopar' \
		--env_name1="BanditUniform121" \
		--env_name2="BanditChange121" \
		--change=1210 \
		--num_episodes=2420 \
		--num_samples=100 \
		--num_repeats=25 \
		--num_processes=40 \
		--par_boredom='(loguniform, 1e-6, 1e-1)' 
