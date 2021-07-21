SHELL=/bin/bash -O expand_aliases
DATA_PATH=/Users/qualia/Code/parkid/data
# DATA_PATH=/home/stitch/Code/parkid/data/

# --------------------------------------------------------------------------
# Test recipes
#
# Should run ok when run from main/HEAD.
test1: 
	-rm -rf $(DATA_PATH)/test1/*
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp1.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --kid_boredom=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/test1/run{1} --master_seed={1}' ::: {0..100} 

test2: 
	-rm -rf $(DATA_PATH)/test2/*
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp2.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/test2/run{1} --master_seed={1}' ::: {0..100} 
	

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
			'python parkid/run/change_bandits.py parpar --num_episodes=120  --change=60 --par_boredom=0.001 --lr_R=0.1 --log_dir=$(DATA_PATH)/exp2/run{1} --master_seed={1}' ::: {0..20} 
			

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
		--model_name='parpar' \
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
		--model_name='parpar' \
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
		--model_name='parpar' \
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
		--model_name='parpar' \
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
# Try 121 bandits (matching tune8-10, more or less)
# 
# RESULT: yay parkid

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
		--par_boredom='(loguniform, 1e-4, 1e-1)' \
		--kid_boredom='(loguniform, 1e-4, 1e-1)' \
		--set_point='(uniform, 1, 2420)'  

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
		--par_boredom='(loguniform, 1e-4, 1e-1)' \
		--kid_boredom='(loguniform, 1e-4, 1e-1)' \
		--set_point='(uniform, 1, 2420)' \
		--share='(uniform, 1e-6, 1)'

tune13: 
	python parkid/run/tune_change.py random $(DATA_PATH)/tune13 \
		--model_name='parpar' \
		--env_name1="BanditUniform121" \
		--env_name2="BanditChange121" \
		--change=1210 \
		--num_episodes=2420 \
		--num_samples=100 \
		--num_repeats=25 \
		--num_processes=40 \
		--par_boredom='(loguniform, 1e-6, 1e-1)' 

# --------------------------------------------------------------------------
# 12-9-20
# a66e601
#
# Generate example for tuned params.
# - top-1 do 10 examples
# - top-10 do 1 example (fixed seed)
# 
# Tune targets are:
# - tune8-10 (4 arm)
# - tune11-13 (121 arm)
#
# RESULT: The scores for parkid are better (as during tune above) but looking the the behave we are still far from the idea. The kid is not having the influence hand tuning exps suggest is possible. I should get better behave if I 
# 1. Rerun tune but constrain boredom  kid < adult
# 2. fix the set_point to the change
# 3. Use par_boredom from parkid on parpar (i will not always want to do this, see 4)
# 4. Compare easily bored and curious adults to the parkid. 

# -
# tune8-10
# -

## -
## top-1
### tune8
exp3: 
	# Get top 1
	head -n 2 $(DATA_PATH)/tune8_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp3.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			"python parkid/run/change_bandits.py parkid --env_name1=BanditUniform4 --env_name2=BanditChange4 --num_episodes=120 --change=60 --par_boredom={par_boredom} --kid_boredom={kid_boredom} --set_point={set_point} --lr_R=0.1 --log_dir=$(DATA_PATH)/exp3/param{index}/run{1} --master_seed={1}" ::: {0..10} :::: tmp
	# Clean up
	rm tmp

### tune9
exp4: 
	# Get top 1
	head -n 2 $(DATA_PATH)/tune9_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp4.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --env_name1=BanditUniform4 --env_name2=BanditChange4 --num_episodes=120 --change=60 --par_boredom={par_boredom} --kid_boredom={kid_boredom} --set_point={set_point} --share={share} --lr_R=0.1 --log_dir=$(DATA_PATH)/exp4/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

### tune10
exp5: 
	# Get top 1
	head -n 2 $(DATA_PATH)/tune10_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp5.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --env_name1=BanditUniform4 --env_name2=BanditChange4 --num_episodes=120 --change=60 --par_boredom={par_boredom} --lr_R=0.1 --log_dir=$(DATA_PATH)/exp5/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

## -
## top-10
### tune8
exp6: 
	# Get top 1
	head -n 11 $(DATA_PATH)/tune8_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp6.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --env_name1=BanditUniform4 --env_name2=BanditChange4 --num_episodes=120 --change=60 --par_boredom={par_boredom} --kid_boredom={kid_boredom} --set_point={set_point} --lr_R=0.1 --log_dir=$(DATA_PATH)/exp6/param{index}/run1 --master_seed=42' :::: tmp
	# Clean up
	rm tmp

### tune9
exp7: 
	# Get top 1
	head -n 11 $(DATA_PATH)/tune9_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp7.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --env_name1=BanditUniform4 --env_name2=BanditChange4 --num_episodes=120 --change=60 --par_boredom={par_boredom} --kid_boredom={kid_boredom} --set_point={set_point} --share={share} --lr_R=0.1 --log_dir=$(DATA_PATH)/exp7/param{index}/run1 --master_seed=42' :::: tmp
	# Clean up
	rm tmp

### tune10
exp8: 
	# Get top 1
	head -n 11 $(DATA_PATH)/tune10_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp8.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --env_name1=BanditUniform4 --env_name2=BanditChange4 --num_episodes=120 --change=60 --par_boredom={par_boredom} --lr_R=0.1 --log_dir=$(DATA_PATH)/exp8/param{index}/run1 --master_seed=42' :::: tmp
	# Clean up
	rm tmp

# -
# tune11-13
# -

## -
## top-1
### tune8
exp9: 
	# Get top 1
	head -n 2 $(DATA_PATH)/tune11_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp9.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			"python parkid/run/change_bandits.py parkid --env_name1=BanditUniform121 --env_name2=BanditChange121 --num_episodes=2420 --change=1210 --par_boredom={par_boredom} --kid_boredom={kid_boredom} --set_point={set_point} --lr_R=0.1 --log_dir=$(DATA_PATH)/exp9/param{index}/run{1} --master_seed={1}" ::: {0..10} :::: tmp
	# Clean up
	rm tmp

### tune9
exp10: 
	# Get top 1
	head -n 2 $(DATA_PATH)/tune12_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp10.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --env_name1=BanditUniform121 --env_name2=BanditChange121 --num_episodes=2420 --change=1210 --par_boredom={par_boredom} --kid_boredom={kid_boredom} --set_point={set_point} --share={share} --lr_R=0.1 --log_dir=$(DATA_PATH)/exp10/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

### tune10
exp11: 
	# Get top 1
	head -n 2 $(DATA_PATH)/tune13_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp11.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --env_name1=BanditUniform121 --env_name2=BanditChange121 --num_episodes=2420 --change=1210 --par_boredom={par_boredom} --lr_R=0.1 --log_dir=$(DATA_PATH)/exp11/param{index}/run{1} --master_seed={1}' ::: {0..10} :::: tmp
	# Clean up
	rm tmp

## -
## top-10
### tune8
exp12: 
	# Get top 1
	head -n 11 $(DATA_PATH)/tune11_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp12.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --env_name1=BanditUniform121 --env_name2=BanditChange121 --num_episodes=2420 --change=1210 --par_boredom={par_boredom} --kid_boredom={kid_boredom} --set_point={set_point} --lr_R=0.1 --log_dir=$(DATA_PATH)/exp12/param{index}/run1 --master_seed=42' :::: tmp
	# Clean up
	rm tmp

### tune9
exp13: 
	# Get top 1
	head -n 11 $(DATA_PATH)/tune12_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp13.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --env_name1=BanditUniform121 --env_name2=BanditChange121 --num_episodes=2420 --change=1210 --par_boredom={par_boredom} --kid_boredom={kid_boredom} --set_point={set_point} --share={share} --lr_R=0.1 --log_dir=$(DATA_PATH)/exp13/param{index}/run1 --master_seed=42' :::: tmp
	# Clean up
	rm tmp

### tune10
exp14: 
	# Get top 1
	head -n 11 $(DATA_PATH)/tune13_sorted.csv > tmp 
	# Run them 10 times
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp14.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --env_name1=BanditUniform121 --env_name2=BanditChange121 --num_episodes=2420 --change=1210 --par_boredom={par_boredom} --lr_R=0.1 --log_dir=$(DATA_PATH)/exp14/param{index}/run1 --master_seed=42' :::: tmp
	# Clean up
	rm tmp


# --------------------------------------------------------------------------
# 12-10-2020
# a2fe023
#
# More refined tune to better seperate parkid from parpar

# 1. Constrain boredom  kid < adult
# 2. fix the `set_point` to the `change`
# 3. Use par_boredom from parkid on parpar (i will not always want to do this, see 4)
# 4. Compare easily bored and curious adults to the parkid. 

# RESULTS: yay parkid. our model offers a far more robust range. there is 
#          a max level for par boredom where the kid can't help anymore;
#          an important limit for the future.

# -
# parkid
# Sweep par_boredom
tune14: 
	python parkid/run/tune_change.py $(DATA_PATH)/tune14 \
		--model_name='parkid' \
		--env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--metric='total_R' \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=25 \
		--num_processes=4 \
		--master_seed=42 \
		--par_boredom='(linspace, 1e-5, 1e-1)' \
		--kid_boredom=0.0001 \
		--set_point=60 \
		--share=0.0

# Sweep kid_boredom
tune15: 
	python parkid/run/tune_change.py $(DATA_PATH)/tune15 \
		--model_name='parkid' \
		--env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--metric='total_R' \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=25 \
		--num_processes=4 \
		--master_seed=42 \
		--par_boredom=1e-3 \
		--kid_boredom='(linspace, 1e-5, 1e-3)' \
		--set_point=60 \
		--share=0.0

# Sample kid_boredom and set_point
tune16: 
	python parkid/run/tune_change.py $(DATA_PATH)/tune16 \
		--model_name='parkid' \
		--env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--metric='total_R' \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=25 \
		--num_processes=4 \
		--master_seed=42 \
		--par_boredom=1e-3 \
		--kid_boredom='(loguniform, 1e-5, 1e-3)' \
		--set_point='(uniform, 1, 120)' \
		--share=0.0 

# Samplle kid_boredom and set_point and share
tune17: 
	python parkid/run/tune_change.py $(DATA_PATH)/tune17 \
		--model_name='parkid' \
		--env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--metric='total_R' \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=25 \
		--num_processes=4 \
		--master_seed=42 \
		--par_boredom=1e-3 \
		--kid_boredom='(loguniform, 1e-5, 1e-3)' \
		--set_point='(uniform, 1, 120)' \
		--share='(uniform, 0.01, 1)'

# -
# parpar
# sweep par_bordom
tune18: 
	python parkid/run/tune_change.py $(DATA_PATH)/tune18 \
		--model_name='parpar' \
		--env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--metric='total_R' \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=25 \
		--num_processes=4 \
		--master_seed=42 \
		--par_boredom='(linspace, 1e-5, 1e-1)' 


# --------------------------------------------------------------------------
# 12-10-2020
# b71b89b
#
# ***Used to generate data for Feb 2021 TRI grant***
#
# Do a parkid run, and parpar run, with a shared boredom level - 0.025.
#
# I choose a boredom level that leads to good performance of both models on the 
# first 60 trials before the chage, but is problem for parpar after 
# the change.
#
# The tuning runs in tune11-18 suggest there are large regions of
# parameter space (boredom) for which this hold true.
#
# ...It is possible to tune parpar to do as well, of course. It's just more
# difficult.
exp15: 
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp15.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --env_name1=BanditUniform4 --env_name2=BanditChange4 --num_episodes=120 --change=60 --par_boredom=0.025 --kid_boredom=0.0001 --set_point=60 --share=0.0 --log_dir=$(DATA_PATH)/exp15/param0/run{1} --master_seed={1}' ::: {0..100} 

exp16: 
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp16.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --env_name1=BanditUniform4 --env_name2=BanditChange4 --num_episodes=120 --change=60 --par_boredom=0.025 --log_dir=$(DATA_PATH)/exp16/param0/run{1} --master_seed={1}' ::: {0..100} 


# --------------------------------------------------------------------------
# 7-19-2021
# c7e3f47 
#
# ***Used to generate data for current biology draft***
#
# Changes from grant results given by exp15/16: 
# - I fixed/unified the way total rewards are calculated. This is new was is more
# fair. 
#
# Run the same simulations as exp15/16 (above) to make sure nothinh 
# inportant changed the results I expect now.
# 
# RESULTS: overall the results are consistent. From this analysis it is clear that sometimes, by change, parpar finds the perfect arm after the change. This give it near perfect scores some of the time. Not often enough. Its avg reward is still lower. And much lower is these lucky guesses (?) are removed.

exp17: 
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp17.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --env_name1=BanditUniform4 --env_name2=BanditChange4 --num_episodes=120 --change=60 --par_boredom=0.025 --kid_boredom=0.0001 --set_point=60 --share=0.0 --log_dir=$(DATA_PATH)/exp17/param0/run{1} --master_seed={1}' ::: {0..100} 

exp18: 
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp18.log' \
			--nice 19 --delay 0 --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --env_name1=BanditUniform4 --env_name2=BanditChange4 --num_episodes=120 --change=60 --par_boredom=0.025 --log_dir=$(DATA_PATH)/exp18/param0/run{1} --master_seed={1}' ::: {0..100} 


# --------------------------------------------------------------------------
# 7-19-2021
# c7e3f47 
# ***Used to generate data for current biology draft***
# 
# Tune sweep for BanditUniform4 -- all sensible params
tune19: 
	python parkid/run/tune_change.py $(DATA_PATH)/tune19 \
		--model_name='parkid' \
		--env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--metric='total_R' \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=100 \
		--num_processes=40 \
		--set_point=60
		--par_boredom='(loguniform, 1e-4, 1e-1)' \
		--kid_boredom='(loguniform, 1e-4, 1e-1)' \

tune20: 
	python parkid/run/tune_change.py $(DATA_PATH)/tune20 \
		--model_name='parpar' \
		--env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--metric='total_R' \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=100 \
		--num_processes=40 \
		--par_boredom='(loguniform, 1e-4, 1e-1)' 


# --------------------------------------------------------------------------
# 7-19-2021
# c7e3f47 
# # ***Used to generate data for current biology draft***

# More refined tune to better seperate parkid from parpar, looking 
# aat only parent boredom

# parkid
# Sweep par_boredom
tune21: 
	python parkid/run/tune_change.py $(DATA_PATH)/tune21 \
		--model_name='parkid' \
		--env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--metric='total_R' \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=100 \
		--num_processes=40 \
		--master_seed=42 \
		--par_boredom='(linspace, 1e-5, 1e-1)' \
		--kid_boredom=0.0001 \
		--set_point=60 \
		--share=0.0

# parpar
tune22: 
	python parkid/run/tune_change.py $(DATA_PATH)/tune22 \
		--model_name='parpar' \
		--env_name1="BanditUniform4" \
		--env_name2="BanditChange4" \
		--change=60 \
		--metric='total_R' \
		--num_episodes=120 \
		--num_samples=200 \
		--num_repeats=100 \
		--num_processes=40 \
		--master_seed=42 \
		--par_boredom='(linspace, 1e-5, 1e-1)' 


# --------------------------------------------------------------------------
# 7-19-2021
# c7e3f47 
# # ***Used to generate data for current biology draft***

# Tune sweep for BanditUniform121 -- all sensible params

# parkid
tune23: 
	python parkid/run/tune_change.py $(DATA_PATH)/tune23 \
		--model_name='parkid' \
		--env_name1="BanditUniform121" \
		--env_name2="BanditChange121" \
		--change=1210 \
		--metric='total_R' \
		--num_episodes=2420 \
		--num_samples=100 \
		--num_repeats=100 \
		--num_processes=40 \
		--par_boredom='(loguniform, 1e-6, 1e-1)' \
		--kid_boredom='(loguniform, 1e-6, 1e-1)' 

# parpar
tune24: 
	python parkid/run/tune_change.py $(DATA_PATH)/tune24 \
		--model_name='parpar' \
		--env_name1="BanditUniform121" \
		--env_name2="BanditChange121" \
		--change=1210 \
		--metric='total_R' \
		--num_episodes=2420 \
		--num_samples=100 \
		--num_repeats=100 \
		--num_processes=40 \
		--par_boredom='(loguniform, 1e-6, 1e-1)' 


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# 7-19-2021
# c7e3f47 
# ***Used to generate data for current biology draft***
# 
# Tune sweep for BanditStaticMonster4 -- all sensible params

tune26: 
	python parkid/run/tune_change.py $(DATA_PATH)/tune26 \
		--model_name='parkid' \
		--env_name1="BanditStaticMonster4" \
		--env_name2="BanditDynamicMonster4" \
		--change=40 \
		--metric='total_R' \
		--stat='median' \
		--num_episodes=120 \
		--num_samples=100 \
		--num_repeats=100 \
		--num_processes=40 \
		--lr_R=0.6 \
		--kid_boredom=0.0 \
		--set_point=20 \
		--kid_scale='(uniform, 0.25, 5)' \
		--par_boredom='(loguniform, 0.0001, 0.99)' 
		

tune27: 
	python parkid/run/tune_change.py $(DATA_PATH)/tune27 \
		--model_name='parpar' \
		--env_name1="BanditStaticMonster4" \
		--env_name2="BanditDynamicMonster4" \
		--change=40 \
		--metric='total_R' \
		--stat='median' \
		--num_episodes=120 \
		--num_samples=100 \
		--num_repeats=100 \
		--num_processes=40 \
		--lr_R=0.6 \
		--par_boredom='(loguniform, 0.0001, 0.99)' 

# -------------------------------------------------------------------------
# Experiments with more faithful replica's of Sumner's design suggest
# then the benfits of change detection are not large, as in her desgin,
# the, what I will call, "chattering" of kids curiosity distracts from
# exploitation in parents. This lead to a net loss in total reward,
# even when change detection happens. 
#
# In the experiments below I will explore a set of monster tasks, with
# fixed parameters for ParKid and ParPar. There parameters came from 
# hand tuning and seem to be good enough. Though, later I'll revisst them?
#
# Onto the tasks.
#
# First lets run a fathful replica of Emily's task, with matched trial numbers.
# Past experiments with test1/test2 suggest this should show the trade-off
# I described above.

# RESULTS: For exp20-31. The trade-off between total and change reward is
#          pretty apparent as I go between the monster tasks. It takes
#          less of a change difference to lead to net a win than I feared
#          at the start of these experiments. Still, I should be able to 
#          show a win in Emily's org task, or at least my version of it.
#   
# 	       Further, adults should be able to gaurd againts the chatter
#          of kids curiosity in general so it does not hamper their day
#          to day exploitation. A nonlinear function/gate would do this?
#
#          Try out a relu with a tunable parent_threshold? More parameters,
#          argh!? Can't see a way round, unfortunatly.

exp20: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --kid_boredom=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp20/run{1} --master_seed={1}' ::: {0..100} 

exp21: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp21/run{1} --master_seed={1}' ::: {0..100} 
	
# Now lets run a (new) series of tasks, which are variations of Sumners. I
# call the the BugMonsrt tasks becuase we exploring the smallest to biggest
# changes we can given the basic task.
#
# The base ('static') Env is BanditBigMonster1. I then try variations for env_2.

# env_name2=BanditBigMonster6
exp22: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster6 --par_boredom=0.01 --kid_boredom=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp22/run{1} --master_seed={1}' ::: {0..100} 

exp23: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster6 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp23/run{1} --master_seed={1}' ::: {0..100} 
	
# env_name2=BanditBigMonster7
exp24: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster7 --par_boredom=0.01 --kid_boredom=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp24/run{1} --master_seed={1}' ::: {0..100} 

exp25: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster7 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp25/run{1} --master_seed={1}' ::: {0..100} 
	
# env_name2=BanditBigMonster8
exp26: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster8 --par_boredom=0.01 --kid_boredom=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp26/run{1} --master_seed={1}' ::: {0..100} 

exp27: 
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp2.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster8 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp27/run{1} --master_seed={1}' ::: {0..100} 

# env_name2=BanditBigMonster9
exp28: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --par_boredom=0.01 --kid_boredom=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp28/run{1} --master_seed={1}' ::: {0..100} 

exp29: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp29/run{1} --master_seed={1}' ::: {0..100} 

# env_name2=BanditBigMonster10
exp30: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --par_boredom=0.01 --kid_boredom=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp30/run{1} --master_seed={1}' ::: {0..100} 

exp31: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp31/run{1} --master_seed={1}' ::: {0..100} 
	
