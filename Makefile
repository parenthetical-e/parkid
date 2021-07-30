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
			--joblog '$(DATA_PATH)/test1.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --kid_boredom=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/test1/run{1} --master_seed={1} --output=False' ::: {0..100} 

test2: 
	-rm -rf $(DATA_PATH)/test2/*
	parallel -j 4 \
			--joblog '$(DATA_PATH)/test2.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/test2/run{1} --master_seed={1} --output=False' ::: {0..100} 
	
test3: 
	-rm -rf $(DATA_PATH)/test3/*
	parallel -j 4 \
			--joblog '$(DATA_PATH)/test3.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py ucbucb --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --temp=0.001 --beta=0.2 --lr_R=0.6 --log_dir=$(DATA_PATH)/test3/run{1} --master_seed={1} --output=False' ::: {0..100} 

test4: 
	-rm -rf $(DATA_PATH)/test4/*
	parallel -j 4 \
			--joblog '$(DATA_PATH)/test4.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py oracle --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --lr_R=0.6 --log_dir=$(DATA_PATH)/test4/run{1} --master_seed={1} --output=False' ::: {0..100} 

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
# I began by making some smaall changes to the TRI results. I became clear
# quick there was more to learn. So, I explored Emily's task and others
# over a series of experiments. These span exp17-exp77.
#
# The data for the current biology was generated from exp78, and on.
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
# 7/21/2021
# aa36bf0 
#
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

# RESULTS: **For exp20-31.** The trade-off between total and change reward is
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
	

# ----------------------------------------------------------------------------
# 7/21/2021
# f806e31
#
# Try out a relu with a tunable parent_threshold? More parameters,
# argh!? Can't see a way round, unfortunatly.
#
# Repeat of exp20-21 (Reg monsters) but trying differnt parent_thresolds.
# 
# RESULTS: **exp32-34** there was maybe a small change by 
#          --parent_threshold=0.01, nothing visible before that. 
#          Bump up a little more? 
#
#          **exp33-39** values past 0.01 don't have much more of an effect. 
#          Having >0.01 does _seem_ to mitigate the loss in total value,
#          which means the loss in the early phases. Changes in behavoir
#          plots seem consistent with this, though I am not sure how to
#          quant this behave change. I am just eyeballing, unfortunatly.
#          Overall this a lot of 'seem to' in this results. This is too 
#          insensitive a test? Why is that? 
#
#          ....
#
#          Looking again at the last boxplot in exp32-39.Rmd does show
#          the trends I talked about in the last para. 
# 
# 	       Lets see is the a gate at 0.01 makes kids more helpful in the 
#          the big bandit series? Kind of a let down here.
exp32: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.001 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp32/run{1} --master_seed={1}' ::: {0..100} 

exp33: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.005 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp33/run{1} --master_seed={1}' ::: {0..100} 

exp34: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp34/run{1} --master_seed={1}' ::: {0..100} 

# Only need one control, with parpar
exp35: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp35/run{1} --master_seed={1}' ::: {0..100} 

# ---
# Try a few more with parent_threshold>0.01

# parent_threshold=0.02
exp36: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.02 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp36/run{1} --master_seed={1}' ::: {0..100} 

# parent_threshold=0.03
exp37: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.03 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp37/run{1} --master_seed={1}' ::: {0..100} 

# parent_threshold=0.04
exp38: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.04 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp38/run{1} --master_seed={1}' ::: {0..100} 

# parent_threshold=0.05
exp39: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.05 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp39/run{1} --master_seed={1}' ::: {0..100} 


# ---------------------------------------------------------------------------
# 7/22/21
# 481783b
#
# Try a progression on the big monster series with gating on. --parent_threshold=0.01 looked best (small amount) in the exp32-39 series.
#
# So, try that.
#
# RESULTS: **exp40-49** Gating w/ 0.01 did not have an effect until exp48
#          (BanditBigMonster10) where is had a very clear positive benefit.
#
#          Try this series again but bump kid_scale, so we are gating
#          and subjectivly amplifiying.


# env_name2=BanditBigMonster6
exp40: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster6 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp40/run{1} --master_seed={1} --output=False' ::: {0..100} 

exp41: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster6 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp41/run{1} --master_seed={1} --output=False' ::: {0..100} 
	
# env_name2=BanditBigMonster7
exp42: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster7 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp42/run{1} --master_seed={1} --output=False' ::: {0..100} 

exp43: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster7 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp43/run{1} --master_seed={1} --output=False' ::: {0..100} 
	
# env_name2=BanditBigMonster8
exp44: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster8 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp44/run{1} --master_seed={1} --output=False' ::: {0..100} 

exp45: 
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp2.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster8 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp45/run{1} --master_seed={1} --output=False' ::: {0..100} 

# env_name2=BanditBigMonster9
exp46: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp46/run{1} --master_seed={1} --output=False' ::: {0..100} 

exp47: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp47/run{1} --master_seed={1} --output=False' ::: {0..100} 

# env_name2=BanditBigMonster10
exp48: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp48/run{1} --master_seed={1} --output=False' ::: {0..100} 

exp49: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp49/run{1} --master_seed={1} --output=False' ::: {0..100} 
	
# ---------------------------------------------------------------------------
# 7/22/21
# 481783b

# Try a progression on the big monster series with gating on. --parent_threshold=0.01 and speculativly sets --kid_scale=2
#
# Builds on exp40-49 
#
# RESULTS: **exp50-59** Uping kid_scale=2 meant the gate began having a 
#          a notable effect by BanditBigMonster8, instead of BanditBigMonster10
#          which was the only task exp40-49 saw an advantage to gating.
#
#          Up to kid_scale=4?


# env_name2=BanditBigMonster6
exp50: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster6 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=2 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp50/run{1} --master_seed={1} --output=False' ::: {0..100} 

exp51: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster6 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp51/run{1} --master_seed={1} --output=False' ::: {0..100} 
	
# env_name2=BanditBigMonster7
exp52: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster7 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=2 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp52/run{1} --master_seed={1} --output=False' ::: {0..100} 

exp53: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster7 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp53/run{1} --master_seed={1} --output=False' ::: {0..100} 
	
# env_name2=BanditBigMonster8
exp54: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster8 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=2 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp54/run{1} --master_seed={1} --output=False' ::: {0..100} 

exp55: 
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp2.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster8 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp55/run{1} --master_seed={1} --output=False' ::: {0..100} 

# env_name2=BanditBigMonster9
exp56: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=2 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp56/run{1} --master_seed={1} --output=False' ::: {0..100} 

exp57: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp57/run{1} --master_seed={1} --output=False' ::: {0..100} 

# env_name2=BanditBigMonster10
exp58: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=2 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp58/run{1} --master_seed={1} --output=False' ::: {0..100} 

exp59: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp59/run{1} --master_seed={1} --output=False' ::: {0..100} 
	

# ---------------------------------------------------------------------------
# 7/22/21
# 44465f3
#
# Try a progression on the big monster series with gating on. 
# --parent_threshold=0.01 and set a (higher) value of --kid_scale=4
#
# Builds on exp50-59 
#
# RESULTS: **exp60-69** Identical pattern of results as exp50-59. I am tempted 
#          to double again, to 8, but it might be a waste of time. Table it for
#          now
#
#          Try the reg. bandits with --parent_threshold=0.01 and a
#          --kid_scale=4. With amplification perhaps the kid will
#          help 'enough'


# env_name2=BanditBigMonster6
exp60: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster6 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=4 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp60/run{1} --master_seed={1} --output=False' ::: {0..100} 

exp61: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster6 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp61/run{1} --master_seed={1} --output=False' ::: {0..100} 
	
# env_name2=BanditBigMonster7
exp62: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster7 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=4 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp62/run{1} --master_seed={1} --output=False' ::: {0..100} 

exp63: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster7 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp63/run{1} --master_seed={1} --output=False' ::: {0..100} 
	
# env_name2=BanditBigMonster8
exp64: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster8 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=4 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp64/run{1} --master_seed={1} --output=False' ::: {0..100} 

exp65: 
	parallel -j 4 \
			--joblog '$(DATA_PATH)/exp2.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster8 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp65/run{1} --master_seed={1} --output=False' ::: {0..100} 

# env_name2=BanditBigMonster9
exp66: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=4 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp66/run{1} --master_seed={1} --output=False' ::: {0..100} 

exp67: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp67/run{1} --master_seed={1} --output=False' ::: {0..100} 

# env_name2=BanditBigMonster10
exp68: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=4 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp68/run{1} --master_seed={1} --output=False' ::: {0..100} 

exp69: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp69/run{1} --master_seed={1} --output=False' ::: {0..100} 
	


# ---------------------------------------------------------------------------
# 7/22/21
# 44465f3
#
# Try the reg. bandits with --parent_threshold=0.01 and a
# --kid_scale=4. 
#
# RESULTS: No benefit nor harm to --kid_scale=4 on the BanditDynamicRegMonster
#          
#          Run an oracle agent to get a sense of what is possible for 
#          improvement?

exp70: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=4 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp70/run{1} --master_seed={1} --output=False' ::: {0..100} 

exp71: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp71/run{1} --master_seed={1} --output=False' ::: {0..100} 


# ---------------------------------------------------------------------------
# 7/22/21
# 02aec5d
#
# Run an oracle exp on BanditDynamicRegMonster. What does best
# perfomance look like in terms of V and total_R?
#
# RESULTS: **exp72** Good news. The reason parpar and and parkid have been
#          hard to seperate is both are a oracle levels of performance
#          in both total_R and change_R
#
#          **exp73-74** Parkid w/ threshold and scale is nearing oracle 
#          performance on BanditBigMonster9 and BanditBigMonster10.
#          This is the reason scaling --kid_scale=2 to --kid_scale=4
#          had little effect.
#          Linear ungate parkid still marginally better than parpar, which
#          is good. (that is --parent_threshold=0.0 --kid_scale=1).
#
#          Compating --kid_scale=1 and --kid_scale=4 on BanditBigMonster9 
#          and BanditBigMonster10 suggests scaling has little downside, and
#          can in the case of BanditBigMonster10 bump change_R. Not sure
#          what the general story for scale will be. Another param is 
#          really worth it?
#
#          **exp75-77** all learning agents are clearly better than 
#          random. This was expected. I should have confirmed earlier.
#
#          In sum, parkid confirmed useful. Parkid w/ gating and 
#          all the more so.
exp72: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py oracle --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --lr_R=0.6 --log_dir=$(DATA_PATH)/exp72/run{1} --master_seed={1} --output=False' ::: {0..100} 

# Run an oracle exp on BanditBigMonster9. What does best
# perfomance look like in terms of V and total_R?
exp73: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py oracle --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp73/run{1} --master_seed={1} --output=False' ::: {0..100} 

# Run an oracle exp on BanditBigMonster10. What does best
# perfomance look like in terms of V and total_R?
exp74: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py oracle --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp74/run{1} --master_seed={1} --output=False' ::: {0..100} 
			
# ---
# 34f121e
#
# Run an random exp on BanditDynamicRegMonster. What does best
# perfomance look like in terms of V and total_R?

exp75: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py random --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --lr_R=0.6 --log_dir=$(DATA_PATH)/exp75/run{1} --master_seed={1} --output=False' ::: {0..100} 

# Run an oracle exp on BanditBigMonster9. What does best
# perfomance look like in terms of V and total_R?
exp76: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py random --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp76/run{1} --master_seed={1} --output=False' ::: {0..100} 

# Run an oracle exp on BanditBigMonster10. What does best
# perfomance look like in terms of V and total_R?
exp77: 
	parallel -j 4 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py random --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp77/run{1} --master_seed={1} --output=False' ::: {0..100} 



# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 7/23/21
# 5d42682

# ***Current biology results (v1). n=500.***

# ---
# Sumner analogs, with 'high' num. repeats. 
# - Based on exp20, exp21, and exp75

# BanditDynamicRegMonster series
exp78: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp78/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp79: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp79/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp80: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py oracle --num_episodes=80  --change=40 --env_name1=BanditStaticRegMonster --env_name2=BanditDynamicRegMonster --lr_R=0.6 --log_dir=$(DATA_PATH)/exp80/run{1} --master_seed={1} --output=False' ::: {0..500} 


# ... I accidently skipped some numbers here... oooops....

# ---
# BanditBigMonster* series. Begining with not-better changes, moving to 
# much better.
#
# Based on, extending, exp60-69. There is no gating, scaling, in this series. 
# For that, see the next one.

# env_name2=BanditBigMonster2
exp90: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster2 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp90/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp91: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster2 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp91/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster3
exp92: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster3 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp92/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp93: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster3 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp93/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster4
exp94: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster4 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp94/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp95: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster4 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp95/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster5
exp96: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster5 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp96/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp97: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster5 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp97/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster6
exp98: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster6 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp98/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp99: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster6 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp99/run{1} --master_seed={1} --output=False' ::: {0..500} 
	
# env_name2=BanditBigMonster7
exp100: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster7 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp100/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp101: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster7 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp101/run{1} --master_seed={1} --output=False' ::: {0..500} 
	
# env_name2=BanditBigMonster8
exp102: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster8 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp102/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp103: 
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp2.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster8 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp103/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster9
exp104: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp104/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp105: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp105/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster10
exp106: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.0 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp106/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp107: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp107/run{1} --master_seed={1} --output=False' ::: {0..500} 
	

# ---
# BanditBigMonster* series. Begining with not-better changes, moving to 
# much better.
#
# Based on, extending, exp60-69. There IS gating, scaling, in this series. 
# --parent_threshold=0.01 --kid_scale=2

# env_name2=BanditBigMonster2
exp108: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster2 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=2 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp108/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp109: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster2 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp109/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster3
exp110: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster3 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=2 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp110/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp111: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster3 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp111/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster4
exp112: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster4 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=2 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp112/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp113: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster4 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp113/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster5
exp114: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster5 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=2 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp114/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp115: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster5 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp115/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster6
exp116: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster6 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=2 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp116/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp117: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster6 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp117/run{1} --master_seed={1} --output=False' ::: {0..500} 
	
# env_name2=BanditBigMonster7
exp118: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster7 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=2 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp118/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp119: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster7 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp119/run{1} --master_seed={1} --output=False' ::: {0..500} 
	
# env_name2=BanditBigMonster8
exp120: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster8 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=2 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp120/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp121: 
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp2.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster8 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp121/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster9
exp122: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=2 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp122/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp123: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp123/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster10
exp124: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=2 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp124/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp125: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp125/run{1} --master_seed={1} --output=False' ::: {0..500} 
	

# ---
# BanditBigMonster* series. Begining with not-better changes, moving to 
# much better.
#
# Based on, extending, exp60-69. There IS gating, NO scaling, in this series. 
# --parent_threshold=0.01 --kid_scale=1

# env_name2=BanditBigMonster2
exp126: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster2 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp126/:run{1} --master_seed={1} --output=False' ::: {0..500} 

exp127: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster2 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp127/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster3
exp128: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster3 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp128/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp129: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster3 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp129/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster4
exp130: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster4 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp130/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp131: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster4 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp131/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster5
exp132: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster5 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp132/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp133: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster5 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp133/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster6
exp134: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster6 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp134/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp135: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster6 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp135/run{1} --master_seed={1} --output=False' ::: {0..500} 
	
# env_name2=BanditBigMonster7
exp136: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster7 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp136/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp137: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster7 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp137/run{1} --master_seed={1} --output=False' ::: {0..500} 
	
# env_name2=BanditBigMonster8
exp138: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster8 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp138/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp139: 
	parallel -j 39 \
			--joblog '$(DATA_PATH)/exp2.log' \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster8 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp139/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster9
exp140: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp140/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp141: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp141/run{1} --master_seed={1} --output=False' ::: {0..500} 

# env_name2=BanditBigMonster10
exp142: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parkid --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --par_boredom=0.01 --kid_boredom=0.0 --parent_threshold=0.01 --kid_scale=1 --set_point=40 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp142/run{1} --master_seed={1} --output=False' ::: {0..500} 

exp143: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py parpar --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --par_boredom=0.01 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp143/run{1} --master_seed={1} --output=False' ::: {0..500} 
	

# ----
# Oracles for the full monster series

# --env_name2=BanditBigMonster2
exp144: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py oracle --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster2 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp144/run{1} --master_seed={1} --output=False' ::: {0..500} 

# --env_name2=BanditBigMonster3
exp145: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py oracle --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster3 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp145/run{1} --master_seed={1} --output=False' ::: {0..500} 

# --env_name2=BanditBigMonster4
exp146: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py oracle --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster4 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp146/run{1} --master_seed={1} --output=False' ::: {0..500} 

# --env_name2=BanditBigMonster5
exp147: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py oracle --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster5 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp147/run{1} --master_seed={1} --output=False' ::: {0..500} 

# --env_name2=BanditBigMonster6
exp148: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py oracle --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster6 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp148/run{1} --master_seed={1} --output=False' ::: {0..500} 

# --env_name2=BanditBigMonster7
exp149: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py oracle --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster7 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp149/run{1} --master_seed={1} --output=False' ::: {0..500} 

# --env_name2=BanditBigMonster8
exp150: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py oracle --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster8 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp150/run{1} --master_seed={1} --output=False' ::: {0..500} 

# --env_name2=BanditBigMonster9
exp151: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py oracle --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp151/run{1} --master_seed={1} --output=False' ::: {0..500} 

# --env_name2=BanditBigMonster10
exp152: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py oracle --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp152/run{1} --master_seed={1} --output=False' ::: {0..500}


# ----
# Randoms for the full monster series

# --env_name2=BanditBigMonster2
exp153: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py random --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster2 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp153/run{1} --master_seed={1} --output=False' ::: {0..500} 

# --env_name2=BanditBigMonster3
exp154: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py random --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster3 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp154/run{1} --master_seed={1} --output=False' ::: {0..500} 

# --env_name2=BanditBigMonster4
exp155: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py random --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster4 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp155/run{1} --master_seed={1} --output=False' ::: {0..500} 

# --env_name2=BanditBigMonster5
exp156: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py random --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster5 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp156/run{1} --master_seed={1} --output=False' ::: {0..500} 

# --env_name2=BanditBigMonster6
exp157: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py random --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster6 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp157/run{1} --master_seed={1} --output=False' ::: {0..500} 

# --env_name2=BanditBigMonster7
exp158: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py random --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster7 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp158/run{1} --master_seed={1} --output=False' ::: {0..500} 

# --env_name2=BanditBigMonster8
exp159: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py random --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster8 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp159/run{1} --master_seed={1} --output=False' ::: {0..500} 

# --env_name2=BanditBigMonster9
exp160: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py random --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster9 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp160/run{1} --master_seed={1} --output=False' ::: {0..500} 

# --env_name2=BanditBigMonster10
exp161: 
	parallel -j 39 \
			--nice 19 --delay 0 --bar --colsep ',' --header : \
			'python parkid/run/change_bandits.py random --num_episodes=80  --change=40 --env_name1=BanditBigMonster1 --env_name2=BanditBigMonster10 --lr_R=0.6 --log_dir=$(DATA_PATH)/exp161/run{1} --master_seed={1} --output=False' ::: {0..500}
