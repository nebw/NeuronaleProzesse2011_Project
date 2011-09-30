# -*- coding: utf-8 -*-
'''
Created on 28.09.2011

@author: matzkeh
'''
import scipy.io as scp
import numpy as np
import pylab as pl
import pyNN.neuron as p
import random
import copy
import visualization
import sys

#########CONFIG#######################################################################################################

runtime = 2000.
timestep = 0.1
maxchange = 0.001
#maxchange = 0.005
weight_change_limit = 1
weight_limit_min = 0.001
weight_limit_max = 1.
debug = False
#choosen_directions = [1,2,4,5]
choosen_directions = [4,5]
trial_sequence = range(0,30) 
test_sequence = range(30,35)
number_of_training_phases = 2
data_set_config = 'peak'
if data_set_config == 'peak':
    initial_weights = [6.4083333333333332, 16.208333333333332, 5.375, 22.5, 16.375]
elif data_set_config == 'average':
    initial_weights = [23.916666666666668, 13.416666666666666, 18.091666666666665, 5.083333333333333, 21.541666666666668]

#########/CONFIG######################################################################################################

def dPrint(str):
    '''print debug output only if debug flag is true'''
    
    if debug:
      print str

def get_sorted_spike_train(id, direction, trial_id):
    '''sort spike train times in order to suppress warnings'''
    
    sorted_spike_train = data[id][direction][trial_id]
    sorted_spike_train.sort()
    return sorted_spike_train
    

def create_input_list():
    ''' 5 input populations for the spike trains
    neuroID, direction, trial, spiketrain as list'''
    
    input_list = []
    input_list.append(p.Population(1, cellclass=p.SpikeSourceArray))
    input_list.append(p.Population(1, cellclass=p.SpikeSourceArray))
    input_list.append(p.Population(1, cellclass=p.SpikeSourceArray)) 
    input_list.append(p.Population(1, cellclass=p.SpikeSourceArray)) 
    input_list.append(p.Population(1, cellclass=p.SpikeSourceArray)) 
    return input_list

def set_input_list(input_list, direction, trial_id):
    '''load spike train times from data'''
  
    input_list[0].set('spike_times', get_sorted_spike_train(0, direction, trial_id))
    input_list[1].set('spike_times', get_sorted_spike_train(1, direction, trial_id))
    input_list[2].set('spike_times', get_sorted_spike_train(2, direction, trial_id))
    input_list[3].set('spike_times', get_sorted_spike_train(3, direction, trial_id))
    input_list[4].set('spike_times', get_sorted_spike_train(4, direction, trial_id))

def print_weight_list(weight_list):
    '''output current list of weights'''
  
    print "weight_list:"
    for i,val in enumerate(weight_list):
        print i,".",val
        
def load_data(data_set):
    '''load data from matlab files'''
    
    data_list = []

    if data_set == 'peak':
        data_list.append(scp.loadmat('../data_peak/bootstrap_joe093-3-C3-MO.mat'))
        data_list.append(scp.loadmat('../data_peak/bootstrap_joe108-7-C3-MO(1).mat'))
        data_list.append(scp.loadmat('../data_peak/bootstrap_joe112-5-C3-MO.mat'))
        data_list.append(scp.loadmat('../data_peak/bootstrap_joe112-6-C3-MO.mat'))
        data_list.append(scp.loadmat('../data_peak/bootstrap_joe145-4-C3-MO.mat'))
    elif data_set == 'average':
        data_list.append(scp.loadmat('../data_av/bootstrap_joe097-5-C3-MO.mat'))
        data_list.append(scp.loadmat('../data_av/bootstrap_joe108-4-C3-MO.mat'))
        data_list.append(scp.loadmat('../data_av/bootstrap_joe108-7-C3-MO.mat'))
        data_list.append(scp.loadmat('../data_av/bootstrap_joe147-1-C3-MO.mat'))
        data_list.append(scp.loadmat('../data_av/bootstrap_joe151-1-C3-MO.mat'))

    return data_list

#########INITIALIZATION##############################################################################################

p.setup(timestep = timestep)

data_list = load_data(data_set_config)

#convert data to internal data representation
data = []
for list in data_list:
    direction_list = []
    data.append(direction_list)
    for direction in range(0,6):
        trial_list = []
        direction_list.append(trial_list)
        for trial in range(1,36):
            if data_set_config == 'peak':
                trial_list.append(list['GDFcell'][0][direction][list['GDFcell'][0][direction][:,0] == trial][:,1])
            elif data_set_config == 'average':
                trial_list.append([elem + 1000 for elem in (list['GDFcell'][0][direction][list['GDFcell'][0][direction][:,0] == trial][:,1])])

# create direction population
dir_list = []
for i in range(0,len(choosen_directions)):
    dir_list.append(p.Population(1, cellclass=p.IF_cond_exp))
for elem in dir_list:
    elem.record()
    elem.record_v()

#random.shuffle(trial_sequence)
direction_sequence = copy.deepcopy(choosen_directions)
direction_sequence_sorted = copy.deepcopy(direction_sequence)

#initialize weights and weight counters
weight_list = []
for input in range(0,5):
    input_dir_list = []
    weight_list.append(input_dir_list)
    for dir in dir_list:
        input_dir_list.append(1/initial_weights[input])
       
weight_count_increases = pl.zeros_like(weight_list)
weight_count_decreases = pl.zeros_like(weight_list)

#initialize input_list
input_list = create_input_list()

output_firing_rates = [0] * len(choosen_directions)

all_firing_rates = [[],[],[],[],[],[]]
average_firing_rates = copy.deepcopy(initial_weights)

#########TRAINING####################################################################################################

flag_projection = True
proj_list = []
num_test = 0

while num_test < number_of_training_phases:
	num_test += 1
	
	for trial_id in trial_sequence:
	    #shuffle sequence of input directions in order to prevent bias
	    #random.shuffle(direction_sequence)
	    
	    for direction_id in direction_sequence:
		set_input_list(input_list, direction_id, trial_id)
		
		input_firing_rates = []
		for index, input in enumerate(data):
		    firing_rate = len(input[direction_id][trial_id]) / runtime * 1000.
		    input_firing_rates.append(firing_rate)
		    all_firing_rates[index].append(firing_rate)

		#initialize projections if flag_projection is set
		if(flag_projection):
		    flag_projection = False
		    for input_id, input in enumerate(input_list):
			input_dir_list = []
			proj_list.append(input_dir_list)
			for dir_id, dir in enumerate(dir_list):
			    input_dir_list.append(p.Projection(input,dir, target="excitatory", method=p.AllToAllConnector()))
			    input_dir_list[-1].setWeights(weight_list[input_id][dir_id])
		#load data for the projections
		else:
		    for input_id, input in enumerate(input_list):
			for dir_id, dir in enumerate(dir_list):
			    proj_list[input_id][dir_id].setWeights(weight_list[input_id][dir_id])
		
		#run for runtime ms
		p.run(runtime)
		
		#compute output firing rates
		output_firing_rates = []
		for i,elem in enumerate(dir_list):
		    output_firing_rates.append(len(elem.getSpikes() / runtime * 1000.))
		    
		dPrint('input firing rates: ' + str(input_firing_rates))
		dPrint('output firing rates: ' + str(output_firing_rates))
		if debug:
		   print_weight_list(weight_list)

		id =  direction_sequence_sorted.index(direction_id)
		highest_output_index = output_firing_rates.index(max(output_firing_rates))
		
		dPrint("Direction: " + str(direction_sequence_sorted.index(direction_id)))
		dPrint("Prediction: " + str(highest_output_index))

		dPrint("============================ resetting weights ============================")
		
		#iterate over input neurons
		for input_id, weight in enumerate(weight_list):
		  
		    #compute firing rate and deviation from mean of current input neuron
		    firing_rate = input_firing_rates[input_id]
		    value = firing_rate - average_firing_rates[input_id]
		    
		    #set limit for weight changes
		    value += weight_change_limit
		    
		    #only change weights if firing rate > average + weight_change_limit
		    if value > 0:
			#if prediction of current output neuron is correct
			if id == highest_output_index:
			    for syn_id, synapse in enumerate(weight):
				#increase synapse weight to correct output neuron
				if syn_id == highest_output_index:
				    weightchange = value * maxchange
				    newWeight = weight[syn_id] + weightchange
				    if debug:
				      weight_count_increases[input_id][syn_id] += weightchange
				#decrease synapse weight to incorrect output neurons
				else:
				    weightchange = -0.2 * value * maxchange
				    newWeight = weight[syn_id] + weightchange
				    if debug:
				      weight_count_decreases[input_id][syn_id] += weightchange
				#check if new weight is within limits
				if newWeight > weight_limit_min and newWeight < weight_limit_max:
				    weight[syn_id] = newWeight
			else:
			    for syn_id, synapse in enumerate(weight):
				if syn_id == highest_output_index:
				    weightchange = -2.0 * value * maxchange
				    newWeight = weight[syn_id] + weightchange
				    if debug:
				      weight_count_increases[input_id][syn_id] += weightchange
				else:
				    weightchange = 0.5 * value * maxchange
				    newWeight = weight[syn_id] + weightchange
				    if debug:
				      weight_count_decreases[input_id][syn_id] += weightchange
				if newWeight > weight_limit_min and newWeight < weight_limit_max:
				    weight[syn_id] = newWeight

		#compute average firing rate over all trials and directions per neuron
		for index, input in enumerate(data):
		    average_firing_rates[index] = np.mean(all_firing_rates[index])    
		
		if not debug:
		  print ('.'),
		  sys.stdout.flush()
		    
		p.reset()
		
print()
        
print("============================ training results ============================")
print "weight list:"
print_weight_list(weight_list)

if debug:
  print "weight_count_increases"
  for i,val in enumerate(weight_count_increases):
      print i,".",val
      
  print "weight_count_decreases"
  for i,val in enumerate(weight_count_decreases):
      print i,".",val
      
  print "weight_count_changes in total"
  for i,val in enumerate(weight_count_decreases+weight_count_increases):
      print i,".",val
print("============================ /training results ===========================")


#########TEST########################################################################################################

count_positives = 0.
count_negatives = 0.
count_test_inputs = 0.

flag_projection = True
for trial_id in test_sequence:

    direction_sequence = copy.deepcopy(choosen_directions)
    random.shuffle(direction_sequence)
    
    for direction_id in direction_sequence:
        set_input_list(input_list, direction_id, trial_id)
        print('============================ testing weights ============================')
        
        input_firing_rates = []
        input_spike_trains = []
        for input in data:
            input_firing_rates.append(len(input[direction_id][trial_id]) / runtime * 1000.)
            input_spike_trains.append(input[direction_id][trial_id])

	#run for runtime ms
        p.run(runtime)
        
        output_firing_rates = []
        output_spike_trains = []
        for i,elem in enumerate(dir_list):
            output_firing_rates.append(len(elem.getSpikes() / runtime * 1000.))
            output_spike_trains.append(elem.getSpikes())
            
        #visualize
        visualization.paint_network(False, input_spike_trains, output_spike_trains, '../output/' + data_set_config + '_' + str(choosen_directions) + '_test_network_trial'+str(trial_id)+'_direction_'+str(direction_id) + '.png',True, trial_id, direction_id+1,choosen_directions)    

        prediction = direction_sequence_sorted[output_firing_rates.index(max(output_firing_rates))]

        print "input firing rates: ", input_firing_rates
        print "resulting output firing rates: ", output_firing_rates
        print "input direction_id: ", direction_id
        print "prediction: " , prediction

        if(prediction == direction_id):
            count_positives += 1.
        else:
            count_negatives += 1.
        count_test_inputs += 1.
        p.reset()
    
print('')
print("============================ test results ===========================")
print "number of test inputs: ", count_test_inputs
print "number of positives: ", count_positives
print "number of negatives: ", count_negatives
print "positive predictions: ", count_positives /count_test_inputs * 100., " %"
print("============================ /test results ==========================")



