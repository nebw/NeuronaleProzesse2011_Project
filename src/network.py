'''
Created on 28.09.2011

@author: matzkeh
'''
import scipy.io as scp
import numpy as np
import pylab as pl
import pyNN.neuron as p
import random

def create_input_list(direction, trial_id):
    ''' 5 input populations for the spike trains
    neuroID, richtung, trial, spiketrain as list'''
    
    input_list = []
    input_list.append(p.Population(1, cellclass=p.SpikeSourceArray, 
                            cellparams={'spike_times':data[0][direction][trial_id]}))
    input_list.append(p.Population(1, cellclass=p.SpikeSourceArray, 
                            cellparams={'spike_times':data[1][direction][trial_id]}))
    input_list.append(p.Population(1, cellclass=p.SpikeSourceArray, 
                            cellparams={'spike_times':data[2][direction][trial_id]}))
    input_list.append(p.Population(1, cellclass=p.SpikeSourceArray, 
                            cellparams={'spike_times':data[3][direction][trial_id]}))
    input_list.append(p.Population(1, cellclass=p.SpikeSourceArray, 
                            cellparams={'spike_times':data[4][direction][trial_id]}))
    return input_list

runtime = 2000.
timestep = 0.1

p.setup(timestep = timestep)

data_list = []
data_list.append(scp.loadmat('bootstrap_joe093-3-C3-MO.mat'))
data_list.append(scp.loadmat('bootstrap_joe108-7-C3-MO(1).mat'))
data_list.append(scp.loadmat('bootstrap_joe112-5-C3-MO.mat'))
data_list.append(scp.loadmat('bootstrap_joe112-6-C3-MO.mat'))
data_list.append(scp.loadmat('bootstrap_joe145-4-C3-MO.mat'))

data = []

for list in data_list:
    direction_list = []
    data.append(direction_list)
    for direction in range(0,6):
        trial_list = []
        direction_list.append(trial_list)
        for trial in range(1,31):
            trial_list.append(list['GDFcell'][0][direction][list['GDFcell'][0][direction][:,0] == trial][:,1])

#print(data_list[0]['GDFcell'][0][1])

dir_list = []

dir_list.append(p.Population(1, cellclass=p.IF_cond_exp))
dir_list.append(p.Population(1, cellclass=p.IF_cond_exp))

trial_sequence = range(0,30)
random.shuffle(trial_sequence)
direction_sequence = [1,5]

#weight_list[input][direction]
weight_list = []
for input in range(0,5):
    input_dir_list = []
    weight_list.append(input_dir_list)
    for dir in dir_list:
        input_dir_list.append(0.8)
        
for trial_id in trial_sequence:
    random.shuffle(direction_sequence)
    for direction_id in direction_sequence:
        input_list = create_input_list(direction_id, trial_id)
        
        input_firing_rates = []
        for input in data:
            input_firing_rates.append(len(input[direction_id][trial_id]) / runtime * 1000.)
            print "Trial: ", trial_id
            print "Direction: ", direction_id
            print "Firing rate: ", input_firing_rates[-1]

        proj_list = []
        for input_id, input in enumerate(input_list):
            input_dir_list = []
            proj_list.append(input_dir_list)
            for dir_id, dir in enumerate(dir_list):
                input_dir_list.append(p.Projection(input,dir, method=p.AllToAllConnector()))
                #weight_list von vorherigem durchlauf wird geladen
                #import pdb; pdb.set_trace()
                print len(input_dir_list), input_id, direction_id
                input_dir_list[-1].setWeights(weight_list[input_id][dir_id]) 

        print weight_list
        
        for elem in dir_list:
            elem.record()
            elem.record_v()
        
        p.run(runtime)

        output_firing_rates = []
        for elem in dir_list:
            print(elem.getSpikes())
            output_firing_rates.append(len(elem.getSpikes()) / runtime * 1000.)
        
        mean_fire = pl.average(input_firing_rates)
        
        # reset the weights for woooooow effect
        for input_id,input in enumerate(weight_list):
            for dir_id, dir in enumerate(input):
                value = input_firing_rates[dir_id] - mean_fire
                if (dir_id == direction_id and value > 0):
                    weight_list[input_id][dir_id] += weight_list[input_id][dir_id] * value / 2.
                elif(dir_id != direction_id and value > 0):
                    weight_list[input_id][dir_id] -= weight_list[input_id][dir_id] * value / 2.
                    
        print direction_id
        print output_firing_rates
         
#        print firing_rates
        
        p.reset()
    
