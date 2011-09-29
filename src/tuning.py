# -*- coding: utf-8 -*-
import scipy.io as scp
import numpy as np
import pylab as pl
import pyNN.neuron as p
import random
import copy

def create_input_list():
    ''' 5 input populations for the spike trains
    neuroID, richtung, trial, spiketrain as list'''
    
    input_list = []
    input_list.append(p.Population(1, cellclass=p.SpikeSourceArray))
    input_list.append(p.Population(1, cellclass=p.SpikeSourceArray))
    input_list.append(p.Population(1, cellclass=p.SpikeSourceArray)) 
    input_list.append(p.Population(1, cellclass=p.SpikeSourceArray)) 
    input_list.append(p.Population(1, cellclass=p.SpikeSourceArray)) 
    return input_list
    
def set_input_list(input_list, direction, trial_id):
    input_list[0].set('spike_times', get_sorted_spike_train(0, direction, trial_id))
    input_list[1].set('spike_times', get_sorted_spike_train(1, direction, trial_id))
    input_list[2].set('spike_times', get_sorted_spike_train(2, direction, trial_id))
    input_list[3].set('spike_times', get_sorted_spike_train(3, direction, trial_id))
    input_list[4].set('spike_times', get_sorted_spike_train(4, direction, trial_id))
    
def get_sorted_spike_train(id, direction, trial_id):
    #print "sort: ", id, ", ", direction,", ", trial_id
    sorted_spike_train = data[id][direction][trial_id]
    sorted_spike_train.sort()
    return sorted_spike_train

data_list = []
data_list.append(scp.loadmat('../data/bootstrap_joe093-3-C3-MO.mat'))
data_list.append(scp.loadmat('../data/bootstrap_joe108-7-C3-MO(1).mat'))
data_list.append(scp.loadmat('../data/bootstrap_joe112-5-C3-MO.mat'))
data_list.append(scp.loadmat('../data/bootstrap_joe112-6-C3-MO.mat'))
data_list.append(scp.loadmat('../data/bootstrap_joe145-4-C3-MO.mat'))

input_list = create_input_list()
    
direction_sequence = [0,1,2,3,4,5]
trial_sequence = range(0,30) 

data = []
for list in data_list:
    direction_list = []
    data.append(direction_list)
    for direction in range(0,6):
        trial_list = []
        direction_list.append(trial_list)
        for trial in range(1,36):
            trial_list.append(list['GDFcell'][0][direction][list['GDFcell'][0][direction][:,0] == trial][:,1])

# create direction population
dir_list = []
for i in range(0,len(direction_sequence)):
    dir_list.append(p.Population(1, cellclass=p.IF_cond_exp))

for elem in input_list:
    elem.record()
    
DirFirRateList = []
for population in input_list:
  DirFirRateList.append([])
  for direction_id in direction_sequence:
    DirFirRateList[-1].append(0)
  
    
trial = 0
for trial_id in trial_sequence[0:5]:
    for dir_index, direction_id in enumerate(direction_sequence):
      set_input_list(input_list, direction_id, trial_id)
      p.run(2000)
      trial += 1
      for index, population in enumerate(input_list):
	DirFirRateList[index][dir_index] += len(population.getSpikes())
	
print trial
for dfList in DirFirRateList:
    print dfList
    print sum(dfList)
    print sum(dfList) / 5 / 6
    print "==================="
