

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

def get_proportional_weight(mean, firing_rate, maxdiff):
    return ((firing_rate - mean) * maxchange / maxdiff)

def get_sorted_spike_train(id, direction, trial_id):
    #print "sort: ", id, ", ", direction,", ", trial_id
    sorted_spike_train = data[id][direction][trial_id]
    sorted_spike_train.sort()
    return sorted_spike_train
    

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

runtime = 2000.
timestep = 0.1
maxchange = 0.02

p.setup(timestep = timestep)
choosed_directions = [0,3]

data_list = []
data_list.append(scp.loadmat('bootstrap_joe093-3-C3-MO.mat'))
data_list.append(scp.loadmat('bootstrap_joe108-7-C3-MO(1).mat'))
data_list.append(scp.loadmat('bootstrap_joe112-5-C3-MO.mat'))
data_list.append(scp.loadmat('bootstrap_joe112-6-C3-MO.mat'))
data_list.append(scp.loadmat('bootstrap_joe145-4-C3-MO.mat'))

#data_list = []
#data_list.append(scp.loadmat('data2/bootstrap_joe097-5-C3-MO.mat'))
#data_list.append(scp.loadmat('data2/bootstrap_joe108-4-C3-MO.mat'))
#data_list.append(scp.loadmat('data2/bootstrap_joe108-7-C3-MO.mat'))
#data_list.append(scp.loadmat('data2/bootstrap_joe147-1-C3-MO.mat'))
#data_list.append(scp.loadmat('data2/bootstrap_joe151-1-C3-MO.mat'))

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
for i in range(0,len(choosed_directions)):
    dir_list.append(p.Population(1, cellclass=p.IF_cond_exp))
for elem in dir_list:
    elem.record()
    elem.record_v()

trial_sequence = range(0,30) 

random.shuffle(trial_sequence)
direction_sequence = copy.deepcopy(choosed_directions)
direction_sequence_sorted = copy.deepcopy(direction_sequence)

count_increase_weight = 0
count_decrease_weight = 0

#weight_list[input][direction]
weight_list = []
for input in range(0,5):
    input_dir_list = []
    weight_list.append(input_dir_list)
    for dir in dir_list:
        input_dir_list.append(0.04 + random.random()*0.01)
       
weight_count_increases = pl.zeros_like(weight_list)
weight_count_decreases = pl.zeros_like(weight_list)


#initialize input_list
input_list = create_input_list()
output_firing_rates = [0] * len(choosed_directions)

average_firing_rates = [0,0,0,0,0]


flag_projection = True
proj_list = []
for trial_id in trial_sequence:
    random.shuffle(direction_sequence)
    for direction_id in direction_sequence:
        set_input_list(input_list, direction_id, trial_id)
        
        input_firing_rates = []
        for index, input in enumerate(data):
            firing_rate = len(input[direction_id][trial_id]) / runtime * 1000.
            input_firing_rates.append(len(input[direction_id][trial_id]) / runtime * 1000.)
            average_firing_rates[index] = np.mean([average_firing_rates[index], firing_rate])
#            print "Trial: ", trial_id
#            print "Direction: ", direction_id
#            print "Firing rate: ", input_firing_rates[-1]

        if(flag_projection):
            flag_projection = False
#            proj_list = []
            for input_id, input in enumerate(input_list):
                input_dir_list = []
                proj_list.append(input_dir_list)
                for dir_id, dir in enumerate(dir_list):
                    input_dir_list.append(p.Projection(input,dir, target="excitatory", method=p.AllToAllConnector()))
                    #weight_list von vorherigem durchlauf wird geladen
                    input_dir_list[-1].setWeights(weight_list[input_id][dir_id])
#            print "weight_list:"
#            for i,val in enumerate(weight_list):
#                print i,".",val
#     
        else:
            for input_id, input in enumerate(input_list):
                for dir_id, dir in enumerate(dir_list):
                    proj_list[input_id][dir_id].setWeights(weight_list[input_id][dir_id])
#                    print ("proj_list.getWeights()[%i][%i] = %f" , (input_id,dir_id,proj_list[input_id][dir_id].getWeights()))
#            print "weight_list:"
#            for i,val in enumerate(weight_list):
#                print i,".",val
        
        p.run(runtime)
        
        old_output_firing_rates = copy.deepcopy(output_firing_rates)
        output_firing_rates = []
#        print "elem.getSpikes()"
        for i,elem in enumerate(dir_list):
#            print(elem.getSpikes())
       #     print "old: ",old_output_firing_rates[i],", ", len(old_output_firing_rates)
        #    print(len(elem.getSpikes()))
            output_firing_rates.append(( len(elem.getSpikes() - old_output_firing_rates[i]) / runtime * 1000.))

        print "unsere: " , output_firing_rates
        
        mean_fire = pl.average(input_firing_rates)
        max_fire = max(input_firing_rates)
        max_diff = max_fire - mean_fire
        
        
        print("============================ resetting weights ==========")
#        print "input_firing_rates", input_firing_rates
#        print("mean: %f" % mean_fire)
        # reset the weights for woooooow effect
        for input_id,input in enumerate(weight_list):
#            for dir_id, dir in enumerate(input):
            firing_rate = input_firing_rates[input_id]
            value = firing_rate - average_firing_rates[input_id]
            highest_output_index = output_firing_rates.index(max(output_firing_rates))
            id = direction_sequence_sorted.index(direction_id)
            
            # uebereinstimmung von input direction mit dem staerksten feuernden output staerken
            if (highest_output_index == id):
                
                calculation = weight_list[input_id][highest_output_index] + get_proportional_weight(mean_fire, firing_rate, max_diff) *0.01# abs(value) * 0.01
                if (calculation < 0. or calculation >= 1.):
                    continue
                else:
                    weight_list[input_id][id] = calculation
                    weight_count_increases[input_id][id] += weight_list[input_id][id]
                    print "increase value",  get_proportional_weight(mean_fire, firing_rate, max_diff)*0.1
                    print("increase weight_list[%i][%i] = %f" %(input_id,id,weight_list[input_id][id]))
                    count_increase_weight +=1
            # wenn das am staerksten feuernde output neuron nicht stimmt, senken wir das gewicht
            elif(highest_output_index != id):
                calculation = weight_list[input_id][highest_output_index] + get_proportional_weight(mean_fire, firing_rate, max_diff) *0.1#abs(value) * 5
                if (calculation < 0. or calculation >= 1.):
                    continue
                else: 
                    weight_list[input_id][highest_output_index] = calculation
                    weight_count_decreases[input_id][highest_output_index] -= weight_list[input_id][highest_output_index]
                    print "decrease value",  get_proportional_weight(mean_fire, firing_rate, max_diff)*0.1
                    print("decrease weight_list[%i][%i] = %f" %(input_id,highest_output_index,weight_list[input_id][highest_output_index]))
                    count_decrease_weight += 1
                

#
#        
        
#        print("============================ resetting weights ==========")
##        print "input_firing_rates", input_firing_rates
##        print("mean: %f" % mean_fire)
#        # reset the weights for woooooow effect
#        for input_id,input in enumerate(weight_list):
#            for dir_id, dir in enumerate(input):
#                firing_rate = input_firing_rates[input_id]
#                value = firing_rate - average_firing_rates[input_id]
##                print("weight_list[%i][%i] = %f" %(input_id,dir_id,weight_list[input_id][dir_id]))
##                import pdb; pdb.set_trace()
##                print("input_firing_rates[%i] = %f" % (dir_id,input_firing_rates[input_id]))
##                print("value = %f" % value)
##                print "dir_id == direction_id?", dir_id == direction_id
##                print "value > 0? ", value > 0
#
##                import pdb; pdb.set_trace()
#                if (direction_sequence_sorted[dir_id] == direction_id and value > 0 and weight_list[input_id][dir_id] < 0.99 and weight_list[input_id][dir_id] > 0.002):
##                    weight_list[input_id][dir_id] += weight_list[input_id][dir_id] * value / 100.
##                    import pdb; pdb.set_trace()
#                    weight_list[input_id][dir_id] += get_proportional_weight(mean_fire, firing_rate, max_diff)
#                    weight_count_increases[input_id][dir_id] += 1
#                    print("increase weight_list[%i][%i] = %f" %(input_id,dir_id,weight_list[input_id][dir_id]))
##                    print("increase weight: %f" % weight_list[input_id][dir_id])
#                    count_increase_weight +=1
##                    elif(direction_sequence_sorted[dir_id] == direction_id and value < 0 and weight_list[input_id][dir_id] > 0.0002):
#                elif(direction_sequence_sorted[dir_id] != direction_id and value > 0 and weight_list[input_id][dir_id] > 0.002 and weight_list[input_id][dir_id] < 0.99):
##                    weight_list[input_id][dir_id] -= weight_list[input_id][dir_id] * value / 200.
#                    weight_list[input_id][dir_id] += get_proportional_weight(mean_fire, firing_rate, max_diff)
#                    weight_count_decreases[input_id][dir_id] -= 1
#                    print("decrease weight_list[%i][%i] = %f" %(input_id,dir_id,weight_list[input_id][dir_id]))
#                    count_decrease_weight +=1



#                print()
#        print("==========================================================")
#        print "direction_id: ", direction_id
#        print "resulting output firing rates: ", output_firing_rates
         
#        print firing_rates
        
        p.reset()
        
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print "weight_list: " , weight_list
print "count_increase_weight: ", count_increase_weight
print "count_decrease_weight: ", count_decrease_weight


####################################################################################
# test data

count_positives = 0.
count_negatives = 0.
count_test_inputs = 0.

test_sequence = range(30,35)
flag_projection = True
for trial_id in test_sequence:

    direction_sequence = copy.deepcopy(choosed_directions)

    random.shuffle(direction_sequence)
    for direction_id in direction_sequence:
#        print direction_id, ", ", trial_id
        set_input_list(input_list, direction, trial_id)
        print "#############################################"
        input_firing_rates = []
        for input in data:
            input_firing_rates.append(len(input[direction_id][trial_id]) / runtime * 1000.)
#            print "Trial: ", trial_id
#            print "Direction: ", direction_id
#            print "Firing rate: ", input_firing_rates[-1]

#        if(flag_projection):
#            flag_projection = False
##            proj_list = []
#            for input_id, input in enumerate(input_list):
#                input_dir_list = []
#                proj_list.append(input_dir_list)
#                for dir_id, dir in enumerate(dir_list):
#                    input_dir_list.append(p.Projection(input,dir, target="excitatory", method=p.AllToAllConnector()))
#                    #weight_list von traingsdurchlauf wird geladen
#                    input_dir_list[-1].setWeights(weight_list[input_id][dir_id])
#                    proj_list[input_id][dir_id].setWeights(weight_list[input_id][dir_id])

#        else:
#            for input_id, input in enumerate(input_list):
#                for dir_id, dir in enumerate(dir_list):
#                    proj_list[input_id][dir_id].setWeights(weight_list[input_id][dir_id])


#                    print ("proj_list.getWeights()[%i][%i] = %f" , (input_id,dir_id,proj_list[input_id][dir_id].getWeights()))
#            print "weight_list:"
#            for i,val in enumerate(weight_list):
#                print i,".",val
        
        p.run(runtime)
        
        old_output_firing_rates = copy.deepcopy(output_firing_rates)
        output_firing_rates = []

        for i,elem in enumerate(dir_list):
            output_firing_rates.append(( len(elem.getSpikes() - old_output_firing_rates[i]) / runtime * 1000.))


        print "input direction_id: ", direction_id
        print "resulting output firing rates: ", output_firing_rates
        print "max: ", max(output_firing_rates)
        print "index: ", output_firing_rates.index(max(output_firing_rates))
        print "direction_sequence: " , direction_sequence_sorted
        prediction = direction_sequence_sorted[output_firing_rates.index(max(output_firing_rates))]
        print "prediction: " , prediction
        # fehler: vertauscht hier irgendwie noch was mit den indizes ?!
        if(prediction == direction_id):
            count_positives += 1.
        else:
            count_negatives += 1.
        count_test_inputs += 1.
        p.reset()
    
print "number of test inputs: ", count_test_inputs
print "number of positives: ", count_positives
print "number of negatives: ", count_negatives
print "positive predictions: ", count_positives /count_test_inputs * 100., " %"

print "weight_list:"
for i,val in enumerate(weight_list):
    print i,".",val
    
    
print "weight_count_increases"
for i,val in enumerate(weight_count_increases):
    print i,".",val
    
print "weight_count_decreases"
for i,val in enumerate(weight_count_decreases):
    print i,".",val
    
print "weight_count_changes in total"
for i,val in enumerate(weight_count_decreases+weight_count_increases):
    print i,".",val
