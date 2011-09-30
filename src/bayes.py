# -*- coding: utf-8 -*-
#import matplotlib.pyplot as plt
import scipy.io
import pylab as plt
import pyNN.neuron as p

def exercise():
    # 1.
    a1 = scipy.io.loadmat('bootstrap_joe093-3-C3-MO.mat')
    a2 = scipy.io.loadmat('bootstrap_joe108-7-C3-MO.mat')
    a3 = scipy.io.loadmat('bootstrap_joe112-5-C3-MO.mat')
    a4 = scipy.io.loadmat('bootstrap_joe112-6-C3-MO.mat')
    a5 = scipy.io.loadmat('bootstrap_joe145-4-C3-MO.mat')
    data1 = a1["GDFcell"][0]
    data2 = a2["GDFcell"][0]
    data3 = a3["GDFcell"][0]
    data4 = a4["GDFcell"][0]
    data5 = a5["GDFcell"][0]
    
    binary = plt.zeros((5, 6, 2000))
    spike_times = [[[[]*1]*35]*6]*5
    for i in plt.arange(5): #neuron
        exec "%s=%s" % ('data', 'data' + str(i+1))
        for j in plt.arange(6): #direction
              data[j] = data[j].astype(int)
              data[j][:,1] = data[j][:,1]-1000
              for k in plt.arange(35)+1: #trials
                  indi = plt.find(data[j][:,0]==k)
                  spike_times[i][j][k-1] = data[j][indi,1]
                  for l in (spike_times[i][j][k-1]+999):
                      if l < 2000:
                          binary[i][j][l] += 1



if __name__ == '__main__':    
    exercise()
    plt.show()