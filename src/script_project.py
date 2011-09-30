# -*- coding: utf-8 -*-
#import matplotlib.pyplot as plt
import scipy.io
import pylab as plt
import pyNN.neuron as p

def exercise1():
    ''' this function gives a visualization of the whole data set
    scatter plots for the different neurons and directions, as well as the mean spikes druing the timeline and the estimated fring rate (by a kernel convolution, window of width 100)''' 
    
    #load data1.
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
    
    # here the binary spike trains of the neurons are stored, for every neuron and every direction
    binary = plt.zeros((5, 6, 2000))
    # and here the spike times, for every neuron, every direction and every trial
    spike_times = [[[[]*1]*35]*6]*5
    for i in plt.arange(5): #neuron
        exec "%s=%s" % ('data', 'data' + str(i+1))
        fig = plt.figure()
        ax=plt.axes([0,0,1,1]);circ=plt.Circle((0.5,0.48),radius=0.051,edgecolor='k',LineStyle='solid',facecolor='w');ax.add_patch(circ);plt.text(0.453,0.465,'Neuron ' + str(i+1))
        ax.set_yticks([])
        ax.set_xticks([])        
        for j in plt.arange(6): #direction
              if j==0:
                ax=plt.axes([.3,.8,.3,.1])
              elif j==1:
                ax=plt.axes([.6,.55,.3,.1])
              elif j==2:
                ax=plt.axes([.6,.3,.3,.1])
              elif j==3:
                ax=plt.axes([.3,.1,.3,.1])
              elif j==4:
                ax=plt.axes([.1,.3,.3,.1])
              elif j==5:
                ax=plt.axes([.1,.55,.3,.1])
              data[j] = data[j].astype(int)
              data[j][:,1] = data[j][:,1]-1000
              for k in plt.arange(35)+1: #trials
                  indi = plt.find(data[j][:,0]==k)
                  spike_times[i][j][k-1] = data[j][indi,1]
                  for l in (spike_times[i][j][k-1]+999):
                      if l < 2000:
                          binary[i][j][l] += 1
                  # plots the scatter plots
                  plt.plot(data[j][indi,1], plt.ones(data[j][indi,1].shape[0])*k, 'b|')
              plt.title('direction'+str(j+1))
              plt.xlabel('time [ms]')    
              plt.ylabel('trial_id')
              plt.xlim([-1000,1000])
              for tick in ax.yaxis.get_major_ticks():
                 tick.label1.set_fontsize(8)
              for tick in ax.xaxis.get_major_ticks():
                 tick.label1.set_fontsize(8)
        fig.savefig('neuron'+str(i+1)+'spikes',format='png')

    
    # this is the definition of the kernel
    w = 100
    h = 1./w
    k = plt.ones((w,)) * h
    
    # normalization by the number of trials
    binary = binary/35.
    
    for i in plt.arange(5):
        fig = plt.figure()
        ax=plt.axes([0,0,1,1]);circ=plt.Circle((0.5,0.48),radius=0.051,edgecolor='k',LineStyle='solid',facecolor='w');ax.add_patch(circ);plt.text(0.453,0.465,'Neuron ' + str(i+1))
        ax.set_yticks([])
        ax.set_xticks([])  
        for j in plt.arange(6):
	      if j==0:
                ax=plt.axes([.3,.8,.3,.1])
              elif j==1:
                ax=plt.axes([.6,.55,.3,.1])
              elif j==2:
                ax=plt.axes([.6,.3,.3,.1])
              elif j==3:
                ax=plt.axes([.3,.1,.3,.1])
              elif j==4:
                ax=plt.axes([.1,.3,.3,.1])
              elif j==5:
                ax=plt.axes([.1,.55,.3,.1])
              # plots the mean spikes along the timeline of the experiment
              plt.plot(plt.arange(-1000,1000),binary[i][j])
              plt.ylabel('mean spikes')
              plt.hold(True)
              ax2=ax.twinx()
              
              # plots the estimated firing rate on a different y axis
              plt.plot(plt.arange(-1000,1000),plt.convolve(binary[i][j], k, mode='same'),'g')
              plt.title('direction'+str(j+1))
              plt.xlabel('time [ms]')    
              plt.ylabel('estimated r')
              for tick in ax.yaxis.get_major_ticks():
                 tick.label1.set_fontsize(8)
              for tick in ax2.yaxis.get_major_ticks():
                 tick.label2.set_fontsize(8)
              for tick in ax.xaxis.get_major_ticks():
                 tick.label1.set_fontsize(8)
        fig.savefig('neuron'+str(i+1)+'rates',format='png')     
                
           


if __name__ == '__main__':    
    exercise1()
    plt.show()