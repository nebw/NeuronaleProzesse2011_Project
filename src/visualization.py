# -*- coding: utf-8 -*-
#import matplotlib.pyplot as plt
import scipy.io
import pylab as plt
import pyNN.neuron as p

def paint_network(save=True, spike_trains1=plt.zeros((5,0)), spike_trains2=plt.zeros((6,1)),save_name = 'test'):
        fig = plt.figure()
        ax=plt.axes([0,0,1,1]);
        
        #input neurons
        for i in (plt.arange(5)+1):
            ax.add_patch(plt.Circle((0.4,(i*0.1)+0.2),radius=0.025,edgecolor='b',LineStyle='solid',facecolor='b',alpha=1));
        
        #connections
        for i in (plt.arange(5)+1):
            for j in (plt.arange(6)+1):
	        plt.plot([0.4, 0.6],[(i*0.1)+0.2,(j*0.1)+0.15],'k-')
        
        #input neurons
        for i in (plt.arange(5)+1):
            ax.add_patch(plt.Circle((0.4,(i*0.1)+0.2),radius=0.025,edgecolor='b',LineStyle='solid',facecolor='b',alpha=1));
        
        #output neurons
        for i in (plt.arange(6)+1):
            ax.add_patch(plt.Circle((0.6,(i*0.1)+0.15),radius=0.025,edgecolor='g',LineStyle='solid',facecolor='g',alpha=1));

        
        ax.set_yticks([])
        ax.set_xticks([]) 
        
        if save:
            plt.ylim([0,1])
            plt.xlim([0.3,0.7])
            plt.text(0.57,0.85,'Output layer')
            plt.text(0.38,0.85,'Input layer')
            fig.savefig('network',format='png')    

        else:
            plt.ylim([0,1])
            plt.xlim([0,1])
            plt.text(0.55,0.85,'Output layer')
            plt.text(0.3,0.85,'Input layer')
            
            # plot left spike trains
            for i in plt.arange(5):
                plt.plot(((spike_trains1[i]+1000) / 10000.) +0.15, plt.ones((spike_trains1[i].shape[0],1)) * ((i*0.1)+0.3), 'k|')

            # plot right spike trains
            for i in plt.arange(6):
                plt.plot(((spike_trains2[i] +1000)/ 10000.) +0.65, plt.ones((spike_trains2[i].shape[0],1)) * ((i*0.1)+0.25), 'k|')

            fig.savefig(save_name,format='png') 

if __name__ == '__main__':    
    paint_network()
    plt.show()
    
    