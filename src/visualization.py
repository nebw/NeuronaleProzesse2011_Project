# -*- coding: utf-8 -*-
#import matplotlib.pyplot as plt
import scipy.io
import pylab as plt
import pyNN.neuron as p

def paint_network(save=True, spike_trains1=plt.zeros((5,0)), spike_trains2=plt.zeros((6,1)),save_name = 'test', show_winner=True, trial=-1, direction=-1,choosen_directions=plt.arange(6)+1):
        '''visualization of the network
        save : if True gives only the basic network, if false network with spike_trains1
        spike_trains1: input
        spike_trains2: output
        save_name: stores the specific network under this name:
        show_winner: marks the winning output neuron (the one with highest frequency
        trial and direction: which trial of the data set and which direction
        choosen directions: elem[0,1,2,3,4,5] which of the directions were trained and tested)'''
        
        fig = plt.figure()
        ax=plt.axes([0,0,1,1]);
        
        #input neurons
        for i in (plt.arange(5)+1):
            ax.add_patch(plt.Circle((0.4,(i*0.1)+0.2),radius=0.025,edgecolor='b',LineStyle='solid',facecolor='b',alpha=1));
        
        #connections
        for i in (plt.arange(5)+1):
            for j in (plt.arange(6)+1):
	        if ((j-1) in choosen_directions):
	            plt.plot([0.4, 0.6],[(i*0.1)+0.2,(((j*-1)+7)*0.1)+0.15],'k-')
        
        #input neurons
        for i in (plt.arange(5)+1):
            ax.add_patch(plt.Circle((0.4,(i*0.1)+0.2),radius=0.025,edgecolor='b',LineStyle='solid',facecolor='b',alpha=1));
        
        dx = [0, 0.017, 0.017, 0, -0.017, -0.017]
        dy = [0.02, 0.01, -0.01, -0.02, -0.01, 0.01]
        
        #output neurons
        for i in (plt.arange(6)+1):
	    if ((i-1) in choosen_directions):
                ax.add_patch(plt.Circle((0.6,(((i*-1)+7)*0.1)+0.15),radius=0.025,edgecolor='g',LineStyle='solid',facecolor='g',alpha=1));
                # direction indicated by an arrow
                plt.arrow(0.65,(((i*-1)+7)*0.1)+0.15,dx[i-1],dy[i-1],width=0.003,color='k')
                # number of direction
                plt.text(0.61, (((i*-1)+7)*0.1)+0.14, str(i))

        
        ax.set_yticks([])
        ax.set_xticks([]) 
        
        # stores the basic network
        if save:
            plt.ylim([0,1])
            plt.xlim([0.3,0.7])
            plt.text(0.57,0.85,'Output layer')
            plt.text(0.38,0.85,'Input layer')
            fig.savefig('network',format='png')    

        # stores the network with spike trains
        else:
            plt.ylim([0,1])
            plt.xlim([0,1])
            plt.text(0.55,0.82,'Output layer')
            plt.text(0.3,0.82,'Input layer')
            plt.text(0.4 ,0.9,'trial: '+str(trial)+' direction: '+str(direction))
            plt.arrow(0.65,0.9,dx[direction-1],dy[direction-1],width=0.003,color='k')
            
            # plot left spike trains
            for i in plt.arange(5):
                # change the times to -1000 to 1000     
                spike_trains1[i] = spike_trains1[i].astype(int)-1000
                plt.plot(((spike_trains1[i]+1000) / 10000.) +0.15, plt.ones((spike_trains1[i].shape[0],1)) * ((i*0.1)+0.3), 'k|')
            
            # get highest firing rate
            if show_winner:
                r_max=0
                for idx, i in enumerate(choosen_directions):
                    r_max = max(r_max, int(spike_trains2[idx].shape[0]/2.*100)/100.)
      


            # plot right spike trains and firing rate and winning neuron
            for i in plt.arange(6):
               if (i in choosen_directions):
                    # change the times to -1000 to 1000     
                    spike_trains2[choosen_directions.index(i)] = spike_trains2[choosen_directions.index(i)].astype(int)-1000
                    plt.plot(((spike_trains2[choosen_directions.index(i)] +1000)/ 10000.) +0.68, plt.ones((spike_trains2[choosen_directions.index(i)].shape[0],1)) * ((((i*-1)+5)*0.1)+0.25), 'k|')
                    r = spike_trains2[choosen_directions.index(i)].shape[0]/2.
                    r = int(r*100)/100.
                    plt.text(0.9, (((i*-1)+5)*0.1)+0.24, str(r)+'Hz')
                    if (show_winner and (r==r_max)):
                        plt.plot([0.89,0.89,0.99,0.99, 0.89],[(((i*-1)+5)*0.1)+0.23, (((i*-1)+5)*0.1)+0.27, (((i*-1)+5)*0.1)+0.27,(((i*-1)+5)*0.1)+0.23, (((i*-1)+5)*0.1)+0.23],'r-',linewidth=2)
            fig.savefig(save_name,format='png') 

if __name__ == '__main__':    
    paint_network()
    plt.show()
    
    