#importing libraries
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

hdf5_path = 'data/dataset.hdf5'

def ReadFile(hdf5_path,split=False):
    '''
    reads in the hdf5-file and returns the training data-set and training labels.
    Can split them if required
    '''
    with h5py.File(hdf5_path) as file:
        x_data = file['train_data'][:,:,:,:,:]
        y_data = file['train_labels'][:]
        
    if split:
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, train_size=0.6, random_state=12345)

        return x_train, x_test, y_train, y_test

    else:
        return x_data, y_data
    
def GetRandomSamples(samples):
    '''
    returns the requested amount of random samples
    '''
    
    with h5py.File(hdf5_path) as file:
        train_labels = file['train_labels'][:]
    
    return train_labels[np.random.randint(low=0,high=train_labels.size,size=samples)]       
        
def GetHydroSamples(hydro,samples):
    '''
    returns random amount of samples with the demanded hydro-label in a numpy array
    '''
    
    with h5py.File(hdf5_path) as file:
        train_labels = file['train_labels'][:]
        train_labels = np.where(train_labels == hydro)[0]

    return train_labels[np.random.randint(low=0, high=train_labels.size, size=samples)]

def TwoDimHeatmap(data,z,c,**kwargs):
    '''
    2D-heatmap in the X,Y-plane of the defined channel and Z-coordinate.
    '''
    data = data[:,:,z,c]
    if 'ax' not in kwargs.keys():
        fig,ax = plt.subplots(figsize=(7,7))
    else:
        ax = kwargs['ax']
        
    ax.set_title('z='+str(z)+',channel='+str(c))
    ax.imshow(data)
    
def AnalyseSimulation(sample,z,c=1,subtraction=False,histogram=False):
    '''
    Does an entire analysis of a simulation in the z-plane
    '''
    #read in the data
    with h5py.File(hdf5_path) as file:
        train_sample = file['train_data'][sample,:,:,:,:]
        hydro_value = file['train_labels'][sample]

    print('> HYDRATION LEVEL: ',hydro_value)    
    #make the 3 different axis for the channels
    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,10))
    axes = (ax1,ax2,ax3)
    
    #iterate over the axes and plot a channel on them
    for x,y in enumerate(axes):
        TwoDimHeatmap(train_sample,z,x,ax=y)
    plt.show()
    
    #plotting the subtractions. Can be turned on if requested
    if subtraction:
        for channel in range(3):
            print('> DIFFERENCE BETWEEN Z AND Z+1, CHANNEL' , channel)
            fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,10))
            TwoDimHeatmap(train_sample,z,1,ax=ax1)
            TwoDimHeatmap(train_sample,z+1,1,ax=ax2)

            ax3.imshow(np.absolute(np.subtract(train_sample[:,:,z,channel],train_sample[:,:,z+1,channel])))

            ax1.set_title(str('z='+str(z)+',channel=')+str(channel))
            ax2.set_title('z='+str(z+1)+',channel='+str(channel))
            ax3.set_title('subtraction & abs. value')

            plt.show()
    
    #plotting histograms. Can be turned on if requested
    if histogram:
        print('> COUNTING VALUES IN DIFFERENT CHANNELS')
        fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))
        axes = (ax1,ax2,ax3)
        for x,ax in enumerate(axes):
            arr = train_sample[:,:,:,x].flatten()
            minim = arr.min()
            maxim = arr.max()
            ticks = np.linspace(minim,maxim,10)

            ax.grid(True)
            ax.set_title('channel '+str(x))
            ax.set_xticks(ticks)
            ax.tick_params(axis ='x', rotation = 45)
            ax.set_xlabel('values')
            ax.set_ylabel('counts')
            ax.hist(arr)
        plt.tight_layout()
        plt.show()
        
def MultipleAnalysis(hydro,samps):
    '''
    Do full analysis on random subset of demanded hydro-level.    
    '''
    test = GetHydroSamples(hydro,samps)
    for sample in test:
        print('> SAMPLE NUMBER: ',sample)
        AnalyseSimulation(sample,15)

def Scale(x_train,x_test=None):
    '''
    Scales the x_train and x_test with the mean and std from x_train
    '''
    avr = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train,axis=(0,1,2,3))
    
    x_train_scaled = (x_train - avr)/std

    if x_test is not None:
        x_test_scaled = (x_test - avr)/std

        return x_train_scaled,x_test_scaled

    else:
        return x_train_scaled

def Rotate(sample,axis):
    '''
    Can mirror a sample in accordance with a defined axis.
    *When using axis = 0: mirrored over y-axis
    *when using axis = 1: mirrored over x-axis
    *when using axis = 2: mirrored over z-axis
    *when using axis = 3: mirrored over channel => lets not do that :D
    '''
    return np.flip(sample,axis=axis)

def RandomRotateSamples(data):
    '''
    Randomly rotates each sample of the dataset over a x,y,z, 
    defined by a random generated vector (seeded)
    '''
    np.random.seed(12345)
    length = data.shape[0]
    randomvector = np.random.randint(0,3,length)
    res = list()
    for x in range(length):
        intermediate = Rotate(data[x],axis=randomvector[x])
        
        #correct up to here. should find something to keep it
        #the same format
        #somehow workes when using list instead and then converting
        #back to an array
        res.append(intermediate)
    return np.asarray(res), randomvector