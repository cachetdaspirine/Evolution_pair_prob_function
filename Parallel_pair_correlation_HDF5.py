import numpy as np
import multiprocessing as mp
import math
from scipy.spatial import distance_matrix
from multiprocessing import shared_memory
import sys
import queue
import copy
import tables as pt
sys.path.append('/home/hcleroy/PostDoc/aging_condensates/Simulation/Gillespie/Gillespie_backend/')
sys.path.append('/home/hugo/PostDoc/aging_condensates/Gillespie/Gillespie_backend/')
import Gillespie_backend as gil
sys.path.append('/home/hcleroy/PostDoc/aging_condensates/Simulation/Gillespie/Analysis/')
sys.path.append('/home/hugo/PostDoc/aging_condensates/Gillespie/Analysis/')
from ToolBox import *
def histogram_float(*args, **kwargs):
    counts, bin_edges = np.histogram(*args, **kwargs)
    return counts.astype(float), bin_edges

def compute_cumulative_distribution(counts, num_bins, max_distance):
    # Calculate bin edges
    bin_edges = np.linspace(0, max_distance, num_bins + 1)

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate bin widths
    bin_widths = np.diff(bin_edges)

    # Calculate shell volumes
    shell_volumes = (4 / 3) * np.pi * ((bin_centers + bin_widths)**3 - bin_centers**3)

    # Calculate probabilities by multiplying counts (probability densities) by shell volumes
    probabilities = counts * shell_volumes

    # Calculate cumulative probabilities
    cumulative_prob = np.cumsum(probabilities)

    return bin_centers, cumulative_prob


def Compute_Pair_Correlation_Function(gillespie,output,group_name,step_tot,check_steps,num_bins,max_distance,linked=False):
    """
    Compute the Pair correlation function of a gillespie system. 
    parameters:
    output (multiprocessing.queue) : shared queue, to pass the result to a unique function that manage the writting in a file.
    group_name (string) : name of the group in a hdf5 file
    step_tot (int) : total time step of the simulation
    check_steps (int) : steps of computation of the pair correlation function
    num_bins (int) : number of bins used to compute the pair correlation function
    max_distance : distance max to compute the pair correlation function
    """
    counts, bin_edges = histogram_float([], bins=num_bins, range=(0, max_distance))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    shell_volumes = (4 / 3) * np.pi * ((bin_centers + bin_widths)**3 - bin_centers**3)

    current_time=0
    for i in range(step_tot//check_steps):
        counts, bin_edges = histogram_float([], bins=num_bins, range=(0, max_distance))
        prev_hist = np.zeros(counts.shape,dtype=float)
        t_tot = 0.
        for t in range(check_steps):
            move, time = gillespie.evolve()
            t_tot +=  time[0]
            counts += prev_hist * time[0]
            if linked :
                dist = np.linalg.norm(gillespie.get_R()[1:]-gillespie.get_R()[:-1],axis=1)
            else:
                dist_matrix = distance_matrix(gillespie.get_R(),gillespie.get_R())
                dist = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
            prev_hist, bin_edges = histogram_float(dist, bins=num_bins, range=(0, max_distance))
        counts = counts / (t_tot * shell_volumes*dist.shape[0])
        current_time+=t_tot
        output.put(('create_array',('/'+group_name,'step_'+str(i),np.stack((bin_centers,counts), axis=-1)),current_time))

def  Run(inqueue,output,step_tot,check_steps,num_bins,max_distance,linked):
    # simulation_name is a "f_"+float.hex() 
    """
    Each run process fetch a set of parameters called args, and run the associated simulation until the set of arg is empty.
    The simulation consists of evolving the gillespie, every check_steps it checks if the entropy of the system is close enough
    to a given entropy function. If it is the case it adds the position of the linkers associated to this state + the value of the entropy
    and the time associated to this measurement. the position of the linkers is a (Nlinker,3) array to which we add the value of the
    entropy S, and time t as [S, Nan, Nan], and [t,Nan,nan].
    parameters:
    inqueue (multiprocessing.queue) : each entry of q is  a set of parameters associated with a specific gillespie simulation.
    output (multiprocessing.queue) : it just fetch the data that has to be outputed inside this queue
    step_tot (int) : total number of steps in the simulation
    check_step (int) : number of steps between two checking
    epsilon (float): minimum distances (in entropy unit) for the picture to be taken
    X,Y : the average entropy curve of reference.
    """
    for args in iter(inqueue.get,None):
        # create the associated gillespie system
        Nlinker = args[4] 
        ell_tot = args[0]
        kdiff = args[2]
        Energy = args[1]
        seed = args[3]
        dimension = args[5]
        # create the system
        gillespie = gil.Gillespie(ell_tot=ell_tot, rho0=0., BindingEnergy=Energy, kdiff=kdiff,
                            seed=seed, sliding=False, Nlinker=Nlinker, old_gillespie=None, dimension=dimension)
        # pass it as an argument, R returns an array of size (step_tot//check_steps,Nlinker+2,3)
        output.put(('create_group',('/','bin_hist_'+hex(seed))))
        Compute_Pair_Correlation_Function(gillespie,output,'bin_hist_'+hex(seed),step_tot,check_steps,num_bins,max_distance,linked)        
        #output.put(('create_array',('/',"R_"+hex(seed),R)))

def handle_output(output,filename,header):
    """
    This function handles the output queue from the Simulation function.
    It uses the PyTables (tables) library to create and write to an HDF5 file.

    Parameters:
    output (multiprocessing.Queue): The queue from which to fetch output data.

    The function retrieves tuples from the output queue, each of which 
    specifies a method to call on the HDF5 file (either 'createGroup' 
    or 'createArray') and the arguments for that method. 

    The function continues to retrieve and process data from the output 
    queue until it encounters a None value, signaling that all simulations 
    are complete. At this point, the function closes the HDF5 file and terminates.
    """
    hdf = pt.open_file(filename, mode='w') # open a hdf5 file
    while True: # run until we get a False
        args = output.get() # access the last element (if there is no element, it keeps waiting for one)
        if args: # if it has an element access it
            if args.__len__() == 3:
                method, args,time = args # the elements should be tuple, the first element is a method second is the argument.
                array = getattr(hdf, method)(*args) # execute the method of hdf with the given args
                array.attrs['time'] = time
            else :
                method, args = args # the elements should be tuple, the first element is a method second is the argument.
                array = getattr(hdf, method)(*args) # execute the method of hdf with the given args
        else: # once it receive a None
            break # it break and close
    hdf.close()
def make_header(args,sim_arg):
    header ='is close enough to the average entropy curve (that has been computed by averaging 50 to 100 systems) '
    header += 'the file is composed of arrays, each array name can be written : h_X...X where X...X represent an hexadecimal '
    header+= 'name for an integer that corresponds to the seed of the simulation. Each array is made of the position of N '
    header+= 'linkers. Additionnally, the two first entry of the array are [S,NaN,Nan] and [t,NaN,NaN] that are respectively  '
    header+= 'the value of the entropy and time of the given picture.\n'
    header += 'Parameters of the simulation : '
    header +='Nlinker = '+str(args[4])+'\n'
    header +='ell_tot = '+str(args[0])+'\n'
    header += 'kdiff = '+str(args[2])+'\n'
    header += 'Energy =  '+str(args[1])+'\n'
    header += 'seed = '+str(args[3])+'\n'
    header += 'dimension = '+str(args[5])+'\n'
    header+='step_tot = '+str(sim_arg[0])+'\n'
    header+='check_steps = '+str(sim_arg[1])+'\n'
def  Parallel_correlation_function(args,step_tot,check_steps,filename,num_bins,max_distance,linked):
    """
    compute the pair correlation function evolution of the system.
    We only compute the correlation of subsequent linked linkers
    parameters:
    args (iterable) : arguments of the gillespie system in the following order : ell_tot, energy, kdiff,seed,Nlinker,dimension
    step_tot (int) : total number of timesteps use.
    check_step : number of steps between two pictures
    filename : name of the file to save the pictures.
    epsilon (float) : minimum distance for the picture to be taken
    X,Y (arrays) : average entropy of reference.
    return:
    nothing, but create a file with the given name
    """
    num_process = mp.cpu_count()
    output = mp.Queue() # shared queue between process for the output
    inqueue = mp.Queue() # shared queue between process for the inputs used to have less process that simulations
    jobs = [] # list of the jobs for  the simulation
    header = make_header(args,[step_tot,check_steps])
    proc = mp.Process(target=handle_output, args=(output,filename,header)) # start the process handle_output, that will only end at the very end
    proc.start() # start it
    for i in range(num_process):
        p = mp.Process(target=Run, args=(inqueue, output,step_tot,check_steps,num_bins,max_distance,linked)) # start all the 12 processes that do nothing until we add somthing to the queue
        jobs.append(p)
        p.daemon = True
        p.start()
    for arg in args:
        inqueue.put(arg)  # put all the list of tuple argument inside the input queue.
    for i in range(num_process): # add a false at the very end of the queue of argument
        inqueue.put(None) # we add one false per process we started... We need to terminate each of them
    for p in jobs: # wait for the end of all processes
        p.join()
    output.put(False) # now send the signal for ending the output.
    proc.join() # wait for the end of the last process
