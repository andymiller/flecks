import os
import glob
import cPickle as pickle
import numpy as np

# Location of included datasets
data_path = os.path.join(os.path.dirname(__file__), 'datasets')

# nba shots dataset - import list of NBA shots
def nba_shots():
  shots_pkl_file = os.path.join(data_path, "nba_shots.pkl")
  if os.path.exists(shots_pkl_file):
    print "loading shots from file: ", shots_pkl_file
    return pickle.load(open(shots_pkl_file, 'rb'))
  else:
    print "loading shots from directory: ", shots_pkl_file
    shots_dir = os.path.join(data_path, "nba_shots")
    shots_glob = glob.glob(shots_dir + "/*.txt", 'rb')
    shot_dict = {}
    for i,sfile in enumerate(shots_glob): 
      pname = os.path.splitext(os.path.basename(sfile))[0]
      shots = np.reshape(np.loadtxt(sfile, skiprows=1), (-1,3))
      # put all shots on the same half court
      for n in range(shots.shape[0]):
        if shots[n,0] > 47: 
          shots[n,0] = 94 - shots[n,0]
          shots[n,1] = 50 - shots[n,1]
      shot_dict[pname] = shots
    pickle.dump(shot_dict, open(shots_pkl_file, "wb"))
    return shot_dict


def mine_data(bin_width = 50): 
    """ Loads one dimensional count data for a semi-toy Poisson regression task.

     xx, yy = get_mine_data(bin_width)
     Inputs:
         bin_width 1x1 Optional, Default=50.
                   Number of days in each bin (except possibly the last)
                   The default bin width of 50 gives 811 bins.

     Outputs:
            xx 1xN Centres of bins. Time measured in days.
                   (Could argue about best definition here. I've picked it so
                   that if a bin only contains first day the bin is at '1', if
                   it contains the first two days it is at '1.5' and so on.)
            yy 1xN Number of events in each bin

        Andrew Miller (acm@seas.harvard.edu), 
        adpated from Iain Murray's matlab code (October 2009)
    """
    num_days = 40550
    num_events = 191

    # intervals = the number of days between incidents (190 intevals => 191 events)
    intervals  = np.loadtxt(os.path.join(data_path, 'mining.dat'))
    event_days = np.concatenate([[1], np.cumsum(intervals)+1])
    assert event_days[-1] == num_days, "event days derived from intervals not correct!"

    #edges = [1:bin_width:num_days, num_days+1];
    edges = np.concatenate([np.arange(1, num_days, bin_width),
                            [num_days + 1]])
    bin_counts, _ = np.histogram(event_days, bins = edges);
    assert sum(bin_counts) == num_events, "histogram didn't work right."

    # Should have no data at exactly num_days+1, also strip off this cruft
    #assert bin_counts[-1] == 0, "last count not equal to zero!"
    bin_counts = bin_counts[1:-1]

    xx = (edges[1:-2] + (edges[2:-1]-1)) / 2.
    yy = bin_counts
    return xx, yy

