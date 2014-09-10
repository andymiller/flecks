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
    shots_glob = glob.glob(shots_dir + "/*.txt")
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



