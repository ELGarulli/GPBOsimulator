# sciNeurotech Lab 
# Theodore
"""
in this file, we will define a class in order to process data 
"""

# import
import numpy as np
from scipy.io import loadmat

class DataSetFake():
    """
    class to process data

    Attributes:
        path_to_dataset_folder (str): path to the dataset folder where there is the
                                        file you want to process
        dataset_type (str): 'nhp' (non human primat) or 'rat' 
        dataset_file (str): name of the file you want the data to be processed
        dataset_name (str): name you want to give to the dataset
        set (dict): dict_keys(['emgs', 'nChan', 'sorted_isvalid', 'sorted_resp',
                               'sorted_respMean', 'ch2xy']) 

    Methods:
        load_matlab_data():
            function from Rose
            use a .mat files and return a dictionary
        get_valid_response(emg_id: int, electrode_id: int):
            get one of the valid responses of one electrode measured by one emg
        get_realistic_response(emg_id: int, electrode_id: int):
            get one of the valid and outlier responses of one electrode measured by one emg
        get_mean_response(emg_id: int, electrode_id: int):
            the mean response of one electrode measured by one emg
    """

    def __init__(self, emgs, nChan, sorted_resp, sorted_isvalid, ch2xy, sorted_respMean, dataset_name: str = 'NO_NAME') -> None:
        """
        initialize a DataProcess instance

        Args:
            - emgs: 1 x e cell array of strings. Muscle names for each implanted EMG.
                - nChan: a single scalar equal to c. Number of cortical array channels.
                - sorted_resp: c x e cell array. Corresponds to "response"*, where each EMG
                  response has been sorted and assigned to the source stimulation site.
                - sorted_isvalid: c x e cell array. Corresponds to "isvalid"*, where each EMG
                  response has been sorted and assigned to the source stimulation site.
                - sorted_respMean: c x e single array. Average of all valid responses,
                  segregated per stimulating channel and per EMG.
                - ch2xy: c x 2 matrix with <x,y> relative coordinates for each stimulation
                  channel. Units are intra-electrode spacing.

                * response: 1 x e cell array. Each entry is a numerical matrix associated to a
                  single EMG. Each entry is j x 1 and represents a sampled cumulative response
                  (during peak activity) for each evoked_emg. Thus, each trace is collapsed to
                  a single outcome value.
                * isvalid : 1 x e cell array. Each entry is a numerical matrix associated to a
                  single EMG. Each entry is j x 1 and determines whether the recorded response
                  can be considered valid. A value of 1 means that we found no reason to exclude
                  the response. A value of 0 means that baseline (pre-stimulus) activity exceeds
                  accepted levels and indicates that spontaneous EMG activity was ongoing at the
                  time of stimulus delivery. A value of -1 indicates that the response is an
                  outlier, yet baseline activity is within range. We consider the "0" and "-1"
                  categories practically possible and impossible to reject during online trials,
                  respectively.

                c : number of channels in the implanted cortical array (variable between species).
                e : implanted EMG count (variable between subjects).
                j : individual stimuli count throughout the session.
                t : number of recorded samples for each stimulus.
                Sampling frequency is sampFreqEMG (variable between species).
            dataset_name (str, optional): name of the dataset. Defaults to 'NO_NAME'.
        """
        self.emgs = emgs
        self.nChan = nChan
        self.sorted_resp = sorted_resp
        self.sorted_isvalid = sorted_isvalid
        self.ch2xy = ch2xy
        self.sorted_respMean = sorted_respMean
        self.dataset_name = dataset_name
        self.set = self.set = {'emgs': self.emgs,
                'nChan': self.nChan,
                'sorted_isvalid': self.sorted_isvalid,
                'sorted_resp': self.sorted_resp,
                'sorted_respMean': self.sorted_respMean,
                'ch2xy': self.ch2xy}


    def get_valid_response(self, emg_id: int, electrode_id: int) -> float:
        """
        get one of the valid responses of one electrode measured by one emg

        Args:
            emg_id (int): emg identifier
            electrode_id (int): electrode identifier

        Returns:
            resp (double): response to one of the query made in electrode_id and measured by emg_id
        """
        #valid_resp = self.set['sorted_resp'][electrode_id, emg_id][:,0][
        #    self.set['sorted_isvalid'][electrode_id, emg_id][:,0] == 1]
        valid_resp = self.set['sorted_resp'][electrode_id, emg_id][:, 0][
            self.set['sorted_isvalid'][electrode_id, emg_id][:, 0] == 1]
        resp = np.random.choice(valid_resp) # select randomly one repetition
        return resp  
    
    def get_realistic_response(self, emg_id: int, electrode_id: int) -> float:
        """
        get one of the valid and outlier responses of one electrode measured by one emg

        Args:
            emg_id (int): emg identifier
            electrode_id (int): electrode identifier

        Returns:
            resp (double): response to one of the query made in electrode_id and measured by emg_id
        """
        realistic_resp = self.set['sorted_resp'][electrode_id, emg_id][:, 0][
            (self.set['sorted_isvalid'][electrode_id, emg_id][:, 0] == 1) | 
            (self.set['sorted_isvalid'][electrode_id, emg_id][:, 0] == -1)]    
        resp = np.random.choice(realistic_resp) # select randomly one repetition
        return resp  

    def get_mean_response(self, emg_id: int, electrode_id: int) -> float:
        """
        get the mean response of one electrode measured by one emg

        Args:
            emg_id (int): emg identifier
            electrode_id (int): electrode identifier

        Returns:
            resp (double): response the mean response for electrode_id and measured by emg_id
        """
        resp = self.set['sorted_respMean'][electrode_id, emg_id] 
        return resp 
