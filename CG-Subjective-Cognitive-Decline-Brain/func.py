
from import_library import *
from scipy import io
# dataloader
class loaders:
    def __init__(self, data_type='mat'):
        if data_type=='mean':
            md ='FCM_mean'
        else:
            md = 'FCM_var'
        self.data_type = md
        self.concat_mat_array = []
    
    def load_csv(self, data_path = None):
        pd_tb = pd.read_csv(data_path)
        pd_patient_index = pd_tb[pd_tb.columns[0]]
        pd_patient_value = pd_tb[pd_tb.columns[1::]]
        return pd_patient_index, pd_patient_value, pd_tb.columns[1::]
    
    def mat_process(self, data):
        get_data = []
        for idx, i in enumerate(data):
            if len(i[idx+1::])!=0:
                get_data.append(list(i[idx+1::]))
        get_data = list(itertools.chain(*get_data))
        # print([idx for idx, i in enumerate(get_data) if (i>-2)==False])
        return get_data
    def load_mat(self, data_path=None):
        mat = io.loadmat(data_path)
        get_value = mat[self.data_type]
        get_value = self.mat_process(get_value)
        self.concat_mat_array.append(get_value)
    def callback(self,):
        return np.array(self.concat_mat_array).astype(np.float16)
        
