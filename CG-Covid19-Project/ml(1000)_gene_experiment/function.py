import numpy as np
import torchvision.transforms as transforms
from Bio import SeqIO
from tqdm.notebook import tqdm
from torchvision.models import alexnet, resnet18
from torchcam.methods import SmoothGradCAMpp, LayerCAM, GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image

import torch
from torch.nn import Module
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
import random
np.random.seed(2020)
random.seed(2020)
torch.manual_seed(2020)

class grad_cam:
    def __init__(self, device='cpu', path2weights=None, pkl_path = None, class_num =1):
        self.device = device
        self.path2weights = path2weights
        self.deepinsight_pkl = pkl_path
        self.class_num = class_num

    def reload_model(self,):
        models = resnet18(pretrained=False, num_classes = self.class_num)
        # checkpoint = torch.load('./models/weights_Multiclass_Covid19(Non-kmer3)_IndexRemark.2022.03.24[NACGTRYKMSWBDHV]/weights_Multiclass_Covid19(Non-kmer3)[NACGTRYKMSWBDHV].2022.03.24.pt', map_location=torch.device('cpu'))
        models.load_state_dict(torch.load(self.path2weights))
        return models

    def loader_cam(self, image_, lable_, class_dict = None, ths = 0.5): #single classes loader
            model = self.reload_model().to(self.device).eval()
            cam_extractor = LayerCAM(model, ["layer2", "layer3", "layer4"])
            classes__ = class_dict[int(lable_)]
            out = model(image_.to(self.device))
            # print(torch.sigmoid(out))
            cams = cam_extractor(out.squeeze(0).argmax().item(), out)
            fused_cam = cam_extractor.fuse_cams(cams)
            cam_extractor.clear_hooks()
            
            # Resize it
            ths = 0.75
            resized_cams = resize(to_pil_image(fused_cam), image_.shape[-2:])
            # resized_cams = [resize(to_pil_image(cam), img.shape[-2:]) for cam in cams]
            segmaps = to_pil_image((resize(fused_cam.unsqueeze(0), image_.shape[-2:]).squeeze(0) >= ths).to(dtype=torch.float32))
            # segmaps = [to_pil_image((resize(cam.unsqueeze(0), img.shape[-2:]).squeeze(0) >= ths).to(dtype=torch.float32)) for cam in cams]

            # Calc cam weight
            capture_image = np.where(np.array(segmaps), np.array(image_[0][0]), np.array(image_[0][0])*0)

            # for name, cam, seg in zip(cam_extractor.target_names, resized_cams, segmaps):
            #     capture_image = np.where(np.array(seg), np.array(images[0][0]), np.array(images[0][0])*0)
            return capture_image, image_[0][0].cpu().numpy()

class TransferDataset(Dataset):
    def __init__(self, data_list, labels, transform, model_type='cnn'):
        self.transform = transform
        self.data_list = data_list
        self.labels = labels
        self.model_type =model_type
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        seed = np.random.randint(1e9)       
        random.seed(seed)
        np.random.seed(seed)
        if self.model_type == 'cnn':
            singel_image_ = np.load(self.data_list[idx]).astype(np.float32)
            singel_image_ = self.transform(singel_image_) #for image
        else:
            singel_image_ = self.data_list[idx]
            singel_image_ = torch.FloatTensor(singel_image_)#for sequence
        
        label = self.labels[idx]
        # print(label)
        return singel_image_, label

class torch_dataset_func:
    def __init__(self, model_type = 'cnn'):
        self.data_ds = None
        self.data_dl = None
        self.model_type =model_type

    def get_TransferDataset(self, data_list = None, labels = None, batch_size = 1, shuffle = False):
        transformer = transforms.Compose([
                    transforms.ToTensor(),
                    ])     

        self.data_ds = TransferDataset(data_list= data_list, labels= labels, transform= transformer, 
                                                                        model_type = self.model_type)
        self.data_dl = DataLoader(self.data_ds, batch_size= batch_size, 
                        shuffle=shuffle)

        return self.data_dl
        # get func item
    def get_ds(self,):
        return self.data_ds


class sequence_dataprocess:
    def __init__(self, gene_list = '-NACGT' ):
        self.gene_list = gene_list
        self.label_ = []
        self.class_ =None
        self.new_lineage_label = []
        self.new_lineage_label_1000 = []

    def clean(self, x):
        x = x.upper() 
        if x == 'T' or x == 'A' or x == 'G' or x == 'C' or x == '-' or x == 'N':
            return x
        if x == 'U' or x == 'Y':
            return 'T'
        if x == 'K' or x == 'S':
            return 'G'
        if x == 'M' or x == 'R' or x == 'W' or x == 'H' or x=='V' or x=='D':
            return 'A'
        if x== 'B':
            return 'C'

    def convert_gene_index(self, lineage_list):
        dict_search = {}
        for idx, i in enumerate(self.gene_list):

            dict_search[i] = idx
        print(dict_search)

        num_new_sequences =[]
        for k in tqdm(lineage_list):
            temp_store=[]
            for j in k:
                temp_store.append(self.clean(j)) #one hot
                # temp_store.append(dict_search[clean(j)])
            num_new_sequences.append(temp_store)
        total_sequence_array = np.array(num_new_sequences)
        # print(total_sequence_array.shape)
        return total_sequence_array

    def  dataframe_dataloader(self, fasta_data_path, lineage_label, selection_filter):
        # loading fasta data
        # plz check ur pandas dataframe context and dimension 
        # only for one lineage function
        fasta_data = SeqIO.parse(fasta_data_path,"fasta")
        for idx, rna in enumerate(fasta_data):
            if "B.1.617.2" == lineage_label[idx][0]:
            # break
            # print(lineage_label[idx][0].split(' ')[0])
                self.label_.append(lineage_label[idx][1].split(' ')[0])
                self.new_lineage_label.append(str(rna.seq))
                self.new_lineage_label_1000.append(np.array(list(str(rna.seq)))[selection_filter])
        print('filter sample:', len(self.new_lineage_label))
        print('-----sample len-----')
        print('total sequence shape', len(self.new_lineage_label[0]),'  ||','  filter sequence shape', len(self.new_lineage_label_1000[0]))
        # get database class name
        self.class_, _, _, _= np.unique(self.label_,return_counts=True,return_index=True,return_inverse=True)
        print("-----class name-----")
        class_dict_ = {}
        for idx, i in enumerate(self.class_):
            class_dict_[i] = idx
        print(class_dict_)
        num_label = []
        for i in self.label_:
            num_label.append(class_dict_[i])
        return self.new_lineage_label, self.new_lineage_label_1000, self.class_, num_label

    def gene_index_remaker(self, seq_array ,c_type='Integer'):
            if c_type == 'Integer': #ver1
                gene_index ={'A': 2, 'C': 1, 'G': 3, 'T': 0, 'N': -1, '-': -1}
            elif c_type == 'EIIP': #ver2
                gene_index ={'A': 0.1260, 'C': 0.1340, 'G': 0.0806, 'T': 0.1335, 'N': -0.1, '-': -0.1}
            elif c_type == 'Atomic': #ver3
                gene_index ={'A': 70, 'C': 58, 'G': 78, 'T': 66, 'N': -1, '-': -1}
            elif c_type =='Voss':
                gene_index ={'-': [1,0,0,0,0,0], 'A': [0,1,0,0,0,0], 'C': [0,0,1,0,0,0],  'G': [0,0,0,1,0,0], 'N': [0,0,0,0,1,0],  'T': [0,0,0,0,0,1]}
            num_new_sequences =[]
            for k in tqdm(seq_array):
                temp_single_seq_transfer = []
                for j in k:
                    temp_single_seq_transfer.append(gene_index[j])
                num_new_sequences.append(temp_single_seq_transfer)
            total_sequence_array = np.array(num_new_sequences)
            if c_type == 'Voss':
                total_sequence_array = np.reshape(total_sequence_array, (total_sequence_array.shape[0], total_sequence_array.shape[1]*total_sequence_array.shape[2]))
            return total_sequence_array

