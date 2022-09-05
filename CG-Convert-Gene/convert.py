import os, sys, Bio, argparse
import pandas as pd
# search gene name
search_dict = {(266,21555): ['ORF1ab',], 
                              (21563, 25384): ['S',], 
                              (25393, 26220): ['ORF3a',], 
                              (26245, 26472): ['E',], 
                              (26523, 27191): ['M',], 
                              (27202, 27387): ['ORF6',],
                              (27394, 27759): ['ORF7a',], 
                              (27756, 27887): ['ORF7b',], 
                              (27894, 28259): ['ORF8',],
                              (28274, 29533): ['N',], 
                              (29558, 29674): ['ORF10',]}

from Bio.Seq import Seq
class convert_process:
    def __init__(self,):
        # parameter list
        self.chrom = [] #sequence name
        self.position = [] #gene position
        self.ref = [] #before gene
        self.alt = [] #after gene
        self.dp = [] #-
        self.ad = [] #-
        self.gene = [] #RNA name
        self.gene_loc = []
        self.trans = [] #Gene
    def convert_gene(self, or_seq, loc_p, loc_gene):
        if 21555 >= loc_gene[0] and 13468<=loc_gene[0]:
            split_strings = ''.join(or_seq).split()
            split_strings.insert(13202, 'C')
            or_seq = ''.join(split_strings)
        first_loc = None
        if loc_p%3==0:
            b_rna = f'{or_seq[loc_p] + or_seq[loc_p+1] + or_seq[loc_p+2]}'.replace(' ','')
            a_rna = f'{loc_gene[1] + or_seq[loc_p+1] + or_seq[loc_p+2]}'.replace(' ','')
            first_loc = loc_p
            # print(loc_p, loc_p+1, loc_p+2)
        elif loc_p%3==1:
            b_rna = f'{or_seq[loc_p-1] + or_seq[loc_p] + or_seq[loc_p+1]}'.replace(' ','')
            a_rna = f'{or_seq[loc_p-1] + loc_gene[1] + or_seq[loc_p+1]}'.replace(' ','')
            first_loc = loc_p-1
            # print(loc_p-1, loc_p, loc_p+1)
        else:
            b_rna = f'{or_seq[loc_p-2] + or_seq[loc_p-1] + or_seq[loc_p]}'.replace(' ','')
            a_rna = f'{or_seq[loc_p-2] + or_seq[loc_p-1] + loc_gene[1]}'.replace(' ','')
            first_loc = loc_p-2
        #     print(loc_p-2, loc_p-1, loc_p)
        # print('first loc', int(first_loc/3))

        try:
            return f'{Seq(b_rna).translate()[0]}{int(first_loc/3)+1}{ Seq(a_rna).translate()[0]}'
        except:
            return f'{Seq(b_rna).translate()[0]}{int(first_loc/3)+1}X'
    
    def process_rna(self, pos, i):
        self.chrom.append('NC_045512.2')
        self.position.append(pos[0])
        self.ref.append(search_dict[i][1][pos[0] - i[0]])
        self.alt.append(pos[1])
        self.gene.append(search_dict[i][0])
        self.gene_loc.append(pos[0] - i[0]+1)
        self.trans.append(self.convert_gene(search_dict[i][1], pos[0] - i[0], pos))

    def search_position(self, pos):
        for idx, i in enumerate(search_dict):
            # if ==
            if (pos[0] >=i[0] and pos[0]<=i[1]) and (args.nullchange)==True:
                self.process_rna(pos, i)
            elif (pos[0] >=i[0] and pos[0]<=i[1]) and (args.nullchange)==False:
                self.process_rna(pos, i)

    def standard(self,):
        # data_vcf = {'CHROM': self.chrom, 'POS': self.position, 'REF': self.ref, 'ALT': self.alt1, 'Gene': self.gene, 'REF_Position': self.debug}
        data_vcf = {'CHROM': self.chrom, 'POS': self.position, 'REF': self.ref, 'ALT': self.alt, 'Gene': self.gene, 'Gene LOC': self.gene_loc, 'Translate': self.trans}
        return pd.DataFrame(data_vcf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=False, default='./demo/test.csv',help='Input Gene Feature data. e.g.: *.csv.  Default: ./demo/test.csv')
    parser.add_argument("--output", "-o", type=str, required=False, default='./demo/output', help='Export translation gene data. Default: ./demo/output')
    parser.add_argument("--nullchange", "-nc", type=bool, required=False, default=True, help='Filter Gene with no change. Default: True')
    # parser.add_argument("-h", "--help", default='help',required=False)
    args = parser.parse_args()
    feature_data = list(pd.read_csv(args.input).values)

    f = open('./demo/sequence.txt', 'r')
    gene_dict = [i.replace('\n','').split('[gbkey=Gene]')[1] for i in f.read().split('>') if len(i)>0 ]
    for idx, i in enumerate(search_dict):
        search_dict[i].append(gene_dict[idx])
    f.close()

    start_convert = convert_process()
    for i in feature_data:
        start_convert.search_position(i)
        # break
    print(start_convert.standard())
    (start_convert.standard()).to_csv('{}.csv'.format(args.output))
# --help data formate = [csv, txt, tsv]
# --input {dataname}.tsv, {dataname}.csv
# --output {dataname}.tsv, {dataname}.csv
# --dtype pandas, tsv
# function 
    # translation Gene -> compare group of 3 rna    