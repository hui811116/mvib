import numpy as np
import mvdataset as dt
import mvutils as ut
import sys
import os
import argparse
import pickle
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

d_base = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('dataset',type=str,choices=dt.getAvailableDataset(),help="select the dataset")
#parser.add_argument('output',type=str,help='testing data filename')
parser.add_argument('-seed',type=int,help='Random Seed for Reproduction',default=None)
parser.add_argument('-num',type=int,help='Number of samples randomly chosen for the dataset',default=10000)

# MACRO for Developing
argsdict = vars(parser.parse_args())
data_dict = dt.select(argsdict['dataset'])

rs = RandomState(MT19937(SeedSequence(argsdict['seed'])))

py = data_dict['py']
pxy_list = data_dict['pxy_list']
nview = len(pxy_list)
pxcy_list = [item/np.sum(item,axis=0) for item in pxy_list]

# cumsum 
map_pxcy_list = [np.cumsum(item,axis=0) for item in pxcy_list]
#print(map_pxcy_list)
# by conditional independence
xdim_list = [item.shape[0] for item in pxy_list]

y_label = rs.randint(len(py),size=argsdict['num'])
#print(y_label)
x_raw = rs.rand(argsdict['num'],nview)
#print(x_raw)

x_sample = np.zeros(x_raw.shape)
for iv in range(argsdict['num']):
	for nn in range(nview):
		tmpmap = map_pxcy_list[nn]# the cumsum map for the n view
		for it in range(tmpmap.shape[0]):
			if x_raw[iv,nn]<tmpmap[it,y_label[iv]]:
				x_sample[iv,nn] = it
				break

fname = 'testdata_{}_num_{}.pkl'.format(argsdict['dataset'],argsdict['num'])
savefile= os.path.join(d_base,fname)
#print(x_sample)
output_dict= {'x_test':x_sample,'y_test':y_label.astype(int)}

print('Saving testing data to: {}'.format(savefile))
with open(savefile,'wb') as fid:
	pickle.dump(output_dict,fid)