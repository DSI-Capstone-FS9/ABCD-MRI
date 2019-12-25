import glob
import pandas as pd

from mricode.utils import return_csv
from mricode.utils import data_generator
from mricode.utils import return_iter

path_csv = '/data2/csv/'
filename_res = {'train': 'intell_residual_train.csv', 'val': 'intell_residual_valid.csv', 'test': 'intell_residual_test.csv'}
train_df, val_df, test_df, norm_dict = return_csv(path_csv, filename_res, False)
batch_size = 8

train_iter, val_iter, test_iter = return_iter('/data/res256/', 'allimages_256', 8, onlyt1=True, bl_cropped=False, dim=256)
config = {}
config['down256'] = {}
config['down256']['iter_train'] = train_iter
config['down256']['iter_val'] = val_iter
config['down256']['iter_test'] = test_iter
config['down256']['norm'] = {}
config['down256']['norm']['t1'] = [1.6431085066123967, 3.774774262557981]
config['down256']['norm']['t2'] = [2.3879970019972974, 5.233771497359756]
config['down256']['norm']['ad'] = [0,1]
config['down256']['norm']['fa'] = [0,1]
config['down256']['norm']['md'] = [0,1]
config['down256']['norm']['rd'] = [0,1]
#(11.781510935110205, 28.61685861073625)
#400.57137179374695
#972.9731927650325
#34.0
#34.0
config['down256']['norm']['t1'] = [11.781510935110205, 28.61685861073625]

train_iter, val_iter, test_iter = return_iter('/data2/res128/down2/', 'allimages_128', 8, onlyt1=True, bl_cropped=False, dim=128)
config['down128'] = {}
config['down128']['iter_train'] = train_iter
config['down128']['iter_val'] = val_iter
config['down128']['iter_test'] = test_iter
config['down128']['norm'] = {}
config['down128']['norm']['t1'] = [1.4577492775905838, 3.575234065990881]
config['down128']['norm']['t2'] = [2.3523901205205155, 5.198199383862082]
config['down128']['norm']['ad'] = [0., 1.]
config['down128']['norm']['fa'] = [0., 1.]
config['down128']['norm']['md'] = [0., 1.]
config['down128']['norm']['rd'] = [0., 1.]
#(1.4577492775905838, 3.575234065990881, 2.3523901205205155, 5.198199383862082)
#(11.659994564651159, 28.596968228796744, 18.81589409157082, 41.57846448189274)
config['down128']['norm']['t1'] = [11.659994564651159, 28.596968228796744]
config['down128']['norm']['t2'] = [18.81589409157082, 41.57846448189274]

train_iter, val_iter, test_iter = return_iter('/data2/res128/cropped/', 'allimages_cropped', 8, onlyt1=True, bl_cropped=True, dim=128)
config['cropped128'] = {}
config['cropped128']['iter_train'] = train_iter
config['cropped128']['iter_val'] = val_iter
config['cropped128']['iter_test'] = test_iter
config['cropped128']['norm'] = {}
config['cropped128']['norm']['t1'] = [0.014236470049431932, 0.018582874870362885]
config['cropped128']['norm']['t2'] = [3.4985802362108696, 5.926868989565343]
config['cropped128']['norm']['ad'] = [0., 1.]
config['cropped128']['norm']['fa'] = [0., 1.]
config['cropped128']['norm']['md'] = [0., 1.]
config['cropped128']['norm']['rd'] = [0., 1.]
#(0.11387223162995555, 0.1486375080508724, 27.98384273984305, 47.40682178073459)
config['cropped128']['norm']['t1'] = [0.11387223162995555, 0.1486375080508724]
config['cropped128']['norm']['t2'] = [27.98384273984305, 47.40682178073459]

train_iter, val_iter, test_iter = return_iter('/data2/res64/cropped/', 'allimages_cropped', 8, onlyt1=False, bl_cropped=True)
config['cropped64'] = {}
config['cropped64']['iter_train'] = train_iter
config['cropped64']['iter_val'] = val_iter
config['cropped64']['iter_test'] = test_iter
config['cropped64']['norm'] = {}
config['cropped64']['norm']['t1'] = [0.014143138115992337, 0.017672428083463775]
config['cropped64']['norm']['t2'] = [3.4986283952101, 5.739028074303452]
config['cropped64']['norm']['ad'] = [8.927673748645995e-05, 0.01840090965400136]
config['cropped64']['norm']['fa'] = [0.022616118307670543, 0.02300114723957744]
config['cropped64']['norm']['md'] = [7.064957440618201e-05, 0.018202582338173495]
config['cropped64']['norm']['rd'] = [6.21606477025423e-05,0.016684894620698765]
#(0.11312570418978232, 0.14135518265387828, 27.98422794577516, 45.90435212793337,
#0.0007140914352312044, 0.1471820359293305, 0.18089792297946083, 0.18397762627431558,
#0.0005650996822530141, 0.14559568945663873, 0.0004971999132421456, 0.13345626959299658)
config['cropped64']['norm']['t1'] = [0.11312570418978232, 0.14135518265387828]
config['cropped64']['norm']['t2'] = [27.98422794577516, 45.90435212793337]
config['cropped64']['norm']['ad'] = [0.0007140914352312044, 0.1471820359293305]
config['cropped64']['norm']['fa'] = [0.18089792297946083, 0.18397762627431558]
config['cropped64']['norm']['md'] = [0.0005650996822530141, 0.14559568945663873]
config['cropped64']['norm']['rd'] = [0.0004971999132421456, 0.13345626959299658]

train_iter, val_iter, test_iter = return_iter('/data2/res64/down/', 'allimages', 8, onlyt1=False)
config['down64'] = {}
config['down64']['iter_train'] = train_iter
config['down64']['iter_val'] = val_iter
config['down64']['iter_test'] = test_iter
config['down64']['norm'] = {}
config['down64']['norm']['t1'] = [1.3779395849814497, 3.4895845243139503]
config['down64']['norm']['t2'] = [2.22435586968901, 5.07708743178319]
config['down64']['norm']['ad'] = [1.3008901218593748e-05, 0.009966655860940228]
config['down64']['norm']['fa'] = [0.0037552628409334037, 0.012922319568740915]
config['down64']['norm']['md'] = [9.827903909139596e-06, 0.009956973204022659]
config['down64']['norm']['rd'] = [8.237404999587111e-06,0.009954672598675338]
#(11.021626502094422, 27.911889384464533, 17.791795714892476, 40.60973499962659, 
#0.0001040533648911113, 0.07971957520595675, 0.03003695147528488, 0.10336083045998391,
#7.860974992344717e-05, 0.07964212723272444, 6.588794040136137e-05, 0.07962372554578312)

config['down64']['norm']['t1'] = [11.021626502094422, 27.911889384464533]
config['down64']['norm']['t2'] = [17.791795714892476, 40.60973499962659]
config['down64']['norm']['ad'] = [0.0001040533648911113, 0.07971957520595675]
config['down64']['norm']['fa'] = [0.03003695147528488, 0.10336083045998391]
config['down64']['norm']['md'] = [7.860974992344717e-05, 0.07964212723272444]
config['down64']['norm']['rd'] = [6.588794040136137e-05, 0.07962372554578312]