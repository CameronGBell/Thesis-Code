import segpix as spr
import multiprocessing as mp
import numpy as np

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from scipy.stats import sem
import os, sys
import warnings
from PIL import Image
import ParticleSpy.api as ps
import hyperspy.api as hs

#loads images for para
def loadims(afolder):
    folderi = f"Ground truth/{afolder}/image"
    folderm = f"Ground truth/{afolder}/mask"

    images = []
    masks = []
    names = []

    for mask in os.listdir((folderm)):
        names.append(mask)
        maskfile = np.asarray(Image.open(os.path.join(folderm,mask)))
        if afolder == 'AuGe TEM':
            image = mask[:-7]+".tiff"

            thing = np.zeros_like(maskfile)
            thing[maskfile==255] = 1
            thing = thing +1
        else:
            image = mask[:-7]+".dm4"
            thing = ps.toggle_channels(maskfile[:,:,:3], colors = ['#0000ff','#ff0000'])
                
        #warnings.filterwarnings("ignore")
        imagefile = hs.load(os.path.join(folderi,image))
        if afolder == 'PdC TEM' or afolder == 'AuGe TEM':
            im = imagefile.data
            normed = (im-im.min())*(1000/(im.max()-im.min()))
            images.append(normed)
        else:
            images.append(imagefile.data)
        masks.append(thing)
    return (images[0:11], masks[0:11])

#subprocess of multi process
def batch_test(clff,pix,imask,output):
    die = np.zeros((2,2,10,4))
    tim = np.zeros((4,1))
    
    die[:,:,:,1], tim[1,:] = spr.test_classifier_per(clff, imask[1],pix)
    print(.1, end='\r')
    die[:,:,:,2], tim[2,:] = spr.test_classifier_per(clff, imask[2], pix)
    print(.2, end='\r')
    p = ps.trainableParameters()
    p.setGlobalSigma(4)
    p.setDiffGaussian(prefilter=False,high_sigma=64)
    die[:,:,:,3], tim[3,:] = spr.test_classifier_per(clff, imask[3], pix, params=p)
    print(.3, end='\r')
    die[:,:,:,0], tim[0,:] = spr.test_classifier_per(clff, imask[0], pix, params=p)
    print(.4, end='\r')
    dietim = (die,tim)
    output.put(dietim)

#perlist = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8000,16000,32000,64000,128000,256000,512000,1000000]
#l is number of runs for each number of pixels
def make_results_para(clff, l):

    perlist = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8000,16000,32000,64000,128000,256000,512000,1000000]
    if str(clff) == 'KNeighborsClassifier()':
        perlist = [8,16,32,64,128,256,512,1024,2048,4096,8000,16000,32000,64000,128000,256000,512000,1000000]
    if str(clff) == 'QuadraticDiscriminantAnalysis()':
        perlist = [128,256,512,1024,2048,4096,8000,16000,32000,64000,128000,256000,512000,1000000]

    #creates image arrays
    imarray = []
    namelist = ['PdC TEM','Pt ADF','PdPtNiAu ADF','AuGe TEM']
    for i in range(len(namelist)):
        imarray.append(loadims(namelist[i]))

    #writes first line to file
    data = open('results/acc{} prl.csv'.format(str(clff)),'w')
    data.write('pixels,ln(pix),,')
    for i in range(4):
        data.write('time,sem,die00,sem,die01,sem,die10,sem,die11,sem,prec,sem,rec,sem,,')
    data.write('\n')
    data.close()

    for i in range(len(perlist)):
        print(f'{i}.0')
        output = mp.Queue()
        processes = [mp.Process(target=batch_test, args = (clff,perlist[i],imarray,output,)) for x in range(l)]
            
        for p in processes:
            p.start()
        
        listtoop = [output.get() for p in processes]
        # join is under get bc https://bit.ly/3sYudkW

        for p in processes:
            p.join()

        die = np.zeros((2,2,10*l,4))
        tim = np.zeros((10*l,4))
        dielist = []
        timlist = []
        for j in range(l):
            dielist.append(listtoop[j][0])
            timlist.append(listtoop[j][1])
        
        die = np.concatenate((dielist), axis=2)
        tim = np.concatenate((timlist), axis=1)

        timsem = sem(tim,axis=1)
        tim=tim.mean(axis=1)
        
        prec = die[0,0,:,:]/(die[1,0,:,:]+die[0,0,:,:])
        rec = die[0,0,:,:]/(die[0,1,:,:]+die[0,0,:,:])
        precsem = sem(prec, axis=0, nan_policy='omit')
        recsem = sem(rec, axis=0, nan_policy='omit')
        prec = np.nanmean(prec,axis=0)
        rec = np.nanmean(rec, axis=0)

        stderr= sem(die, axis=2, nan_policy='omit')
        die = die.mean(axis=2)

        data = open('results/acc{} prl.csv'.format(str(clff)),'a')
        data.write(f'{perlist[i]},,,')
        for j in range(4):
            data.write(f'{tim[j]},{timsem[j]},{die[0,0,j]},{stderr[0,0,j]},{die[0,1,j]},{stderr[0,1,j]},{die[1,0,j]},{stderr[1,0,j]},{die[1,1,j]},{stderr[1,1,j]},{prec[j]},{precsem[j]},{rec[j]},{recsem[j]},,')
        data.write('\n')
        data.close()
    
    return

if __name__ == '__main__':
    
    # this creates CSVs of dice coefficients and their errors

    clff = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    make_results_para(clff,10)
    clff = KNeighborsClassifier()
    make_results_para(clff,10)
    clff = GaussianNB()
    make_results_para(clff,10)
    clff = QuadraticDiscriminantAnalysis()
    make_results_para(clff,10)


    