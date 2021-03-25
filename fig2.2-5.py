import ParticleSpy.api as ps
import segpix as sp
import hyperspy.api as hs
import numpy as np
import accuracy as ac
import os
import matplotlib.pyplot as plt
import math as m

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import scale
from PIL import Image
from scipy.stats import ks_2samp
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib as mpl


folder = 'C:/Users/cimca/Google Drive (cameronbell2236@gmail.com)/Diamond/Python Stuff/Results/Distribution Analysis'

def create_smol_features(afolder, num, intensity = True, membrane = [1,1,1,1,1,1], edges=True, texture = True, test=True, p=[1,16,20]):
    #creates set of featues using CS and opens the right folders
    folderi = "C:/Users/cimca/Google Drive (cameronbell2236@gmail.com)/Diamond/Python Stuff/Ground truth/{}/image".format(afolder)
    folderm = "C:/Users/cimca/Google Drive (cameronbell2236@gmail.com)/Diamond/Python Stuff/Ground truth/{}/mask".format(afolder)

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
        
        imagefile = hs.load(os.path.join(folderi,image))
        images.append(imagefile.data)
        masks.append(thing)

    sigma = p[0]
    high_sigma = p[1]
    disk_size = p[2]

    mask = masks[0]
    image = images[0]
    features = ps.CreateFeatures(image, intensity=intensity, edges=edges, texture=texture, membrane=membrane, test=test,
                                     sigma=sigma, high_sigma=high_sigma, disk_size=disk_size)
    
    area = image.shape[0]*image.shape[1]
    indexes = np.random.randint(0,high=area,size=num)
    frac_array = np.zeros(area)
    frac_array[indexes] = 1
    frac_array = np.reshape(frac_array, (image.shape[0],image.shape[1])).astype(np.bool)
    features = np.rot90(np.rot90(features, axes=(2,0)), axes=(1,2))

    training_data = features[:, frac_array].T
    
    mask = mask[frac_array]
    mask = mask.ravel()


    labelled_features = np.concatenate((np.reshape(mask,(mask.shape[0],1)),training_data), axis=1)

    return labelled_features



def gen_pairs(n):

    pair_list = []
    for i in range(n):
        for ii in range(i+1,n):
            pair_list.append((i+1,ii+1))
    return pair_list

def make_names(intensity = True,
                edges = True, texture = True, 
                membrane = [1,1,1,1,1,1], test = True):
    names = ['labels']
    if intensity:
        names.append('Gaussian')
        names.append('Diff. of Gaussians')
        names.append('Median')
        names.append('Maximum')
        names.append('Minimum')
    if edges:
        names.append('Sobel')
    if texture:
        names.append( 'Hessian')
        names.append('laplacian')
    if membrane != [0,0,0,0,0,0]:
        mems = ['Memb. -Sum', 'Memb. -Mean', 'Memb. -StdDev', 'Memb. -Med', 'Memb. -Max', 'Memb. -Min']
        for i in range(6):
            if membrane[i] == 1:
                names.append(mems[i])
    if test:
        names.append('laplacian')
    return names


def ksztest(im_type = ['AuGe TEM','PdC TEM','PdPtNiAu ADF','PT ADF'], n=1000, membrane = [1,1,1,1,1,1], texture = True,test=True,p = [1,16,20]):

    filter_names = make_names(membrane=membrane,texture=texture,test=test)
    filter_names = filter_names[1:]
    numfilt = len(filter_names)
    stat_array = np.zeros((numfilt,3,len(im_type)))

    for i in range(len(im_type)):
        lab_feat = create_smol_features(im_type[i],num=n, membrane = membrane, texture = texture,test=test, p=p)
        
        index0 = lab_feat[:,0] == 1
        index1 = lab_feat[:,0] == 2

        phile = open(f'{folder}/KSZ test{im_type[i]}.csv','w')

        for ii in range(numfilt):
            data_0 = lab_feat[index0,ii+1]
            data_1 = lab_feat[index1,ii+1]   

            sep = ks_2samp(data_0,data_1)
            z_stat = abs(data_0.mean()-data_1.mean())/(m.sqrt(data_0.std()**2+data_1.std()**2))

            stat_array[ii,:,i] = [sep.statistic,sep.pvalue,z_stat]
            phile.write(f'{filter_names[ii]},{sep.statistic},{sep.pvalue},{z_stat}\n')

        phile.close()

    x = np.arange(numfilt)
    width = 0.75

    fig, ax = plt.subplots(figsize=(0.5*numfilt,4))

    rects1 = ax.bar(x - 3*width/8,stat_array[:,0,0], width/4, label=im_type[0])
    rects2 = ax.bar(x - width/8, stat_array[:,0,1], width/4, label=im_type[1])
    rects3 = ax.bar(x + width/8, stat_array[:,0,2], width/4, label=im_type[2])
    rects4 = ax.bar(x + 3*width/8, stat_array[:,0,3], width/4, label=im_type[3])
    ax.set_xticks(x)
    ax.set_ylabel('KS Statistic')
    ax.set_ylim([0,1])
    ax.tick_params(axis='y', which='major', labelsize=9)
    ax.set_xticklabels(filter_names, fontsize = 9, rotation = 45,ha='right')
    ax.legend(fontsize = 8)
    fig.tight_layout()
    plt.savefig(f'{folder}/KSZ test/KSZ.svg')
    plt.savefig(f'{folder}/KSZ test/KSZ.png')
    return


def pearsonplotmulti(im_type = ['AuGe TEM','PdC TEM','PdPtNiAu ADF','Pt ADF'], num=100000,membrane = [1,0,0,0,0,0], texture = False, intensity=True, edges = True, test=False,p = [1,16,20]):

    filter_names = make_names(membrane=membrane,texture=texture,test=test, intensity=intensity,edges=edges)
    covari = []
    for i in range(4):
        if im_type[i][-3:] == 'TEM':
            p = [4,64,20]
        lab_feat = create_smol_features(im_type[i], num, membrane = membrane, texture = texture,test=test, intensity=intensity,edges=edges,p=p)
        lab_feat = lab_feat[:,1:]
        lab_feat = scale(lab_feat,axis=0)

        #covari.append(np.cov(lab_feat.T))
        covari.append(np.absolute(np.corrcoef(lab_feat.T)))
    
    fig, axs= plt.subplots(2,3,figsize=(covari[0].shape[0]*3/2,covari[0].shape[1]*2/2))
    
    letters = ['A','B','C','D']

    axs[0,2].get_xaxis().set_visible(False)
    axs[0,2].get_yaxis().set_visible(False)
    axs[0,2].axis("off")
    axs[1,2].get_xaxis().set_visible(False)
    axs[1,2].get_yaxis().set_visible(False)
    axs[1,2].axis("off")
    fnt = 12
    count = 0
    for i in range(2):
        for ii in range(2):
            ax = axs[i,ii]
            cov = covari[count]
            #ax.set_title(letters[count], loc='left')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            im = ax.matshow(cov, interpolation='nearest', cmap='YlGn',vmin = 0, vmax = 1)
            if letters[count] == 'A' or letters[count] == 'C':
                ax.get_yaxis().set_visible(True)
                ax.set_yticklabels(filter_names, fontsize = fnt)
            if letters[count] == 'A' or letters[count] == 'B':
                ax.get_xaxis().set_visible(True)
                ax.set_xticklabels(filter_names, fontsize = fnt, rotation = 90)
                ax.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
            for x in range(cov.shape[0]):
                for y in range(cov.shape[1]):
                    c= (str(cov[x,y])[:6]) if cov[x,y] < 0 else (str(cov[x,y])[:5])
                    ax.text(x,y,c, va='center', ha='center', fontsize = 8)
            count +=1

    
    
    fig.subplots_adjust(hspace = 0.1, wspace=0.1)
    cbaxes = fig.add_axes([0.75, 0.1, 0.03, 0.6]) 
    cbar = fig.colorbar(im, ax=axs, cax = cbaxes, shrink=0.7, fraction = 0.5)
    cbar.set_label('Absolute PCC', fontsize = fnt)
    
    
    plt.tight_layout()
    plt.savefig(f'{folder}/Covariance/covariance {lab_feat.shape[1]}.svg')
    plt.savefig(f'{folder}/Covariance/covariance {lab_feat.shape[1]}.png')
    plt.show()
    return


def make_graph(im_type, num=500, membrane = [1,0,0,0,0,0], texture = True,test=False,p = [1,16,20]):
    lab_feat = create_smol_features(im_type, num=num, membrane = membrane, texture = texture,test=test, p=p)
    filter_names = make_names(membrane=membrane,texture=texture,test=test)

    nf = lab_feat.shape[1]-1
    planes = gen_pairs(nf)

    fig, axs = plt.subplots(nf-1,nf-1, figsize=(2.5*(nf-1),2.5*(nf-1)))
    fig.subplots_adjust(hspace = 0, wspace=0)
    fig.suptitle(f'Kernel Planes for {im_type}', y=0.9, fontsize=12, fontweight="bold")


    for i in range(nf-1):
        for ii in range(nf-1):

            for L in range(len(planes)):
                pair = planes[L]
                if i == (pair[0]-1) and ii == (pair[1]-2):
                    x = 1
                    break
                    #notes to plot a graph at this i,ii coord
            if x == 1:
                axs[i,ii].scatter(lab_feat[:,pair[1]],lab_feat[:,pair[0]],c=lab_feat[:,0], cmap='RdYlBu', s=5)
                axs[i,ii].tick_params( labelbottom=False, labelleft = False, bottom=False,left=False)
                axs[i,ii].locator_params(tight=True, nbins=5)
                #plots the scatter on the right one
                
                if i == ii:
                    axs[i,ii].locator_params(tight=True, nbins=5)
                    axs[i,ii].tick_params(axis='both',labelsize=7, labelbottom=True, labelleft = True,bottom=True, left=True)
                    axs[i,ii].set(xlabel=filter_names[pair[1]],ylabel= filter_names[pair[0]])
                    #adds labels along diagonals
                x=0
            else:
                axs[i,ii].set_axis_off()
                #deletes lower triangle plots

    

    #plt.tight_layout()
    plt.savefig(f'{folder}/Kernel Planes/{im_type} {nf} kernelplanes.svg')
    plt.savefig(f'{folder}/Kernel Planes/{im_type} {nf} kernelplanes.png')

    plt.show()

if __name__ == '__main__':
    
    #fig 2.2
    ksztest()

    #fig 2.3
    pearsonplotmulti()

    #fig 2.4
    pearsonplotmulti(membrane=[1,1,1,1,1,1], intensity=False, edges=False)

    #fig 2.5
    make_graph()
