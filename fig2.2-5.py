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


folder = 'results/Distribution Analysis'

def create_smol_features(afolder, num, params):
    #creates set of featues using CS and opens the right folders
    folderi = "Ground truth/{}/image".format(afolder)
    folderm = "Ground truth/{}/mask".format(afolder)

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
    mask = masks[0]
    image = images[0]
    features = ps.CreateFeatures(image, params)
    
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

def make_names(params):
    names = ['labels']
    if params.gaussian[0]:
        names.append('Gaussian')
    if params.diff_gaussian[0]:
        names.append('Diff. of Gaussians')
    if params.median[0]:
        names.append('Median')
    if params.minimum[0]:
        names.append('Minimum')
    if params.maximum[0]:
        names.append('Maximum')
    if params.sobel[0]:
        names.append('Sobel')
    if params.hessian[0]:
        names.append( 'Hessian')
    if params.laplacian[0]:
        names.append('laplacian')
    if params.membrane[1:] != [0,0,0,0,0,0]:
        mems = ['Memb. -Sum', 'Memb. -Mean', 'Memb. -StdDev', 'Memb. -Med', 'Memb. -Max', 'Memb. -Min']
        for i in range(6):
            if params.membrane[i+1]:
                names.append(mems[i])
    return names


def ksztest(im_type = ['AuGe TEM','PdC TEM','PdPtNiAu ADF','PT ADF'], n=1000, params=None):

    if params == None:
        parameters = ps.trainableParameters()
    filter_names = make_names(params)
    filter_names = filter_names[1:]
    numfilt = len(filter_names)
    stat_array = np.zeros((numfilt,3,len(im_type)))

    for i in range(len(im_type)):
        if i < 2:
            temp_params = ps.trainableParameters()
            temp_params.setGlobalSigma(4)
            temp_params.setGlobalPrefilter(4)
            temp_params.diff_gaussian[3] = 64
            temp_params.membrane = [[False,1],True,True,True,True,True,True]
            temp_params.hessian[0] = True
            temp_params.laplacian[0] = True
        else:
            temp_params = params

        lab_feat = create_smol_features(im_type[i],n, temp_params)
        
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
    width = 0.5
    letters = ['a','b','c','d']
    count = 0
    fig, axs = plt.subplots(2,2,figsize=(1*numfilt,10))

    for j in range(2):
        for jj in range(2):
            axs[j,jj].bar(x,stat_array[:,0,count], width, label=letters[count], color='#2560A4')
            axs[j,jj].set_xticks(x)
            axs[j,jj].set_ylabel('KS Statistic', fontsize = 12)
            axs[j,jj].set_ylim([0,1])
            axs[j,jj].tick_params(axis='y', which='major', labelsize=12)
            #axs[j,jj].text(-0.1,0.9,letters[count], fontsize = 14)
            axs[j,jj].set_xticklabels(filter_names, fontsize = 12, rotation = 45,ha='right')
            axs[j,jj].set_ylabel('KS Statistic', fontsize = 12)

            count += 1
    fig.tight_layout()
    plt.savefig(f'{folder}/KSZ test/KSZ.svg')
    plt.savefig(f'{folder}/KSZ test/KSZ.png')
    plt.show()
    return




def covarianceplot(im_type, params, num=1000):

    lab_feat = create_smol_features(im_type, num, params)
    filter_names = make_names(params)
    
    lab_feat = lab_feat[:,1:]
    filter_names = filter_names[1:]
    lab_feat = scale(lab_feat,axis=0)

    covari = np.cov(lab_feat.T)

    fig = plt.figure(figsize=(covari.shape[0]*1.5,covari.shape[1]*1.5))
    ax = fig.add_subplot(111)
    grid = fig.add_subplot()
    cax = ax.matshow(covari, interpolation='nearest', cmap='YlGn')

    xaxis = np.arange(len(filter_names))
    ax.set_xticks(xaxis)
    ax.set_yticks(xaxis)
    ax.set_xticklabels(ax.get_xticks(), rotation = 90)
    ax.set_xticklabels(filter_names,fontsize = 15)
    ax.set_yticklabels(filter_names,fontsize = 15)
    for x in range(covari.shape[0]):
        for y in range(covari.shape[1]):

            c= (str(covari[x,y])[:6]) if covari[x,y] < 0 else (str(covari[x,y])[:5])
            grid.text(x,y,c, va='center', ha='center', fontsize = 20)

    plt.tight_layout()
    plt.savefig(f'{folder}/Covariance/covariance {im_type}{lab_feat.shape[1]}.svg')
    plt.savefig(f'{folder}/Covariance/covariance {im_type}{lab_feat.shape[1]}.png')
    plt.show()
    return

def pearsonplotmulti(im_type = ['AuGe TEM','PdC TEM','PdPtNiAu ADF','Pt ADF'], num=100000,params=None):

    if params == None:
        params = ps.trainableParameters()
    filter_names = make_names(params)
    print(filter_names)
    covari = []
    for i in range(4):
        if im_type[i][-3:] == 'TEM':
            p = [4,64,20]
        lab_feat = create_smol_features(im_type[i], num, params)
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

            cmap = plt.cm.YlGn
            norm = plt.Normalize(cov.min(), cov.max())
            matrix = cmap(norm(cov))
            matrix[range(cov.shape[0]), range(cov.shape[1]), :3] = 0.5, 0.5, 0.5
            im = ax.imshow(matrix, interpolation='nearest',vmin = 0, vmax = 1)
            

            if letters[count] == 'A' or letters[count] == 'C':
                ax.set_yticks(range(cov.shape[0]))
                ax.get_yaxis().set_visible(True)
                ax.set_yticklabels(filter_names[1:], fontsize = fnt)
            if letters[count] == 'A' or letters[count] == 'B':
                ax.get_xaxis().set_visible(True)
                ax.set_xticks(range(cov.shape[0]))
                ax.set_xticklabels(filter_names[1:], fontsize = fnt, rotation = 90)
                ax.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
            for x in range(cov.shape[0]):
                for y in range(cov.shape[1]):
                    if x != y:
                        c= (str(cov[x,y])[:5]) if cov[x,y] < 0 else (str(cov[x,y])[:4])
                        ax.text(x,y,c, va='center', ha='center', fontsize = 10)
            count +=1

    
    
    fig.subplots_adjust(hspace = 0.1, wspace=0.1)
    cbaxes = fig.add_axes([0.75, 0.1, 0.03, 0.6]) 
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap), ax=axs, cax = cbaxes, shrink=0.7, fraction = 0.5)
    cbar.set_label('Absolute PCC', fontsize = fnt)
    
    
    plt.tight_layout()
    plt.savefig(f'{folder}/Covariance/covariance {lab_feat.shape[1]}.svg')
    plt.savefig(f'{folder}/Covariance/covariance {lab_feat.shape[1]}.png')
    plt.show()
    return


def make_graph(im_type, num=500, params=None):
    if params == None:
        params = ps.trainableParameters()
    lab_feat = create_smol_features(im_type, num,params)
    filter_names = make_names(params)[1:]

    nf = lab_feat.shape[1]-1
    planes = gen_pairs(nf)

    fig, axs = plt.subplots(nf-1,nf-1, figsize=(2.5*(nf-1),2.5*(nf-1)))
    fig.subplots_adjust(hspace = 0, wspace=0)
    #fig.suptitle(f'Kernel Planes for {im_type}', y=0.9, fontsize=12, fontweight="bold")
    cmap = mpl.colors.ListedColormap(['#F64141','#2560A4'])

    for i in range(nf-1):
        for ii in range(nf-1):

            for L in range(len(planes)):
                pair = planes[L]
                if i == (pair[0]-1) and ii == (pair[1]-2):
                    x = 1
                    break
                    #notes to plot a graph at this i,ii coord
            if x == 1:

                axs[i,ii].scatter(lab_feat[:,pair[1]],lab_feat[:,pair[0]],c=lab_feat[:,0], cmap=cmap, s=5)
                axs[i,ii].tick_params( labelbottom=False, labelleft = False, bottom=False,left=False)
                axs[i,ii].locator_params(tight=True, nbins=5)
                #plots the scatter on the right one
                
                if i == ii:
                    axs[i,ii].locator_params(tight=True, nbins=5)
                    axs[i,ii].tick_params(axis='both',labelsize=7, labelbottom=False, labelleft = False,bottom=False, left=False)
                    axs[i,ii].text(-0.5,0.5,filter_names[i], va='center', ha='center', fontsize = 18,rotation=-45, transform=axs[i,ii].transAxes)
        
                    if i == 7:
                        axs[i,ii].text(0.5,-0.5,filter_names[i+1], va='center', ha='center', fontsize = 18,rotation=-45, transform=axs[i,ii].transAxes)
                    
                    axs[i,ii].set_yticklabels([])
                    axs[i,ii].set_xticklabels([])
                    #adds labels along diagonals
                x=0

            else:
                axs[i,ii].set_axis_off()
                #deletes lower triangle plots
            

    

    #plt.tight_layout()
    plt.savefig(f'{folder}/Kernel Planes/{im_type} {nf} kernelplanes.svg')
    plt.savefig(f'{folder}/Kernel Planes/{im_type} {nf} kernelplanes.png')

    plt.show()


i = 1



if i == 0:
    params = ps.trainableParameters()
    pearsonplotmulti(params=params)
    params.membrane = [[False,1],True,True,True,True,True,True]
    params.gaussian[0] = False
    params.diff_gaussian[0] = False
    params.median[0] = False
    params.minimum[0] = False
    params.maximum[0] = False
    params.sobel[0] = False
    pearsonplotmulti(params=params)
    
elif i == 1:
    params = ps.trainableParameters()
    params.membrane = [[False,1],True,True,True,True,True,True]
    params.hessian[0] = True
    params.laplacian[0] = True
    ksztest(params=params)
elif i == 2:
    params = ps.trainableParameters()
    
    params.hessian[0] = True
    params.laplacian[0] = True

    make_graph('PDPtNiAu ADF', params=params)
