import numpy as np

from ParticleSpy.custom_kernels import membrane_projection
import ParticleSpy.api as ps
from skimage import filters, morphology, util
from skimage.measure import label, regionprops, perimeter
from skimage.exposure import rescale_intensity
from sklearn import preprocessing

import hyperspy.api as hs
import numpy as np
import accuracy as ac
import os
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from PIL import Image
import time
import warnings


def ClusterPercent(image, labels, classifier, number, 
                    intensity = True, 
                    edges = True, 
                    texture = False, 
                    membrane = [1,0,0,0,0,0],
                    test=False,
                    sigma = 1, high_sigma = 16, disk_size = 20):
#just does one image trained on a random percentage of the image pixels

    if len(labels.shape) != 2:
            labels = ps.toggle_channels(labels)

    #makes sure labels aren't empty
    if (labels != 0).any() == True:
        thin_mask = labels.astype(np.float64)
        image = image.data

        features = ps.CreateFeatures(image, intensity=intensity, edges=edges, texture=texture, membrane=membrane, test=test,sigma=sigma, high_sigma=high_sigma, disk_size=disk_size)
        
        area = image.shape[0]*image.shape[1]
        indexes = np.random.randint(0,high=area,size=number)
        frac_array = np.zeros(area)
        frac_array[indexes] = 1
        frac_array = np.reshape(frac_array, (image.shape[0],image.shape[1])).astype(np.bool)
        
        features = np.rot90(np.rot90(features, axes=(2,0)), axes=(1,2))
        #features are num/x/ymini

        training_data = features[:, frac_array].T
        #training data is number of labeled pixels by number of features
        training_labels = thin_mask[frac_array].ravel()
        training_labels = training_labels.astype('int')
        #training labels is labelled pixels in 1D array

        tic = time.perf_counter()
        classifier.fit(training_data, training_labels)
        #will crash for one image with no labels
        toc = time.perf_counter() - tic

        thin_mask = labels.astype(np.float64)
        output = np.copy(thin_mask)
        if (labels == 0).any() == True:
            #train classifier on  labelled data
            data = features[:, thin_mask == 0].T
            #unlabelled data
            pred_labels = classifier.predict(data)
            #predict labels for rest of image

            output[thin_mask == 0] = pred_labels
            #adds predicted labels to unlabelled data

    return output, classifier, toc


def test_classifier_per(clf, imask, number, p=[1,16,20]):

    #tests the classifier trained on a number of an image's pixels. returns the average dice coefficients the number of pixels trained on and 
    
    images = imask[0]
    masks = imask[1]

    output, clf, clock = ClusterPercent(images[0], masks[0], clf, number, sigma = p[0], high_sigma = p[1], disk_size = p[2])

    output = []
    for i in range(10):
        output.append(ps.ClassifierSegment(clf, images[i], sigma = p[0], high_sigma = p[1], disk_size = p[2]))

    alldice = np.zeros((2,2,10))
    for i in range(10):
        #tempp = ps.toggle_channels(output[i]) #for image saving
        output[i] = 2-output[i]
        masks[i] = 2-masks[i]

        alldice[:,:,i] = ac.dice(output[i],masks[i])
    
    #avdice = np.zeros((2,2))
    #avdice[:,:] = alldice[:,:,:].mean(axis=2)

    return alldice, clock



def make_results(clff):
    perlist = [2,4,8,16,32,64,128,256,512,1024,2048,4192,8000,16000,32000]
    if str(clff) == 'KNeighborsClassifier()':
        perlist = [8,16,32,64,128,256,512,1024,2048,4192,8000,16000,32000]
        
    data = open('acc{}.csv'.format(str(clff)),'w')
    data.write('pixels,ln(pix),,time,die00,die01,die10,die11,,time,die00,die01,die10,die11,,time,die00,die01,die10,die11,\n')

    for i in range(len(perlist)):
        die, tim = test_classifier_per('PdC TEM', clff, perlist[i], p=[4,64,20])
        data.write('{},,,{},{},{},{},{},,'.format(perlist[i],tim,die[0,0],die[0,1],die[1,0],die[1,1]))
        die, tim = test_classifier_per('Pt ADF', clff, perlist[i], p=[1,16,20])
        data.write('{},{},{},{},{},,'.format(tim,die[0,0],die[0,1],die[1,0],die[1,1]))
        die, tim = test_classifier_per('PdPtNiAu ADF', clff, perlist[i], p=[1,16,20], membrane=[1,1,1,1,1,1])
        data.write('{},{},{},{},{}\n'.format(tim,die[0,0],die[0,1],die[1,0],die[1,1]))

        print(i)
    data.close()
    return

def make_results_au(clff):
    perlist = [2,4,8,16,32,64,128,256,512,1024,2048,4192,8000,16000,32000]
    if str(clff) == 'KNeighborsClassifier()':
        perlist = [8,16,32,64,128,256,512,1024,2048,4192,8000,16000,32000]

    data = open('acc{}.csv'.format(str(clff)),'w')
    data.write('pixels,ln(pix),,time,die00,die01,die10,die11\n')

    for i in range(len(perlist)):
        die, tim = test_classifier_per('AuGe TEM', clff, perlist[i], p=[4,64,20])
        data.write('{},,,{},{},{},{},{}\n'.format(perlist[i],tim,die[0,0],die[0,1],die[1,0],die[1,1]))

        print(i)
    data.close()
    return


#clff = KNeighborsClassifier()
clff = MLPClassifier(alpha=1, max_iter=10000) #todo
#clff = QuadraticDiscriminantAnalysis()

#clff = SVC(kernel="linear", C=0.5) # this is trash
#clff = GaussianNB()
#clff = AdaBoostClassifier()


if __name__ == '__main__':

    from datetime import datetime
    print(datetime.now())

    #clff = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs = -1)
    #make_results(clff)

    #clff = KNeighborsClassifier()
    #make_results(clff)
    #print('finished {} at {}'.format(str(clff),datetime.now()))



    clff = GaussianNB()
    make_results_au(clff)