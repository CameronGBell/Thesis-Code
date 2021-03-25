import hyperspy.api as hs
import ParticleSpy.api as ps
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

def test_classifier(afolder, clf):

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
            imagefile = hs.load(os.path.join(folderi,image))
            images.append(imagefile.data)
            thing = np.zeros_like(maskfile)
            thing[maskfile==255] = 1
            thing = thing +1
        else:
            image = mask[:-5]+".dm4"
            imagefile = hs.load(os.path.join(folderi,image))
            images.append(imagefile.data)
            thing = ps.toggle_channels(maskfile[:,:,:3], colors = ['#0000ff','#ff0000'])


        masks.append(thing)

    data = open(f"dicedata{afolder}{str(clff)}.csv",'w')

    data.write("pixel accuracy,dice01,dice11,dice00,dice10\n")

    for i in range(len(masks)):


        mk = np.copy(masks[i])
        im = np.copy(images[i])
        
        if i == 0:
            tic = time.perf_counter()
            _, clf = ps.ClusterTrained(im, mk, clf, sigma = 1, high_sigma = 16, disk_size = 20,)
            toc = time.perf_counter() - tic
            print('trained classifier in {} seconds'.format(toc))
            data.write('{}\n'.format(toc))
            tic = time.perf_counter()
        output = ps.ClassifierSegment(clf, im, sigma = 10, high_sigma = 16, disk_size = 20,)
        im = ps.toggle_channels(output)
        mk = 2-mk
        output = 2-output

        

        maskim = Image.fromarray(im)
        maskim.save("ims/{}".format(names[i]))
        
        print("accuracy: {}".format(ac.check_ground(output,mk)))
        dice = ac.dice(output,mk)
        print("dice {} {}\n     {} {}".format(dice[0,1],dice[1,1],dice[0,0],dice[1,0]))
        data.write("{},{},{},{},{}\n".format(ac.check_ground(output,mk),dice[0,1],dice[1,1],dice[0,0],dice[1,0]))
    toc = time.perf_counter() - tic
    data.write('{}\n'.format(toc))
    print('classified images in {} seconds'.format(toc))
    data.close()
    return

def test_classifier_mult(afolder, clf, n=1, p=[1,16,20]):

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
        import matplotlib.pyplot as plt
        im = imagefile.data
        normed = (im-im.min())*(1000/(im.max()-im.min()))
        
        images.append(normed)
        masks.append(thing)

    data = open(f"dicedata{afolder}{str(clff)}.csv",'w')
    data.write("pixel accuracy,dice01,dice11,dice00,dice10\n")

    tic = time.perf_counter()
    output, clf = ps.ClusterTrained(images[:n], masks[:n], clf, sigma = p[0], high_sigma = p[1], disk_size = p[2])
    #output, clf = ps.ClusterTrained(images[:n], masks[:n], clf, membrane = [1,1,1,1,1,1], texture= True, minimum = True,sigma = p[0], high_sigma = p[1], disk_size = p[2])
    toc = time.perf_counter() - tic
    print('trained classifier in {} seconds'.format(toc))
    data.write('{}\n'.format(toc))
    tic = time.perf_counter()

    if n == 1:
        output = [output]
    for i in range(n,len(masks)):
        output.append(ps.ClassifierSegment(clf, images[i], sigma = p[0], high_sigma = p[1], disk_size = p[2]))
        #output.append(ps.ClassifierSegment(clf, images[i], membrane = [1,1,1,1,1,1], texture= True, minimum = True, sigma = p[0], high_sigma = p[1], disk_size = p[2]))

    toc = time.perf_counter() - tic
    data.write('{}\n'.format(toc))
    print('classified images in {} seconds'.format(toc))

    for i in range(len(masks)):
        
        output[i] = 2-output[i]
        masks[i] = 2-masks[i]

        tempp = ps.toggle_channels(2-output[i])
        maskim = Image.fromarray(tempp)
        maskim.save(names[i])
        
        pixacc = ac.check_ground(output[i],masks[i])
        print("accuracy: {}".format(pixacc))
        dice = ac.dice(output[i],masks[i])
        print("dice {} {}\n     {} {}".format(dice[0,1],dice[1,1],dice[0,0],dice[1,0]))
        data.write("{},{},{},{},{}\n".format(pixacc,dice[0,1],dice[1,1],dice[0,0],dice[1,0]))
    
    data.close()
    return

def test_man_seg(afolder):

    folderm = "C:/Users/cimca/Google Drive (cameronbell2236@gmail.com)/Diamond/Python Stuff/Ground truth/{}/mask".format(afolder)
    folderm2 = "C:/Users/cimca/Google Drive (cameronbell2236@gmail.com)/Diamond/Python Stuff/Ground truth/{}/mask2 EB".format(afolder)

    masks = []
    masks2 = []
    names = []

    for mask in os.listdir((folderm2)):

        maskfile2 = np.asarray(Image.open(os.path.join(folderm2,mask)))
        if afolder == 'AuGe TEM':
            maskfile1 = np.asarray(Image.open(os.path.join(folderm,mask)[:-3]+'tif'))
            thing = np.zeros_like(maskfile1)
            thing[maskfile1==255] = 1
            maskfile1 = thing
               
        else:
            maskfile1 = np.asarray(Image.open(os.path.join(folderm,mask)))
            maskfile1 = ps.toggle_channels(maskfile1[:,:,:3], colors = ['#0000ff','#ff0000'])
            maskfile1 = (maskfile1-1).astype(np.bool_).astype(np.int)

        maskfile2 = ps.toggle_channels(maskfile2[:,:,:3], colors = ['#0000ff','#ff0000'])
        maskfile2 = maskfile2-1
        masks.append(maskfile1)
        masks2.append(maskfile2)

    data = open(f"manual seg{afolder}.csv",'w')
    data.write("pixel accuracy,dice01,dice11,dice00,dice10\n")

    

    for i in range(len(masks)):

        pixacc = ac.check_ground(masks[i],masks2[i])
        dice = ac.dice(masks[i],masks2[i])
        data.write("{},{},{},{},{}\n".format(pixacc,dice[0,1],dice[1,1],dice[0,0],dice[1,0]))
    


    data.close()
    return


#clff = KNeighborsClassifier()
#clff = MLPClassifier(alpha=1, max_iter=10000)

#clff = QuadraticDiscriminantAnalysis()
#clff = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
#clff = SVC(kernel="linear", C=0.5) # this is trash
clff = GaussianNB()
#clff = AdaBoostClassifier()




#test_man_seg('PdC TEM')
#test_man_seg('AuGe TEM')
test_classifier_mult('Pt ADF', clff, p=[1,16,20], n=1)
#test_classifier_mult('PdPtNiAu ADF', clff, p=[1,16,20], n=1)
#test_classifier_mult('PdC TEM', clff, p=[4,64,20], n=1)
#test_classifier_mult('AuGe TEM', clff, p=[4,64,20], n=1)
