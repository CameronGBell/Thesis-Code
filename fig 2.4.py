import hyperspy.api as hs
import ParticleSpy.api as ps
import numpy as np
import accuracy as ac
import os
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import time
from ParticleSpy.segptcls import process

afolder = 'AuGe TEM'

folderi = "C:/Users/cimca/Google Drive (cameronbell2236@gmail.com)/Diamond/Python Stuff/Ground truth/{}/image".format(afolder)
folderm = "C:/Users/cimca/Google Drive (cameronbell2236@gmail.com)/Diamond/Python Stuff/Ground truth/{}/mask".format(afolder)

images = []
masks = []
names = []

i= 0
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
        image = mask[:-7]+".dm4"
        imagefile = hs.load(os.path.join(folderi,image))
        images.append(imagefile.data)
        thing = ps.toggle_channels(maskfile[:,:,:3], colors = ['#0000ff','#ff0000'])
    if i == 0:
        imtest = imagefile
        i += 1



    masks.append(thing)


data = open("autosegdata{}.csv".format(afolder),'w')
data.write("pixel accuracy,dice01,dice11,dice00,dice10\n")

for i in range(len(masks)):


    mk = np.copy(masks[i])
    im = np.copy(images[i])
    
    if i == 0:
        tic = time.perf_counter()

        
        
        out = ps.SegUI(imtest)
        params = out.params

        toc = time.perf_counter() - tic
        print('trained classifier in {} seconds'.format(toc))
        tic = time.perf_counter()
    
    output = process(im, params)
    output[output != 0] = 1


    immask = ps.toggle_channels(output)
    mk = mk-1
    

    maskim = Image.fromarray(immask)
    maskim.save(names[i])
    
    print("accuracy: {}".format(ac.check_ground(output,mk)))
    #dice uses 0 and 1 for bck and part
    dice = ac.dice(output,mk)
    print("dice {} {}\n     {} {}".format(dice[0,1],dice[1,1],dice[0,0],dice[1,0]))
    data.write("{},{},{},{},{}\n".format(ac.check_ground(output,mk),dice[0,1],dice[1,1],dice[0,0],dice[1,0]))

toc = time.perf_counter() - tic
print('classified images in {} seconds'.format(toc))
data.close()