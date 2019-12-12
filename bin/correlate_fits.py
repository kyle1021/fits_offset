#!/usr/bin/env python

#from analysisconf import *
import numpy as np
import sys, os.path
from subprocess import call
from glob import glob
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.signal import correlate
from astropy.stats import sigma_clip
from astropy.io import fits
from astropy.stats import sigma_clip, biweight_scale
#from OTfunc import *


def kern(k='disc5', norm=True):
    '''
    produces a small kernel image to blur the catalog of stars
    into a mock image

    norm = True --> kernel is integrated to unity
    nrom = False --> kernel has a peak of unity

    '''
    if (k == 'disc5' or k == 'd5' or k == 'd'):
	a = 5.	# disc radius
	m = 5	# mid-point
	n = 11	# size
	img = np.zeros((n,n))
	for i in range(n):
	    y = float(i-m)
	    for j in range(n):
		x = float(j-m)
		r = np.sqrt(x**2 + y**2)
		if (r<=a):
		    img[i,j] = 1.

    elif (k == 'gauss3' or k == 'g3' or k == 'g'):
        s = 3   # sigma = 3 pix
        m = 15  # mid-point pix
        n = 31  # size in pix
	img = np.zeros((n,n))
	for i in range(n):
	    y = float(i-m)
	    for j in range(n):
		x = float(j-m)
		r = np.sqrt(x**2 + y**2)
                img[i,j] = np.exp(-r**2/2./s**2)

    else:
	print 'unknown kernel type:', k
	return None

    if (norm):
        img /= img.sum()
    else:
        img /= np.abs(img).max()

    return img


def Xoffset(imgs, autop, ds, refid=0, upward=False, savedir=None, dmax=0.):
    '''
    cross-correlate the images and find out the offset (relative to the refid)
    <input>
    imgs:   float, shape(nimg, ny, nx); the input images
    autop:  float, shape(nimg,); the peaks of auto-correlation of images
    ds:     float, pixel size in arcmin
    refid:  integer, scalar; specify which image as reference
    upward: bool, scalar; True --> calculate offsets only for image id greater than the refid
                          False --> calculate offsets for all image id
    dmax:   float, the max offset acceptable (a search radius in the xcorr space)

    <return>
    offsets: complex, shape(nimg,); dx + j dy
        (if upward==True, offsets[i]=0+0j for i<=refid)
        (if upward==False, offsets[refid]=0+0j)
    rpeaks:  float, shape(nimg,); xcorr_ij.max()
    npeaks:  float, shape(nimg,); xcorr_ij.max()/sqrt(auto_i.max * auto_j.max)
    '''
    #print 'in Xoffset: savedir =', savedir
    #print 'in Xoffset: imgs.shape =', imgs.shape

    (nv, ny, nx) = imgs.shape
    cx = (nx - (nx%2)) / 2
    cy = (ny - (ny%2)) / 2
    extent = [-cx, nx-cx-1, -cy, ny-cy-1]
    # grid coordinates in arcmin
    gx = ds * (np.arange(nx) - cx)
    gy = ds * (np.arange(nx) - cy)
    gX, gY = np.meshgrid(gx,gy)     # default indexing='xy'; put faster dimension in the first array
    darr = np.sqrt(gX**2 + gY**2)   # 2D array of pixel relative distance to central pixel

    offsets_x = np.zeros(nv)
    offsets_y = np.zeros(nv)
    rpeaks  = np.zeros(nv) # raw x-corr peak strength
    npeaks  = np.zeros(nv) # normalized x-corr peak strength

    #if (not imgs[refid].max() == 1.):       # return zeros arrays if imgs[refid] is a dummy image
    if (autop.mask[refid] == False):       # return zeros arrays if imgs[refid] is a dummy image
        if (upward):
            j0 = refid
            rpeaks[j0] = autop[j0]
            npeaks[j0] = 1.
        else:
            j0 = -1


        for j in range(j0+1, nv):
            #if (not imgs[j].max() == 1.):   # if the variant image is dummy, use zeros (default values)
            if (autop.mask[j] == False):   # if the variant image is dummy, use zeros (default values)
                corr = correlate(imgs[j], imgs[refid], mode='same')
                ncorr = corr / np.sqrt(autop[j] * autop[refid])
                mcorr = np.ma.array(corr, mask=np.zeros_like(corr, dtype=bool))
                mncorr = np.ma.array(ncorr, mask=np.zeros_like(ncorr, dtype=bool))
                if (dmax > 0.):     # if a max offset is specified
                    mcorr.mask = darr > dmax
                    mncorr.mask = darr > dmax

                rpeaks[j] = mcorr.max()
                npeaks[j] = mncorr.max()
                my, mx = np.unravel_index(corr.argmax(), corr.shape)
                #print my-cy, mx-cx, '%.4e' % cpeaks[-1]
                #offsets[j] = mx-cx + 1.j*(my-cy)
                offsets_x[j] = ds * (mx-cx)
                offsets_y[j] = ds * (my-cy)

                #print 'in Xoffset: j, offsets[j], my, mx', j, offsets[j]
                if (savedir):
                    png = '%s/cross_%03d_ref_%03d.png' % (savedir, j, refid)
                    plt.imshow(corr, origin='lower', extent=extent)
                    plt.xlabel('x offset (pix)')
                    plt.xlabel('y offset (pix)')
                    plt.savefig(png)
                    plt.close()

    return offsets_x, offsets_y, rpeaks, npeaks



def Xfull(imgs, autop, ds, refid=0, savedir=None, dmax=0.):
    '''
    full cross-corr of all images (top-right triangle)
    produce a map of the correlation strength
    '''
    #print 'debug: in Xfull'
    #print autop

    print 'Xfull: ref =', refid
    nv = autop.size
    offset_full_x = np.zeros((nv, nv))
    offset_full_y = np.zeros((nv, nv))
    rpeaks_full = np.zeros((nv, nv))
    npeaks_full = np.zeros((nv, nv))

    print 'Xfull: nimg = ', nv
    for j in range(nv):
        if (autop.mask[j]):
            continue    # skip if img is masked
        print '  ref = ', j
	offsets_x, offsets_y, rpeaks, npeaks = Xoffset(imgs, autop, ds, refid=j, upward=True, savedir=savedir, dmax=dmax)
	offset_full_x[j] = offsets_x
	offset_full_y[j] = offsets_y
	rpeaks_full[j] = rpeaks
	npeaks_full[j] = npeaks

    return offset_full_x, offset_full_y, rpeaks_full, npeaks_full


def trim_xcorr(xcorr, nimg):
    '''
    input:
        xcorr       shape(nimg, nimg)
    output:
        xtrim       shape(ncorr,)
                    ncorr = nimg * (nimg-1) / 2
    '''
    ncorr = nimg * (nimg-1) / 2
    xtrim = []
    b = -1
    for i in range(nimg-1):
        for j in range(i+1, nimg):
            b += 1
            xtrim.append(xcorr[i,j])

    return np.array(xtrim)
            


def save_xcorr(ftxt, offset_full_x, offset_full_y, rpeaks_full, npeaks_full):
    '''
    given a full xcorr matrix (nimg, nimg)
    save only the unique correlation (top half triangle) to a txt file
    '''
    nimg, nimg = rpeaks_full.shape

    TXT = open(ftxt, 'w')
    print >> TXT, '# img1   img2   offset_x   offset_y    rpeak   npeak'
    for i1 in range(nimg-1):
        for i2 in range(i1+1, nimg):
            print >> TXT, '%03d   %03d   % 7.2f   % 7.2f   % 10.0f   % 10.4f' % (i1, i2, offset_full_x[i1, i2], offset_full_y[i1, i2], rpeaks_full[i1, i2], npeaks_full[i1, i2])
    TXT.close()
    # no need to return anything
    return


def load_xcorr(ftxt, nimg):
    '''
    load the trimmed xcorr data and rearrange to full matrix (nimg, nimg)
    '''
    offset_full_x = np.zeros((nimg, nimg))
    offset_full_y = np.zeros((nimg, nimg))
    rpeaks_full = np.zeros((nimg, nimg))
    npeaks_full = np.zeros((nimg, nimg))

    tab = np.loadtxt(ftxt, unpack=True)
    r = -1
    for i1 in range(nimg-1):
        for i2 in range(i1+1, nimg):
            r += 1
            offset_full_x[i1, i2] = tab[2, r]
            offset_full_y[i1, i2] = tab[3, r]
            rpeaks_full[i1, i2] = tab[4, r]
            npeaks_full[i1, i2] = tab[5, r]

    return offset_full_x, offset_full_y, rpeaks_full, npeaks_full


def myVAR(X):
    '''
    X is a masked array
    return the mean deviation^2 (without removing the mean) of the valid values
    '''
    return np.ma.sum(X**2)/np.ma.count(X)


def pick_ref(X, Y):
    '''
    pick a reference image that would minimize the RMS of X and Y
    (X and Y are masked arrays)
    '''
    nimg     = X.size
    var      = np.ma.zeros(nimg)
    var.mask = np.zeros(nimg, dtype=bool)
    #print X, Y
    for i in range(nimg):
        if (X.mask[i] or Y.mask[i]):
            var.mask[i] = True

        else:
            var[i] += myVAR(X - X[i])
            var[i] += myVAR(Y - Y[i])
    #print np.ma.sqrt(var)

    return var.argmin()



def rotate(x, y, theta, unit='rad', sense='vector'):
    '''
    rotate the input vector (x,y) by an angle theta

    input:
        x, y        arrays of original coordinates of the vectors
        theta       rotation angle

        [optional]
        unit        'deg' or 'rad'; the unit of theta
        sense       'vector' or 'axes'; rotate the vector or the axes counter-clockwise

    return:
        x2, y2      arrays of new coordinates of the vectors

    '''

    #-- parsing and preprocessing 
    x = np.array(x, ndmin=1)
    y = np.array(y, ndmin=1)
    n = len(x)

    if (unit.startswith('deg')):    # 'deg' or 'degree'
        angle = theta / 180. * np.pi
    elif (unit.startswith('rad')):  # 'rad' or 'radian'
        angle = theta
    else:
        print 'unknown unit in rotate():', unit
        sys.exit()

    if (sense == 'vector'):
        angle *= 1.
    elif (sense == 'axes'):
        angle *= -1.
    else:
        print 'unknown sense in rotate():', sense
        sys.exit()


    #-- rotation
    x2 =  np.cos(angle) * x - np.sin(angle) * y
    y2 =  np.sin(angle) * x + np.cos(angle) * y

    return x2, y2


def resample(rimg, img, X2, Y2):
    '''
    given a image and the transformed pixel coordinates
    reample and add to a new image frame

    input:
        rimg        shape(my, mx); the new image to be returned (might be all zeros)
        img         shape(ny, nx); the original image
        X2          shape(ny, nx); the img x coordinate (float) in the new image frame
        Y2          shape(ny, nx); the img y coordinate (float) in the new image frame

    return:
        rimg
    '''
    my, mx = rimg.shape

    # X-lower
    lX = np.floor(X2).astype(int)
    sX = X2 - lX        # the relative position of img pixel to the new X-axis
    # Y-lower
    lY = np.floor(Y2).astype(int)
    sY = Y2 - lY        # the relative position of img pixel to the new Y-axis

    # protect the pixel range
    lX[lX<0] = 0
    lX[lX>mx-2] = mx-2
    lY[lY<0] = 0
    lY[lY>my-2] = my-2

    # lower-left corner
    rimg[lY, lX] += (1.-sX) * (1.-sY) * img
    # lower-right corner
    rimg[lY, lX+1] += sX * (1.-sY) * img
    # upper-left corner
    rimg[lY+1, lX] += (1.-sX) * sY * img
    # upper-right corner
    rimg[lY+1, lX+1] += sX * sY * img

    return rimg


def resamp_rotate(img, theta, unit='rad', sense='vector'):
    '''
    rotate the input vector (x,y) by an angle theta

    input:
        img         shape(ny, nx); unrotated image
        theta       rotation angle

        [optional]
        unit        'deg' or 'rad'; the unit of theta
        sense       'vector' or 'axes'; rotate the vector or the axes counter-clockwise

    return:
        rimg        shape(2*nmax, 2*nmax); rotated and resampled image

    '''

    #-- parsing and preprocessing 
    ny, nx = img.shape
    cx = (nx - nx%2) / 2
    cy = (ny - ny%2) / 2
    rmax = np.sqrt(cx**2 + cy**2)
    nmax = int(rmax)
    rimg = np.zeros((2*nmax, 2*nmax))   # new center = (nmax, nmax)


    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    X, Y = np.meshgrid(x, y)

    RX, RY = rotate(X-cx, Y-cy, theta, unit=unit, sense=sense)
    X2 = RX + nmax
    Y2 = RY + nmax

    rimg = resample(rimg, img, X2, Y2)

    return rimg



if (__name__ == '__main__'):

    inp	        = sys.argv[0:]
    pg	        = inp.pop(0)
    refid       = 0
    conf_file   = 'ref_image.conf.auto'
    use_conf    = True
    full	= False     # whether to derive full cross-corr
    load_full   = False     # whether to load saved full cross-corr
    save_full   = True      # whether to save the full cross-corr
    auto_ref    = True      # whether to pick a ref image automatically
    use_cat     = False
    tail        = '.fits'

    save_png    = False
    savedir     = 'check'
    do_search   = False
    win	        = kern('gauss3')
    align       = 'XY'
    #align       = 'RADEC'
    dmax        = 0.
    skypol	= 180.


    OTn_valid = [
                    'OT3_OT3', 
                    'OT4_10054118',
                    'ST-i_12232',
                    'ST-i_12234',
                    'ST-i_12239',
                    'ST-i_12244'
                ]


    usage	    = '''
    usage: %s <OT_name> [options]

        <OT_name> = one of [OT3_OT3, OT4_10054118, ST-i_12232, ST-i_12234, ST-i_12239, ST-i_12244]

        options are:
	--full			derive the full cross-corr of all images

        -l <TXT>                (together with --full) load saved full offsets. do not re-derive.

        --plot		        save diagnostic plots

        --conf <conf_file>      specify another ref_image.conf
                                default is to read from / save to %s

        --ref NUM               specify the ref image number
                                (default is the the zero-th image as ref image)

        --align <XY|RADEC>      choose whether to measure offset in X/Y or RA/DEC coordinates
                                the default is 'XY'

        --dmax <DMAX>           specify the max offset that can be true
                                flag offsets that are larger

        --cat                   use .cat instead of .fits
                                (default use .fits)

	--skypol SPOL		specify the skypol the images were taken with
				default: 180.        

    ''' % (pg, conf_file)

    if (len(inp) < 1):
        print usage
        sys.exit()

    while (inp):
        k = inp.pop(0)
        if (k == '--plot'):
            save_png = True
	elif (k == '--full'):
	    full = True
        elif (k == '-l'):
            ftxt = inp.pop(0)
            load_full = True
            save_full = False
        elif (k == '--conf'):
            try:
                conf_file = inp.pop(0)
            except:
                print 'invalid filename for conf_file'
                sys.exit()
            if (not os.path.isfile(conf_file)):
                print 'error specifying conf_file:', conf_file
                sys.exit()
        elif (k == '--ref'):
            try:
                refid = int(inp.pop(0))
                use_conf = False
                auto_ref = False
            except:
                print 'invalid ref_id'
                sys.exit()
        elif (k == '--align'):
            try:
                align = inp.pop(0)
                if (align == 'XY' or align == 'RADEC'):
                    pass
                else:
                    print 'unknown align mode:', align
                    sys.exit()
            except:
                print '--align needs and argument'
                sys.exit()
        elif (k == '--dmax'):
            try:
                dmax = float(inp.pop(0))
            except:
                print 'error parsing dmax'
                sys.exit()
	elif (k == '--skypol'):
	    try:
		skypol = float(inp.pop(0))
	    except:
		sys.exit('error getting skypol.')
        elif (k == '--cat'):
            use_cat = True
            tail = '.cat'
        elif (k.startswith('-')):
            sys.exit('unknown option: %s' % k)
        else:
            OTn = k
            if (not OTn in OTn_valid):
                print 'invalid OT name (e.g. ST-i_12232):', OTn
                sys.exit()
            OTt, OTsn =  OTn.split('_')


    call(['mkdir', '-p', savedir])


    #-- OT specific settings
    if (OTt == 'OT3'):
        nx = 657
        ny = 495
    elif (OTt == 'OT4'):
        nx = 765
        ny = 510
    elif (OTt == 'ST-i'):
        nx = 648
        ny = 486
    else:
        print 'undefined OT type.'
        sys.exit()

    # central index, starting from 0
    cx = (nx - (nx%2)) / 2
    cy = (ny - (ny%2)) / 2
    rmax = np.sqrt(cx**2 + cy**2)
    nmax = int(rmax)


    #-- build list of image/star-catalogs
    files = glob('*_s*_u*%s' % tail)
    files.sort()
    #files = files[:15]  # testing
    nimg = len(files)
    if (True): # debug
        flog1 = '%s/list_id.log' % savedir
        LOG1 = open(flog1, 'w')
        print >> LOG1, '# id,  filename'
        for fid, fname in enumerate(files):
            print >> LOG1, ' % 3d   %s' % (fid, fname)
        LOG1.close()

    # the un-rotated images
    imgs = np.ones((nimg, ny, nx))
    # the rotated images
    if (align == 'XY'):
        rimgs = np.ones((nimg, ny, nx))
    elif (align == 'RADEC'):
        rimgs = np.ones((nimg, nmax*2, nmax*2))



    ytla = False
    if (ytla):
        #-- build a list of datetime of the images according to the yymmdd (fake to UT 05:00:00 or HST 7pm) 
        dtarr = []
        for f in files:
            tmp = f.split('_')
            try:
                if (tmp[0] != OTsn):
                    print 'inconsistent OT serial:', f
                    sys.exit()
                yymmdd = tmp[1]
                dt = datetime.strptime(yymmdd, '%y%m%d')
                dt += timedelta(0, 3600*5)
                dtarr.append(dt)
            except:
                print 'error parsing catalog name:', f
                sys.exit()
        dtarr = np.array(dtarr)

        #-- obtain the CCD calibration database
        dtcal_dict, angcal_dict, pixcal_dict = getCCDcal()
        dtcal  = dtcal_dict[OTn]        # masked array of datetime obj
        angcal = angcal_dict[OTn]       # masked array of floats
        pixcal = pixcal_dict[OTn]       # scalar float

        oangle = matchCCDcal(dtarr, dtcal, angcal)  # the CCD orientation at each dtarr timestamp
    else:
        pixcal = 0.255



    #-- read specified ref image name (yymmdd_s??_u??) and define refid
    if (use_conf and os.path.isfile(conf_file) and os.path.getsize(conf_file)>0):
        try:
            #tmp = np.loadtxt(conf_file, usecols=(0,), dtype=stri, ndmin=1)
            #print 'conf_file > ', tmp
            #fref = tmp[0]
            with open(conf_file, 'r') as CONF:
                fref = CONF.readline()
                fref.strip()
            print 'ref image:', fref
            for fid, fname in enumerate(files):
                if (fref in fname):
                    refid = fid
                    auto_ref = False
        except:
            print 'error parsing:', conf_file
            sys.exit()



    #-- preprocess images/star-catalogs
    #   the images are reconstructed from star-catalogs
    #   so that we can have the freedom to easily reorient the image
    #   e.g. X/Y vs. RA/DEC ... etc.
    #
    #align = 'XY'
    #align = 'RADEC'

    print 'preprocessing...'
    print 'orientation:', align
    ipeaks = np.ma.array(np.ones(nimg), mask=np.zeros(nimg, dtype=bool))
    scales = np.ma.array(np.ones(nimg), mask=np.zeros(nimg, dtype=bool))

    autop = np.ma.zeros(nimg)
    autop.mask = np.zeros(nimg, dtype=bool)

    for i in range(nimg):
        f = files[i]
        if (use_cat):                       # use .cat
            if (os.path.getsize(f) > 0):
                tmp = imgs[i];
                flux, x, y = np.loadtxt(f, unpack=True, ndmin=2)
                for j in range(len(flux)):
                    jy = int(y[j]-1.)
                    jx = int(x[j]-1.)
                    if (jy<ny and jx<nx):
                        tmp[jy, jx] = flux[j]
                        
                imgs[i] = correlate(tmp, win, mode='same')
                ipeaks[i] = imgs[i].max()

            else:
                ipeaks.mask[i] = True

        else:                               # use .fits instead
            hdul = fits.open(f)
            # in CCD X/Y coordinates 
            imgs[i] = hdul[0].data.astype(float)
            imgs[i] -= np.median(imgs[i])   # reset img median to zero
            hdul.close()

            #ipeaks[i] = imgs[i].max()
            s = biweight_scale(imgs[i].flatten())
            #print 'i, scale:', i, s
            scales[i] = s

            # re-define ipeaks as the sum of all 5-sigma peaks
            ipeaks[i] = imgs[i][np.abs(imgs[i]) > 5.*s].sum()

            scut = (5. * s) * 5.    # the minimum acceptable number of 5-sigma pixels
                                    # consider as noise peaks if below this threshold
            if (ipeaks[i] < scut):
                ipeaks.mask[i] = True

        if (ipeaks.mask[i]):
            autop.mask[i] = ipeaks.mask[i]
        else:
            auto = correlate(imgs[i], imgs[i], mode='same')
            autop[i] = auto.max()

        if (align == 'RADEC'):
	    theta = oangle[i] + (skypol - 180.)
            rimgs[i] = resamp_rotate(imgs[i], theta, unit='deg', sense='vector')
        elif (align == 'XY'):
            rimgs[i] = imgs[i]


        if (save_png):
            png = '%s/img_%03d.png' % (savedir, i)
            #plt.imshow(imgs[i], origin='lower')
            #plt.imshow(np.log(rimgs[i]), vmin=1, vmax=10, origin='lower')
            tmp = rimgs[i].copy()
            tmp[tmp<0.] = 1.
            plt.imshow(np.log(tmp), vmin=1, vmax=10, origin='lower')
            plt.savefig(png)
            plt.close()
	    #if (i > 10):
		#sys.exit()

    print 'data loaded.'


    print biweight_scale(scales)
    scales = sigma_clip(scales, sigma=5.)
    #for i in range(nimg):
    #    print i, scales.data[i], scales.mask[i]
    autop.mask = np.logical_or(autop.mask, scales.mask)

    ipeaks_c = sigma_clip(ipeaks)
    iflag = ipeaks_c.mask	# 0 = good; 1 = bad


    if (full):
        if (load_full and os.path.isfile(ftxt)):
            #offset_x, offset_y, rpeaks, npeaks = np.loadtxt(ftxt, usecols=(2,3,4,5), unpack=True)
            offset_full_x, offset_full_y, rpeaks_full, npeaks_full = load_xcorr(ftxt, nimg)

        else:
            if (save_png):
                pngdir = savedir
            else:
                pngdir = None

            offset_full_x, offset_full_y, rpeaks_full, npeaks_full = Xfull(rimgs, autop, pixcal, dmax=dmax, savedir=pngdir)
            if (save_full):
                ftxt = 'Xcorr_offsets.txt'
                save_xcorr(ftxt, offset_full_x, offset_full_y, rpeaks_full, npeaks_full)

                ## plot the x-corr peak strength
                png = 'Xcorr_strength.png'
                fig, axs = plt.subplots(2, 1, figsize=(10, 12))

                # top panel, raw peaks
                im0 = axs[0].imshow(rpeaks_full)
                axs[0].set_title('raw x-corr')
                axs[0].set_ylabel('image_i')
                axs[0].set_xlabel('image_j')
                cb0 = plt.colorbar(im0, ax=axs[0])

                # bottom panel, normalized peaks
                im1 = axs[1].imshow(npeaks_full)
                axs[1].set_title('normalized x-corr')
                axs[1].set_ylabel('image_i')
                axs[1].set_xlabel('image_j')
                cb1 = plt.colorbar(im1, ax=axs[1])

                fig.tight_layout()
                fig.savefig(png)
                plt.close(fig)

        ## full Xcorr in X/Y coordinates loaded

        ## trim the data 
        offset_x = trim_xcorr(offset_full_x, nimg)
        offset_y = trim_xcorr(offset_full_y, nimg)
        rpeaks   = trim_xcorr(rpeaks_full, nimg)
        npeaks   = trim_xcorr(npeaks_full, nimg)
        #print rpeaks_full
        #print rpeaks
        #sys.exit()

        ncorr = offset_x.size

        ## NOTE:
        ## doing x-corr of XY images and then rotate the offsets only works when the rotation of both images are identical
        ## there is no guarantee this is true general.
        ## it is thus safer to rotate the image into RADEC and then do x-corr
        if (False):
            ## rotate to the correct align
            if (align == 'RADEC'):
                medangle = np.median(oangle) + (skypol - 180.)
                for i in range(ncorr):
                    offset_x[i], offset_y[i] = rotate(offset_x[i], offset_y[i], medangle, unit='deg', sense='vector')
        

        ## solve the image offset
        DX = np.ma.array(offset_x, mask=np.zeros(ncorr, dtype=bool))
        DY = np.ma.array(offset_y, mask=np.zeros(ncorr, dtype=bool))
        A = np.zeros((ncorr, nimg))
        print 'max offset (arcmin):', dmax
        corr = -1
        for i in range(nimg-1):
            for j in range(i+1, nimg):
                corr += 1
                d = np.sqrt(offset_x[corr]**2 + offset_y[corr]**2)
                #print rpeaks[corr], offset_x[corr], offset_y[corr], d
                if (npeaks[corr] == 1. or npeaks[corr] == 0.):    # some correlations are skipped if one of the images is dummy
                    #print 'npeaks flag: i, j', i, j
                    DX.mask[corr] = True
                    DY.mask[corr] = True
                elif (dmax > 0. and d > dmax):
                    DX.mask[corr] = True
                    DY.mask[corr] = True
                else:
                    A[corr, i] =  1.
                    A[corr, j] = -1.
                    #print i, j, A[corr,i], A[corr,j]

        Ainv = np.linalg.pinv(A, rcond=1.e-6)
        X = np.ma.dot(Ainv, DX)
        Y = np.ma.dot(Ainv, DY)
        X.mask[np.ma.abs(X) < 1.e-6] = True
        Y.mask[np.ma.abs(Y) < 1.e-6] = True
        if (auto_ref):
            ref2 = pick_ref(X, Y)

            fname = files[ref2]
            tmp = fname.strip(tail).split('_')
            refname = '_'.join(tmp[1:4])
            REF = open(conf_file, 'w')
            print >> REF, refname
            REF.close()

        else:
            red2 = refid
        if (not X.mask[ref2]):
            X -= X[ref2]
        if (not Y.mask[ref2]):
            Y -= Y[ref2]

        fsol = 'Xcorr_solution.txt'
        SOL = open(fsol, 'w')
        print >> SOL, '#ref:  %d' % ref2
        print >> SOL, '#img_id  filename                 offset_ra   offset_dec'
        for i in range(nimg):
            print >> SOL, '%03d      %s     % 9.2f   % 9.2f' % (i, files[i].strip(tail), X[i], Y[i])
        SOL.close()

        # model and residual
        for t in range(2):
            if (t == 0):    # RA or X
                Z = X
                DZ = DX
                name = 'ra'
            elif (t == 1):  # DEC or Y
                Z = Y
                DZ = DY
                name = 'dec'
            M = np.ma.dot(A, Z)
            R =  DZ - M
        
            fres = 'Xcorr_residual_%s.txt' % name
            RES = open(fres, 'w')
            print >> RES, '#img1  img2  offset   model   residual   mask'
            corr = -1
            for i in range(nimg-1):
                for j in range(i+1, nimg):
                    corr += 1
                    print >> RES, '%03d   %03d   % 7.2f   % 7.2f   % 7.2f   %1d' % (i, j, DZ[corr], M[corr], R[corr], DZ.mask[corr])
            print >> RES, 'rms         % 7.2f   % 7.2f   %7.2f' % (np.sqrt(myVAR(DZ)), np.sqrt(myVAR(M)), np.sqrt(myVAR(R)))
            RES.close()
        

    else:
	print 'cross-correlating...'
	print 'refid = ', refid
	offsets_x, offsets_y, rpeaks, npeaks = Xoffset(rimgs, autop, pixcal, refid=refid, savedir=savedir, dmax=dmax)
	print offsets_x, offset_y


	flog2 = '%s/list_offsets.log' % savedir
	LOG2 = open(flog2, 'w')
	print >> LOG2, '# refid: ', refid
	print >> LOG2, '# id,  name,  peak, flagged, offset_x, offset_y, auto_peak'
	for i in range(len(ipeaks)):
	    f = files[i]
	    tmp = f.strip(tail).split('_')
	    name = '_'.join(tmp[1:4])
	    #print '% 4d' % i, '% 8.0f' % ipeaks[i], iflag[i]
	    print >> LOG2, '% 4d   %14s   % 6.0f\t  %-5s\t' % (i, name, ipeaks_c[i], iflag[i]), '   % 5.1f   % 5.1f' % (offsets_x[i], offsets_y[i]), '   % 8.4e' % autop[i]

	LOG2.close()

