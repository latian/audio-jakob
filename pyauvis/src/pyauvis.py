import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep,splev,splint
from PIL import Image
import sys

def afilter(f):
    C1 = 12200**2*f**4
    C2 = f**2+ 20.6**2
    C3 = f**2+12200**2
    C4 = (f**2+107.7**2)**(1./2)
    C5 = (f**2+737.9**2)**(1./2)
    Ra = C1/(C2*C3*C4*C5)
    return 20.*np.log10(Ra)+2.

#vsize = (640,480)
vsize = (80,60)
nop = vsize[0]*vsize[1] # number of pixels

with open('../wav/toccata.wav','rb') as f:
    riffheader = struct.unpack('4si4s',f.read(12))
    if riffheader[0] != 'RIFF': exit('not a valid riff wave file...')
    fmtmeta    = struct.unpack('4si',f.read(8))
    if fmtmeta[1] != 16: exit('length of fmt info not known...')
    fmtdata    = struct.unpack('2h2i2h',f.read(16))
    noc        = fmtdata[1] # number of channels
    sr         = fmtdata[2] # samplerate
    bps        = fmtdata[5] # bits per sample
    datameta   = struct.unpack('4si',f.read(8))
    nos        = datameta[1]/(bps/8) # number of samples
    data       = np.array(struct.unpack('%dh'%nos,f.read(2*nos)))

if noc > 1: print 'warn: multichannel audio currently not implemented.'

print riffheader
print fmtmeta
print fmtdata
print datameta
print 'sample rate:      ',sr
print 'bits per sample:  ',bps
print 'number of samples:',nos

fps  = 25.
nof  = int(float(nos)/sr*fps)
ws   = sr/fps # window size
nfft = 2**int(np.log2(ws)+1)

print 'nfft:',nfft

'''
http://www.sengpielaudio.com/Rechner-notennamen.htm
'''
reff   = 440.                        # Kammerton
n      = np.arange(88,dtype=float)+1 # 88 piano keys
f_0    = 2**((n-49)/12)*440          # Mittenfrequenzen
f_u    = f_0/(2**(1./24.))           # untere Grenzfrequenz
f_o    = f_0*(2**(1./24.))           # obere Grenzfrequenz
hue    = ((n%12.)/12.*256.).astype(int)       # octave-wise same color for same note.
val    = ((np.floor(n/12)+1)/8*256-1).astype(int)  # frequency-dependent brightness
af     = 10**(afilter(f_0)/20.)
#plt.semilogx(f_0,af);plt.show();exit()

# pixel index list:
pinli = list()
for j in xrange(vsize[1]):
    for i in xrange(vsize[0]):
        pinli.append([i,j])

np.random.shuffle(pinli)

img    = Image.new('HSV',vsize,'black')
pixels = img.load()

energy  = np.zeros((len(n),))
energy2 = np.zeros_like(energy)
ecum    = np.zeros_like(energy)
maxe = 0.
perc = 0
for i in xrange(nof): # for each frame
    #np.random.shuffle(pinli)
    G      = 2.*np.abs(np.fft.fft(data[i*ws:(i*ws)+nfft])/nfft)**2
    freqs  = np.abs(np.fft.fftfreq(nfft, 1./sr)[0:(nfft/2+1)])
    G      = G[0:(nfft/2+1)]/((2**15)**2/freqs[1])
    #plt.loglog(freqs,G);plt.show();exit()
    tck    = splrep(freqs,G,k=1)
    for j in xrange(len(n)):
        energy[j] = splint(f_u[j],f_o[j],tck)*af[j]
        if j==0: ecum[j] = energy[j]
        else:    ecum[j] = energy[j]+ecum[j-1]
    energy2 += energy
    sume = np.sum(energy)
    if sume > maxe: maxe=sume
    ecum /= sume
    #plt.plot(ecum);plt.show();exit()
    ct = 0 # current tone
    for j in range(nop): # each pixel
        #print ct, float(j)/nop,ecum[ct]
        #print pinli[j]
        pixels[pinli[j][0],pinli[j][1]] = (hue[ct],255,val[ct])#(hue[ct],int(255.*sume/maxe),val[ct])
        if float(j)/nop>ecum[ct]: ct+=1
    img.convert('RGB').save('img/%06d.png'%i,'PNG')
    perc_new = int(float(i+1)/nof*100)
    if perc_new > perc:
        perc = perc_new
        print '\r%3d%% processed.'%(perc),
        sys.stdout.flush()
#plt.loglog(energy2);plt.show()
