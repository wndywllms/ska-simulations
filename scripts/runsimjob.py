from __future__ import print_function

import os, sys, shutil
import subprocess
import select
import numpy as np
from datetime import datetime, timedelta
from astropy.table import Table, Column

def run_log(cmd,slogfile,quiet=False):
    with open(slogfile, 'w') as logfile:
        logfile.write('Running process with command: '+cmd+'\n')
        proc=subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,universal_newlines=True)
        while True:
            try:
                select.select([proc.stdout],[],[proc.stdout])
            except select.error:
                pass
            line=proc.stdout.readline()
            if line=='':
                break
            if not quiet:
                sys.stdout.write(line)
            ts='{:%Y-%m-%d %H:%M:%S}'.format(datetime.now())
            logfile.write(ts+': '+line)
            logfile.flush()
        retval=proc.wait()
        logfile.write('Process terminated with return value %i\n' % retval)
    return retval


class bcolors(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def separator(s):
    print()
    width,_=get_terminal_size_linux()
    if len(s)%2 == 1:
        s+=' '
    if width is None:
        width=80
    lw=(width-len(s)-2)//2
    sep='='*lw
    print("%s%s %s %s%s"%(bcolors.FAIL,sep,s,sep,bcolors.ENDC))
    print()

    
def die(s):
    print(bcolors.FAIL+s+bcolors.ENDC)
    raise RuntimeError(s)

def report(s):
    print(bcolors.OKGREEN+s+bcolors.ENDC)

def warn(s):
    print(bcolors.OKBLUE+s+bcolors.ENDC)

def run(s,proceed=False,dryrun=False,log=None,quiet=False,database=True):
    report('Running: '+s)
    if not dryrun:
        if log is None:
            retval=subprocess.call(s,shell=True)
        else:
            retval=run_log(s,log,quiet)
        if not(proceed) and retval!=0:
           die('FAILED to run '+s+': return value is '+str(retval))
        return retval
    else:
        warn('Dry run, skipping this step')
        
def get_ska_sens_calc(ra,dec, f0, bw, inttime, AA='AA2', weight="natural"):
    """
     bw in mhz
    """
    import requests
    if AA == "AA2":
        array = "LOW_AA2_all"
    elif AA == "AAstarmod":
        array = "LOW_AAstar_all"
        
    freq_mid = f0 + bw/2.

    ## Perform continuum calculation
    url = "https://sensitivity-calculator.skao.int/api/v11/low/continuum/calculate"
    calc_response = requests.get(url, params={"integration_time_h": inttime,
                                              "pointing_centre":f"{ra} {dec}",
                                              "elevation_limit":20.,
                                              "freq_centre_mhz":freq_mid,
                                              "spectral_averaging_factor":1,
                                              "bandwidth_mhz":bw,
                                              "n_subbands":1,
                                              "subarray_configuration":array,
                                              "weighting_mode":weight,
                                              "robustness":None,
                                             }
                                 )

    ## Get JSON data from response
    contresp=calc_response.json()
    contcalc=calc_response.json()['transformed_result']

    sensitivity_cont_weight = contcalc['weighted_continuum_sensitivity']
          
    return sensitivity_cont_weight


def get_cosmos_sky_table(skytabfits, with_var=False, ddcal=False, fluxlim=0):
    if with_var:
        print('using variables in cosmos field')
        skytabfile = '/home/w.williams/projects/skasims/data/SRCnet_Combined_input_catalogue_alpha_variable.fits'
    elif ddcal:
        print('using ddcals in the cosmos field')
        skytabfile = '/home/w.williams/projects/skasims/data/SRCnet_Combined_input_catalogue_alpha_ddcal.fits'
    else:
        skytabfile = '/home/w.williams/projects/skasims/data/SRCnet_Combined_input_catalogue_alpha.fits'
    
        
    skytab = Table.read(skytabfile)
    Nsrc = len(skytab)
    print (f'{len(skytab)} sources in sky')
    Freqs = 120.e6*np.ones(Nsrc)
    Spidxs = -0.7*np.ones(Nsrc)
    #sky_data = np.array([ras, decs, Is, Qs, Us, Vs, Freqs, Spidxs, RMs, Majs, Mins, PAs]).T
    skytab.rename_column('Total_flux','I')
    skytab.rename_column('DC_Maj','Maj')
    skytab.rename_column('DC_Min','Min')
    skytab.rename_column('DC_PA','PA')
    skytab.rename_column('alpha','Spidx')
    skytab['Spidx'] = -1*skytab['Spidx']  # i think oskar wants -0.7 not 0.7 as in the catalogue
    skytab.keep_columns(['RA','DEC','I','Maj','Min','PA','Spidx'])
    skytab.add_column(Column(name='Q',data=np.zeros(Nsrc)))
    skytab.add_column(Column(name='U',data=np.zeros(Nsrc)))
    skytab.add_column(Column(name='V',data=np.zeros(Nsrc)))
    skytab.add_column(Column(name='Freq',data=Freqs))
    skytab.add_column(Column(name='RM',data=np.zeros(Nsrc)))
        
    sel = (skytab['I'] >= fluxlim)
    skytab = skytab[sel]
    print (f'{len(skytab)} sources in sky')
    
    skytab.write(skytabfits, overwrite=True)
    ra = 150.
    dec = 2.23
    return ra,dec

def get_sky_table_cal(skytabfits, RA0=0, DEC0=-30, fluxcal=100):
    skytab = Table({
        "RA":  [RA0 ],
        "DEC": [DEC0 ],
        "I":   [fluxcal],
        "Maj":   [0],
        "Min":   [0],
        "PA":   [0],
        "Q":   [0],
        "U":   [0],
        "V":   [0],
        "Freq":   [120.e6],
        "Spidx":   [-0.7],
        "RM":   [0],
           })
    skytab.write(skytabfits, overwrite=True)
    return
    
        
def get_test_sky_table(skytabfits, RA0=0, DEC0=-30, fluxlim=0, smallfield=False):
    # input skymodel is a simple model with 5 sources - generated with make_test_skymodel.py
    skytab = Table.read('/home/w.williams/projects/skasims/data/test_skymodel_simplesquare.fits')
    skytab['RA'] = skytab['RA']+RA0
    skytab['DEC'] = skytab['DEC']+DEC0
    print (f'{len(skytab)} sources in sky')
    Nsrc = len(skytab)
    Freqs = 120.e6*np.ones(Nsrc)
    Spidxs = -0.7*np.ones(Nsrc)
    #sky_data = np.array([ras, decs, Is, Qs, Us, Vs, Freqs, Spidxs, RMs, Majs, Mins, PAs]).T
    
    skytab.keep_columns(['RA','DEC','I','Maj','Min','PA'])
    skytab.add_column(Column(name='Q',data=np.zeros(Nsrc)))
    skytab.add_column(Column(name='U',data=np.zeros(Nsrc)))
    skytab.add_column(Column(name='V',data=np.zeros(Nsrc)))
    skytab.add_column(Column(name='Freq',data=Freqs))
    skytab.add_column(Column(name='Spidx',data=Spidxs))
    skytab.add_column(Column(name='RM',data=np.zeros(Nsrc)))
    
    skytab.write(skytabfits, overwrite=True)
    return
   
    
def get_sky_table(skytabfits, RA0=0, DEC0=-30, fluxlim=0, smallfield=False):
    # input skymodel comes from LoTSS deep combined 4 deep fields spread out and resampled  repositioned to 0,0 - shift this to RA0, DEC0 and apply a flux cut in Jy
    #skytab = Table.read('../data/LoTSS_DR3_random_20degsquare.fits')
    #skytab = Table.read('/home/w.williams/projects/skasims/data/LoTSS_deep_merge_resampled_100k.fits')
    if not smallfield:
        skytab = Table.read('/home/w.williams/projects/skasims/data/LoTSS_deep_merge_renorm.fits')
    else:
        skytab = Table.read('/home/w.williams/projects/skasims/data/LoTSS_deep_merge_renorm_smallfield.fits')
    #sel = (skytab['RA'] >-5) & (skytab['RA']<5) & (skytab['DEC']>-5) & (skytab['DEC']<5)
    sel = (skytab['RA'] >-3.4) & (skytab['RA']<3.4) & (skytab['DEC']>-3.4) & (skytab['DEC']<3.4)
    skytab = skytab[sel]
    skytab['RA'] = skytab['RA']+RA0
    skytab['DEC'] = skytab['DEC']+DEC0
    sel = (skytab['I'] >= fluxlim)
    skytab = skytab[sel]
    print (f'{len(skytab)} sources in sky')
    Nsrc = len(skytab)
    Freqs = 120.e6*np.ones(Nsrc)
    Spidxs = -0.7*np.ones(Nsrc)
    #sky_data = np.array([ras, decs, Is, Qs, Us, Vs, Freqs, Spidxs, RMs, Majs, Mins, PAs]).T
    
    skytab.keep_columns(['RA','DEC','I','Maj','Min','PA'])
    skytab.add_column(Column(name='Q',data=np.zeros(Nsrc)))
    skytab.add_column(Column(name='U',data=np.zeros(Nsrc)))
    skytab.add_column(Column(name='V',data=np.zeros(Nsrc)))
    skytab.add_column(Column(name='Freq',data=Freqs))
    skytab.add_column(Column(name='Spidx',data=Spidxs))
    skytab.add_column(Column(name='RM',data=np.zeros(Nsrc)))
    
    skytab.write(skytabfits, overwrite=True)
    return
   
def run_setup():
    # make the working directory, change there, and copy the simulation script and this script into the work directory for future reference
    if not restart:
        if os.path.isdir(WORKdir):
            print(f"working directory {WORKdir} already exists")
            while True:
                response = input("(y) to Remove and continue or (n) to quit? [y/n]: ").strip().lower()
                if response == "y":
                    shutil.rmtree(WORKdir)
                    print("Continuing...")
                    break
                elif response == "n":
                    print("Exiting.")
                    exit(1)
                else:
                    print("Invalid input, please enter y or n")
        os.mkdir(WORKdir)
        os.chdir(WORKdir)
        os.system(f"cp {SCRIPTname} .")
        os.system(f"cp {SIMcode} .")
    else:
        os.chdir(WORKdir)
        os.system(f"cp {SCRIPTname} .")
        os.system(f"cp {SIMcode} .")
    return

SCRIPTname = __file__
smallfield = False

POL_MODE = "Full"
nhrs = 1
dTIME = 6.8
RA0 = 0
DEC0 = -30
FREQ0 = 120.
dFREQ = 43.4
nCHAN = 500
AA = "AA2"

'''
nhrs = 4
dTIME = 6.8
RA0 = 0
DEC0 = -30
FREQ0 = 120.
dFREQ = 43.4
nCHAN = 1000
AA = "AAstarmod"
cat_Slim = 0

this was taking days and died at 99%
'''

'''
# settings for AA2 v2 dataset for SRCnet
nhrs = 4
dTIME = 3.4
FREQ0 = 120.
dFREQ = 21.7
nCHAN = 6912
nBANDS = 6
AA = "AA2"
#cat_Slim = 0.45e-3
cat_Slim = 0
identifier = "_cosmos"
usesky = 'COSMOS'
#usesky = 'cal'
#usesky = 'LoTSSDEEP'
chan_out = 40
dchan_out = 20
'''

'''
nhrs = 4
dTIME = 3.4
FREQ0 = 120.
dFREQ = 21.7
nCHAN = 6912
nBANDS = 2
AA = "AA2"
cat_Slim = 0.45e-3
identifier = ""
usesky = 'LoTSSDEEP'
chan_out = 2
dchan_out = 2
'''

'''
nhrs = 10./60
dTIME = 0.85
FREQ0 = 120.
dFREQ = 5.4
nCHAN = 27777
nBANDS = 6
AA = "AA2"
#identifier = "_raw_cal"
#usesky = 'cal'
identifier = "_raw_target"
usesky = 'LoTSSDEEP'
cat_Slim = 0.4e-3
chan_out = 12
dchan_out = 6
'''

'''
# settings for AAstarmod v1 dataset for SRCnet
nhrs = 30./60
dTIME = 3.4
FREQ0 = 50.
dFREQ = 21.7
nCHAN = 13824
nBANDS = 12
AA = "AAstarmod"
#cat_Slim = 0.45e-3
cat_Slim = 0
identifier = "_cosmos_full"
#usesky = 'LoTSSDEEP'
usesky = 'COSMOS'
chan_out = 30
dchan_out = 15
do_avg = False
avgtimestep = 1
avgfreqstep = 1
tecscreen = "None"
'''
'''
# settings for AA2 test ionoshpere
nhrs = 4
dTIME = 3.4
FREQ0 = 120.
dFREQ = 21.7
nCHAN = 6912
nBANDS = 6
AA = "AA2"
#cat_Slim = 0.45e-3
cat_Slim = 0
identifier = "_cosmos_ion"
#usesky = 'LoTSSDEEP'
usesky = 'COSMOS'
chan_out = 12
dchan_out = 6
do_avg = True
avgtimestep = 4
avgfreqstep = 4
tecscreen = "/home/w.williams/projects/skasims/data/scan_0.fits"
#tecscreen = "None"
'''

'''
# settings for AA2 test ionoshpere with variable in cosmos field
nhrs = 4
dTIME = 3.4
FREQ0 = 120.
dFREQ = 21.7
nCHAN = 6912
nBANDS = 6
AA = "AA2"
#cat_Slim = 0.45e-3
cat_Slim = 0
identifier = "_cosmos_ion_var"
#usesky = 'LoTSSDEEP'
usesky = 'COSMOS'
cosmos_var = True
chan_out = 12
dchan_out = 6
do_avg = True
avgtimestep = 4
avgfreqstep = 4
tecscreen = "/home/w.williams/projects/skasims/data/scan_0.fits"
'''
'''
# settings for minimal cosmos skymodel at lower resolution - for DD sol derivation
nhrs = 4
dTIME = 6.8
FREQ0 = 120.
dFREQ = 43.4
nCHAN = 3456
nBANDS = 6
AA = "AA2"
#cat_Slim = 0.45e-3
cat_Slim = 0
identifier = "_cosmos_ion_ddcals1"
#usesky = 'LoTSSDEEP'
usesky = 'COSMOS'
cosmos_var = False
cosmos_dd = True
chan_out = 12
dchan_out = 6
do_avg = True
avgtimestep = 2
avgfreqstep = 2
tecscreen = "/home/w.williams/projects/skasims/data/scan_0.fits"
'''

# settings for AA2 test with/without ionoshpere with bright lotss only
# 2hr obs with around 10MHz bandwidth
nhrs = 1
dTIME = 6.8
FREQ0 = 120.
dFREQ = 43.4
nCHAN = 250
nBANDS = 1
AA = "AA2"
identifier = "_testsky2_tec_line2"
POL_MODE = "Full"
usesky = 'testsky'
chan_out = 2
dchan_out = 2
do_avg = False
avgtimestep = 1
avgfreqstep = 1
#tecscreen = "/home/w.williams/projects/skasims/data/scan_0.fits"
tecscreen = "/home/w.williams/projects/skasims/data/blank_line2_iono.fits"
#tecscreen = "None"

'''
# settings for short repeat obs for sideral averaging test
nhrs = 10./60
dTIME = 3.4
FREQ0 = 120.
dFREQ = 21.7
nCHAN = 1000
nBANDS = 1
AA = "AA2"
cat_Slim = 1e-3
identifier = "_20270101"
usesky = 'LoTSSDEEP'
chan_out = 2
dchan_out = 2
do_avg = True
avgtimestep = 2
avgfreqstep = 2
tecscreen = "None"
# gives 6288 sources
'''
'''
# settings for AA2 test ionoshpere - using only bright sources for cosmos calibration
nhrs = 4
dTIME = 3.4
FREQ0 = 120.
dFREQ = 21.7
nCHAN = 6912
nBANDS = 6
AA = "AA2"
#cat_Slim = 0.45e-3
cat_Slim = 0
identifier = "_cosmos_ion"
#usesky = 'LoTSSDEEP'
usesky = 'COSMOS'
chan_out = 12
dchan_out = 6
do_avg = True
avgtimestep = 4
avgfreqstep = 4
tecscreen = "/home/w.williams/projects/skasims/data/scan_0.fits"
#tecscreen = "None"
'''
LENGTH_s = nhrs*60*60

restart = 0
root = f'{AA}_{LENGTH_s/60:.1f}min_{nCHAN}channels_{dTIME}s_{dFREQ}kHz_{FREQ0:.1f}MHz'

WORKdir = f"{root}{identifier}"


dirtyonly = False
applybeam = True

if nBANDS == 1:
    MS = [f"{root}.ms"]
    MSOUT = [f"{root}_applybeam.ms"]
else:
    MS = [f"{root}_SB{index:03d}.ms" for index in range(nBANDS)]
    MSOUT = [f"{root}_SB{index:03d}_applybeam.ms" for index in range(nBANDS)]
if not applybeam:
    MSOUT = MS
WEIGHT = "briggs -1"
NPIX = "16384"   # 6.8 deg for 1.5asec pixels
NPIX = "8192"
NPIX = "4096"  # for test skymodel
#NPIX = "512"   # for calibrator
IM1=f"ws_{root}_robm1"
IM2=f"{IM1}_mask"
MASK=f"{IM2}.fits"
CELL="1.5arcsec"

SKYTAB = f"{root}_skytab.fits"


SIMcode = "/home/w.williams/projects/skasims/python/oskar_sim.py"

run_setup()


if usesky == 'COSMOS':
    RA0, DEC0 = get_cosmos_sky_table(SKYTAB, with_var=cosmos_var, ddcal=cosmos_dd)
elif usesky == 'LoTSSDEEP':
    RA0 = 0
    DEC0 = -30
    get_sky_table(SKYTAB,RA0,DEC0,fluxlim=cat_Slim,smallfield=smallfield)
elif usesky == 'testsky':
    RA0 = 30.5
    DEC0 = -30
    get_test_sky_table(SKYTAB,RA0,DEC0)
elif usesky == 'cal':
    RA0 = 15
    DEC0 = -30
    get_sky_table_cal(SKYTAB,RA0,DEC0,fluxcal=100.)


imsense = get_ska_sens_calc(RA0,DEC0, FREQ0, dFREQ*nCHAN/1e3, nhrs, AA=AA, weight="natural")
report(f"Sensitivity calculator reports {imsense} for these settings and natural weighting")
report(f"Sensitivity calculator reports {imsense['value']*1e3:.2f} mJy/bm for these settings and natural weighting")

noise_thresh = f"{imsense['value']*1e3:.3f}mJy"


oskar_cmd = f"python {SIMcode} --array {AA} --nhours {nhrs} --timestep {dTIME} --freq {FREQ0} --freqstep {dFREQ} --nchan {nCHAN} --ra {RA0} --dec {DEC0} --skytabfits {SKYTAB} --nsubbands {nBANDS} --tecscreenfits {tecscreen} --pol_mode {POL_MODE}"

#ws_cmd = f"wsclean -size 12000 12000 -scale 3arcsec -name {IM}  -weight briggs 1 -pol I -join-channels -channels-out 15 -niter 1000000 -local-rms -auto-mask 3 -auto-threshold 0.3 -mgain 0.8 -mgain-boosting 1.3  -fit-spectral-pol 2  -apply-primary-beam  {MSOUT}"
# -multiscale -multiscale-scale-bias 0.8 -multiscale-max-scales 3
ws_cmd1 = f"wsclean -log-time -temp-dir /data-hot/not-backup/w.williams/wsclean_temp -size {NPIX} {NPIX} -scale {CELL} -name {IM1}  -weight {WEIGHT} -pol I -niter 0 -no-reorder  {' '.join(MSOUT)}"
make_mask_cmd = f"python /home/w.williams/projects/skasims/python/make_mask_from_sources.py {IM1}-image.fits {SKYTAB} {MASK}"
ws_cmd2 = f"wsclean -log-time -temp-dir /data-hot/not-backup/w.williams/wsclean_temp -size {NPIX} {NPIX} -scale {CELL} -name {IM2}  -weight {WEIGHT} -pol I -join-channels -channels-out {chan_out} -deconvolution-channels {dchan_out} -fit-spectral-pol 2  -niter 1000000 -dd-psf-grid 5 5  -fits-mask {MASK} -abs-threshold {noise_thresh} -mgain 0.8 -mgain-boosting 1.3  -parallel-deconvolution 2048   -apply-primary-beam  {' '.join(MSOUT)}"
msoverview_cmd = f"python /home/w.williams/scripts/msoverview.py {MSOUT}"
find_srcs_cmd = f"echo python /home/w.williams/projects/skasims/python/find_sources.py --blankpb {IM2}-MFS-image-pb.fits {IM2}-MFS-image-pb-pybdsf.fits {IM2}-MFS-image.fits"


'''
do beam in facets
-facet-regions ds9.reg \
-apply-facet-beam \
-facet-beam-update 120 \
'''

def log_time_to_file(s, filename, lastime=None):
    timenow = datetime.now()
    if lastime is not None:
        writetime = timenow - lastime
    else:
        writetime = timenow
    with  open(filename,"a") as logfile:
        logfile.write(f"{s}: {writetime} \n")
    return timenow

logfile = "simulation_summary.log"
with open(logfile,"w") as f:
    f.write("Starting job\n")
    f.write(f"Sensitivity calculator reports {imsense} for these settings and natural weighting\n")
    

starttime = log_time_to_file("oskar start", logfile)
run_log(oskar_cmd, "oskarsim.log")
_ = log_time_to_file(f"*oskar run took", logfile, lastime=starttime)


if applybeam:
    starttime = log_time_to_file("dppp start", logfile)
    for msi,msouti in zip(MS, MSOUT):
        if do_avg:
            dppp_cmd = f"DP3 msin={msi} steps=[avg,applybeam] msout={msouti} avg.type=averager avg.timestep={avgtimestep} avg.freqstep={avgfreqstep}"
        else:
            dppp_cmd = f"DP3 msin={msi} steps=[applybeam] msout={msouti}"
        run_log(dppp_cmd, f"dppp_applybeam_{msi}.log")
    _ = log_time_to_file(f"*dppp run(s) took", logfile, lastime=starttime)

run_log(msoverview_cmd, "msoverview.log")
    
starttime = log_time_to_file("ws1 start", logfile)
run_log(ws_cmd1, "wsclean1.log")
_ = log_time_to_file(f"*ws1 run took", logfile, lastime=starttime)

if not dirtyonly:
    run_log(make_mask_cmd, "paint_mask.log")

    starttime = log_time_to_file("ws2 start", logfile)
    run_log(ws_cmd2, "wsclean2.log")
    _ = log_time_to_file(f"*ws2 run took", logfile, lastime=starttime)

    run_log(find_srcs_cmd, "pybdsf.log")



