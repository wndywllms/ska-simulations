from __future__ import print_function

import os, sys
import argparse
import numpy as np
import oskar
from datetime import timedelta
#import aplpy

rng = np.random.default_rng(seed=42)

from astropy.coordinates import EarthLocation, Angle
from astropy.time import Time
from astropy import units as u
from astropy.table import Table, Column



def transit_time(
    ra: u.Quantity,
    location: EarthLocation,
    date: Time,
    offset: float = 0,
) -> Time:
    """
    More precise version using linear interpolation around the minimum.
    Accurate to around a second.
    """
    ra_hours = ra.to(u.hourangle).value

    # Coarse search first (2-minute resolution)
    times_coarse = date + np.linspace(0, 24, 720) * u.hour
    lst_coarse   = np.array([
        t.sidereal_time("apparent", longitude=location.lon).hour
        for t in times_coarse
    ])

    diff = np.abs(lst_coarse - ra_hours)
    diff = np.minimum(diff, 24 - diff)
    idx  = np.argmin(diff)

    # Refine over a 4-minute window around the coarse minimum
    t_centre    = times_coarse[idx]
    times_fine  = t_centre + np.linspace(-2, 2, 1000) * u.minute
    lst_fine    = np.array([
        t.sidereal_time("apparent", longitude=location.lon).hour
        for t in times_fine
    ])

    diff_fine = np.abs(lst_fine - ra_hours)
    diff_fine = np.minimum(diff_fine, 24 - diff_fine)
    idx_fine  = np.argmin(diff_fine)

    return times_fine[idx_fine] - offset*u.hour



def run_oskar(
    skytabfits: str = "None",
    tecscreenfits: str = "None",
    AA: str = "AA2",
    nhrs: float = 1,
    RA0: float = 0.,
    DEC0: float = -30.,
    dTIME: float = 3.4, # in sec - length of integration time

    FREQ0: float = 120e6,   # in hertz
    nCHAN: int = 1000,
    dFREQ: float = 21.7e3,   # in hertz
    nogpu: bool = False,
    hassubbands: bool = False,
    index: int = 0,
    namefreq: float = 120e6,
    namechans : int = 1000,
    POL_MODE : str = "Full",
) -> None:
    """
    Run oskar

    Parameters
    ----------
    param   : ..
    """
    '''
    target is 4hrs, 3.4s, 21.7kHz, 150MHz
    '''
    
    if AA == "AA2":
        TELDIR = "/home/w.williams/projects/skasims/teldir/SKA-Low_AA2_68S_rigid-rotation_model_noerr.tm"
    elif AA == "AAstarmod":
        TELDIR = "/home/w.williams/projects/skasims/teldir/telescope_model_AAstar_mod1"
    else:
        print(f"No oskar telescope model for array {AA}")
        sys.exit(1)
        
        
    print(f"Array is {AA}")
    print(f"Using telescope model {TELDIR}")


    LENGTH_s = nhrs*60*60
    
    if not hassubbands:
        outroot = f'{AA}_{LENGTH_s/60}min_{nCHAN}channels_{dTIME}s_{dFREQ/1e3}kHz_{FREQ0/1e6}MHz'
    else:
        outroot = f'{AA}_{LENGTH_s/60}min_{namechans}channels_{dTIME}s_{dFREQ/1e3}kHz_{namefreq/1e6}MHz_SB{index:03d}'

    # note that these noise files should match the dTIME and dFREQ values
    NOISE_FREQ_FILE = f"/home/w.williams/projects/skasims/data/oskar_noise_freqs_{dTIME}s_{dFREQ/1e3}kHz.txt"
    NOISE_RMS_FILE = f"/home/w.williams/projects/skasims/data/oskar_noise_rms_{dTIME}s_{dFREQ/1e3}kHz.txt"
    
    if not os.path.isfile(NOISE_FREQ_FILE):
        print(f"Oskar noise frequency file is missing: {NOISE_FREQ_FILE}")
        sys.exit(1)
    if not os.path.isfile(NOISE_RMS_FILE):
        print(f"Oskar noise rms file is missing: {NOISE_RMS_FILE}")
        sys.exit(1)

    
    has_ionosphere = True
    if tecscreenfits == "None":
        has_ionosphere = False
    else:
        tec_screen = tecscreenfits
        
        if not os.path.isfile(tec_screen):
            print(f"TEC screen {tec_screen} does not exist!")
            sys.exit(1)

    BW = dFREQ*nCHAN
    nTIME = int(np.floor(LENGTH_s / dTIME))  # number of timesteps

    # for internal - covert for oscar
    LENGTH_s_actual = nTIME*dTIME
    LENGTH_f = str(timedelta(seconds=LENGTH_s_actual))  #f"{nhrs:02d}:00:00.000" - format for oscar input
            
    skao_low_core = EarthLocation.from_geodetic(
        116.7644482, -26.82472208, 365.0, ellipsoid="WGS84"
    )


    # get the start time from the tranit time and length of observation
    date = "2027-01-01"
    TIME0 = transit_time(RA0*u.deg, skao_low_core, Time(date), nhrs/2.)
    
    #observing_time = Time(TIME0, scale='utc', location=skao_low_core)
    #LST = observing_time.sidereal_time('mean')


    #RA0 = (LST+(nhrs/2)*u.hourangle).deg


    # this is for simulating averaging smearing within oskar ...  TBC
    chan_avg = 1
    time_avg = 1
    dFREQavg = dFREQ / chan_avg   # Hz
    #nCHANavg = nCHAN /chan_avg
    dTIMEavg =  dTIME / time_avg
    #nTIMEavg = nhrs*60*60. / dTIMEavg



    LENGTH_f = str(timedelta(seconds=LENGTH_s_actual))  #f"{nhrs:02d}:00:00.000" - format for oscar input

    double_precision = True
    use_gpu = True
    if nogpu:
        use_gpu = False
    params = {
        "simulator/use_gpus": use_gpu,
        "simulator/max_sources_per_chunk": 200000,
        "simulator/keep_log_file": True,
        "simulator/write_status_to_log_file": True,
        "simulator/double_precision": double_precision,
        "observation/num_channels": nCHAN,
        "observation/start_frequency_hz": FREQ0 ,#100e6,
        "observation/frequency_inc_hz": dFREQ ,#10e6,
        "observation/phase_centre_ra_deg": RA0,
        "observation/phase_centre_dec_deg": DEC0,
        "observation/num_time_steps": nTIME,
        "observation/start_time_utc": TIME0,
        "observation/length": LENGTH_f,
        "telescope/pol_mode": POL_MODE,
        "telescope/input_directory": TELDIR,
        "telescope/normalise_beams_at_phase_centre": False,
        "telescope/aperture_array/array_pattern/normalise": True,
        "interferometer/oskar_vis_filename": f"{outroot}.vis",
        "interferometer/ms_filename": f"{outroot}.ms",
        "interferometer/ms_dish_diameter": 39,
        "interferometer/channel_bandwidth_hz": dFREQavg,
        "interferometer/time_average_sec": dTIMEavg,
        "interferometer/max_time_samples_per_block": 8,
        "interferometer/max_channels_per_block": nCHAN,
        "interferometer/noise/enable": True,
        "interferometer/noise/seed": 42,
        "interferometer/noise/freq": "Data file",
        "interferometer/noise/freq/file": NOISE_FREQ_FILE,
        "interferometer/noise/rms": "Data file",
        "interferometer/noise/rms/file": NOISE_RMS_FILE,
    }
    if has_ionosphere:
        params.update({
        "telescope/ionosphere_screen_type": "External",
        "telescope/external_tec_screen/input_fits_file": tec_screen ,
        #"telescope/external_tec_screen/screen_height_km": 300 ,  # default val = in km
                      })
    settings = oskar.SettingsTree("oskar_sim_interferometer")
    settings.from_dict(params)

    with open(f"{outroot}_oskar_settings.txt", "w") as f:
        for key, value in params.items():
            f.write(f"{key}={value}\n")

    sim = oskar.Interferometer(settings=settings)
    
    if skytabfits != "None":

        skytab = Table.read(skytabfits)
        sky_data = np.array([skytab['RA'] , skytab['DEC'] , skytab['I'] , skytab['Q'], skytab['U'], skytab['V'], skytab['Freq'], skytab['Spidx'], skytab['RM'], skytab['Maj'], skytab['Min'], skytab['PA']]).T
        sky = oskar.Sky.from_array(sky_data)  # Pass precision here.


        sim.set_sky_model(sky)
        
        
        
    print('simulating observation')
    sim.run()
    
    return


def main():
    # 1. Initialize the parser
    parser = argparse.ArgumentParser(
        description="A simple script to run oskar simulations"
    )
    parser.add_argument("--nogpu", action="store_true", help="Flag to turn off using gpus in oskar simulation")
    parser.add_argument("--array", choices=["AA2", "AAstarmod"], default="AA2", help="SKA Array release to simulate")
    parser.add_argument("--ra", type=float, default=0., help="Phase centre RA in deg")
    parser.add_argument("--dec", type=float, default=-30, help="Phase centre DEC in deg")
    parser.add_argument("--nhours", type=float, default=1., help="Number of hours")
    parser.add_argument("--timestep", type=float, default=8., help="Time step in s")
    parser.add_argument("--freq", type=float, default=120., help="Start frequency in MHz")
    parser.add_argument("--freqstep", type=float, default=48., help="Frequency step in kHz")
    parser.add_argument("--nchan", type=int, default=10, help="Number of channels")
    parser.add_argument("--skytabfits", type=str, default="None", help="Sky catalogue in fits format")
    parser.add_argument("--tecscreenfits", type=str, default="None", help="TEC screen in fits format")
    parser.add_argument("--pol_mode", type=str, default="Full", help="The polarisation mode of simulations which use the telescope model.")
    parser.add_argument("--nsubbands", type=int, default=1, help="Number of subbands to split the simulation into")
    
    args = parser.parse_args()
    
    freqs = np.arange(args.freq*1e6, args.freq*1e6+args.nchan*args.freqstep*1e3, args.freqstep*1e3)
    if len(freqs) != args.nchan:
        print(f"length of freqs {len(freqs)} doesn't match the number of channels {args.nchan}")
        print(freqs)
        sys.exit(1)
    freqs_chunked = np.array_split(freqs, args.nsubbands)

    if args.nsubbands > 1:
        hassubbands = True
    else:
        hassubbands = False

    for run_i, freqs_i in enumerate(freqs_chunked):
        print(f"starting oskar simulation for subband {run_i} with {len(freqs_i)} channels")
        run_oskar(
            AA = args.array,
            RA0 = args.ra,
            DEC0 = args.dec,
            nhrs = args.nhours,
            dTIME = args.timestep, # in sec - length of integration time
            FREQ0 = freqs_i[0],   # in hertz
            nCHAN = len(freqs_i),
            dFREQ = args.freqstep*1e3,   # in hertz
            skytabfits = args.skytabfits,
            nogpu = args.nogpu,
            hassubbands = hassubbands,
            index = run_i,
            namefreq = args.freq*1e6,
            namechans = args.nchan,
            tecscreenfits = args.tecscreenfits,
            POL_MODE = args.pol_mode,
        )

if __name__ == "__main__":
    main()

