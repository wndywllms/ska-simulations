import numpy as np
import requests
import astropy.units as u

# 1s , 1kHz
df = 0.1  # in MHz
dt = 1./3600  # in hr
ns = 68
off = 0.5*df
band_freqs = np.arange(100+off, 150+off, 10)
AA2_sens = np.zeros_like(band_freqs)

for i,freq in enumerate(band_freqs):
    ## Perform continuum calculation
    url = "https://sensitivity-calculator.skao.int/api/v11/low/continuum/calculate"
    calc_response = requests.get(url, params={"integration_time_h": dt,
                                              "subarray_configuration": "LOW_AA2_all",
                                              "pointing_centre": "0 -27",
                                              "freq_centre_mhz": freq,
                                              "bandwidth_mhz": df,
                                              "spectral_averaging_factor":1,
                                              "weighting_mode":"natural",
                                              "elevation_limit": 88
                                              }
                                )
    
    ## Get JSON data from response
    contresp=calc_response.json()
    contcalc=calc_response.json()['transformed_result']
    sensitivity_cont_weight = contcalc['weighted_continuum_sensitivity']['value']

    # print(contresp)      ## uncomment to print full response
    
    ### Print weighted continuum sensitivty
    print(f"Frequency {freq} sensitivity {sensitivity_cont_weight*1e3} mJy/bm")
    
    AA2_sens[i] = sensitivity_cont_weight

low_station_sefd = AA2_sens*u.Jy*np.sqrt((ns*(ns-1)*df*u.MHz*dt*u.s).decompose())

import astropy.units as u
vis_dt = 3.4*u.s
vis_df = 21.7*u.kHz

vis_noise = np.sqrt((low_station_sefd**2)/(2*(vis_df*vis_dt).decompose()))
