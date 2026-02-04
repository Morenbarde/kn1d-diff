This file saves various statement used to caputur inputs/outputs/parameters from various kn1d functions


### Kinetic_H_Mesh inputs
```

```


### Kinetic_H inputs
```
file = 'kinetic_h_in.npz'
print("Saving to file: " + file)
np.savez("kn1ddiff/test/init_kinetic_h/"+file, mu=mu, vxiA=vxiA, fHBC=fHBC, GammaxHBC=GammaxHBC)
input()
```

### Kinetic_H parameters excluding common blocks
```
file = 'kinetic_h_params.npz'
print("Saving to file: " + file)
np.savez("kn1ddiff/test/init_kinetic_h/"+file, mu=kinetic_h.mu, vxi=kinetic_h.vxi, fHBC=kinetic_h.fHBC, 
            GammaxHBC=kinetic_h.GammaxHBC, nvr=kinetic_h.nvr, nvx=kinetic_h.nvx, nx=kinetic_h.nx, vx_neg=kinetic_h.vx_neg,
            vx_pos=kinetic_h.vx_pos, vx_zero=kinetic_h.vx_zero, vth=kinetic_h.vth, vr2_2vx2_2D=kinetic_h.vr2_2vx2_2D,
            dvr_vol=kinetic_h.dvr_vol, dvx=kinetic_h.dvx, fHBC_input=kinetic_h.fHBC_input)
input()
```


### Kinetic_H Internal Block
```
file = 'kinetic_h_internal.json'
data = kinetic_h.Internal
print("Saving to file: " + file)
sav_data = {'vr2vx2' : data.vr2vx2,
            'vr2vx_vxi2' : data.vr2vx_vxi2,
            'fi_hat' : data.fi_hat,
            'ErelH_P' : data.ErelH_P,
            'Ti_mu' : data.Ti_mu,
            'ni' : data.ni,
            'sigv' : data.sigv,
            'alpha_ion' : data.alpha_ion,
            'v_v2' : data.v_v2,
            'v_v' : data.v_v,
            'vr2_vx2' : data.vr2_vx2,
            'vx_vx' : data.vx_vx,

            'Vr2pidVrdVx' : data.Vr2pidVrdVx,
            'SIG_CX' : data.SIG_CX,
            'SIG_H_H' : data.SIG_H_H,
            'SIG_H_H2' : data.SIG_H_H2,
            'SIG_H_P' : data.SIG_H_P,
            'Alpha_CX' : data.Alpha_CX,
            'Alpha_H_H2' : data.Alpha_H_H2,
            'Alpha_H_P' : data.Alpha_H_P,
            'MH_H_sum' : data.MH_H_sum,
            'Delta_nHs' : data.Delta_nHs,
            'Sn' : data.Sn,
            'Rec' : data.Rec}

sav_data = make_json_compatible(sav_data)
sav_to_json("kn1ddiff/test/init_kinetic_h/"+file, sav_data)
input()
```


### MH_value input/output
```
file = 'mh_in_out1.json'
print("Saving to file: " + file)
sav_data = {'fH' : fHG,
            'nH' : NHG[:,igen-1],
            'TH2_Moment' : self.H2_Moments.TH2,
            'VxH2_Moment' : self.H2_Moments.VxH2,

            'MH_H' : m_vals.H_H,
            'MH_P' : m_vals.H_P,
            'MH_H2' : m_vals.H_H2}

sav_data = make_json_compatible(sav_data)
sav_to_json("kn1ddiff/test/mh_values/"+file, sav_data)
input()
```