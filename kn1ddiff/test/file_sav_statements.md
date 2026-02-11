This file saves various statement used to caputur inputs/outputs/parameters from various kn1d functions


### Kinetic_H_Mesh inputs
```
file = 'h_mesh_in.npz'
print("Saving to file: " + file)
np.savez("kn1ddiff/test/mh_values/"+file, mu=mu, x=x, Ti=Ti, Te=Te, n=n, PipeDia=PipeDia, E0=E0, fctr=fctr)
input()
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


### Beta_CX input/output
```
file = 'beta_cx_in_out.json'
print("Saving to file: " + file)
sav_data = {'fH' : fH,
            
            'fi_hat' : self.Internal.fi_hat,
            'Alpha_CX' : self.Internal.Alpha_CX,
            'ni' : self.Internal.ni,
            'SIG_CX' : self.Internal.SIG_CX,

            'Beta_CX' : Beta_CX}

sav_data = make_json_compatible(sav_data)
sav_to_json("kn1ddiff/test/beta_cx/"+file, sav_data)
input()
```



### KH_Generations input/output
```
file = 'kh_gens_in.json'
print("Saving to file: " + file)
sav_data = {'fH' : fH,
            'A' : meq_coeffs.A,
            'B' : meq_coeffs.B,
            'C' : meq_coeffs.C,
            'D' : meq_coeffs.D,
            'F' : meq_coeffs.F,
            'G' : meq_coeffs.G,
            'CF_H_H' : collision_freqs.H_H,
            'CF_H_P' : collision_freqs.H_P,
            'CF_H_H2' : collision_freqs.H_H2,
            
            'TH2_Moment' : self.H2_Moments.TH2,
            'VxH2_Moment' : self.H2_Moments.VxH2,
            'fi_hat' : self.Internal.fi_hat,
            'Alpha_CX' : self.Internal.Alpha_CX,
            'ni' : self.Internal.ni,
            'SIG_CX' : self.Internal.SIG_CX
            }
sav_data = make_json_compatible(sav_data)
sav_to_json("kn1ddiff/test/h_gens/"+file, sav_data)
input()

file = 'kh_gens_out.json'
print("Saving to file: " + file)
sav_data = {'fH' : fH_total,
            'Beta_CX_sum' : Beta_CX_sum,
            'Msum_H_H' : m_sums.H_H,
            'Msum_H_P' : m_sums.H_P,
            'Msum_H_H2' : m_sums.H_H2
            }

sav_data = make_json_compatible(sav_data)
sav_to_json("kn1ddiff/test/h_gens/"+file, sav_data)
input()
```