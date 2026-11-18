# Superseded (microstrip only)

The microstrip records here are superseded; bragg and lcavity stand. The gfem worker at this commit passed the case film thickness to gfem's `Conductor` on the voided microstrip, which doubles it on mesh-exterior faces, so gfem modelled twice the metal film (S off by about 1.6e-2 from the corrected model). Fixed in the worker at 246b06d (benchmark-suite). 

Rerun at 7a078ba: `20261118-7a078ba-release-gfem`. Records here are left as written.
