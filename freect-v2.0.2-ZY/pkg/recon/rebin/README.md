# Rebin Package

This package was originally going to be entirely contained within the WFBP code, however given the utility of rebinning in other applications, I've made it its own package.

Furthermore, given that modern scanners are starting to have increasingly complex geometry (e.g., the Force's non-fully-cylindrical detector) and the fact that this is handled in the rebinning step, it's likely that this will further expand.  Grouping it all into a single package is likely to keep the rebinning code somewhat more clear than mixing it in with the WFBP code.

## Currently on the road map

### CPU
- [ ] No FFS
- [ ] Phi FFS
- [ ] Z FFS
- [ ] Phi+Z FFS
- [ ] Diag FFS 

### GPU
- [ ] No FFS
- [ ] Phi FFS
- [ ] Z FFS
- [ ] Phi+Z FFS
- [ ] Diag FFS 

