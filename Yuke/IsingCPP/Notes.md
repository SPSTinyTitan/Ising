#  Notes

### Implementation
This is a monte carlo simulation of the ising model using the Metropolis algorithm. 

### Performance
Calculations are done in Armadillo (linked with BLAS). Convolutions are far faster than shift+addition due to caching.

