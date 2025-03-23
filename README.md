# Résultats
| Version | Device | Temps FFT | Temps IFFT|
|:-------|:---:|:----------:|---------:|
|  Numpy  |  CPU   | 3.172216 s |3.212742 s|
| Pytorch |  GPU   | 0.220442 s |0.093029 s|
| Notre version récursive| CPU |  131.204758 s| 0.426998 s|
| Notre version itérative| CPU |  0.746637 s| 0.426478 s|
| Notre version | GPU |  0.661062 s| 0.453959 s|
| Notre version C++| CPU |  0.28 s| 0.28 s|
| Notre version Cuda (portage C++ )| GPU |  0,053 s| 0,002516 s |



