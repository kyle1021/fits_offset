Please contact Kai-yang Lin (kylin@asiaa.sinica.edu.tw) if you have any questions.

2019/Dec/12


To test the code directly:
- create a test folder
- link all fits files to the test folder
- run the correlation command as following:
../bin/correlate_fits.py ST-i_12239 --full --dmax 50.

Notes:
1. ST-i_12239 --> is the serial ID of the test data CCD. 
2. --full --> enables the full pair-wise cross-correlation
3. --dmax 50. --> limit the max offset to 50 arcmin (this test set is a mosaic field, so offset can be pretty large)
4. the current default is to not de-rotate the image, so this measures offsets in the physical coordinates in arcmin

Outputs:

Xcorr_offsets.txt --> the pairwise offsets and correlation strengths for diagnostics

Xcorr_solution.txt --> the relative offset of each image (relative to the ref image shown in the first line)

Xcorr_residual_ra.txt --> comparing the pair-wise offset in X(ra) direction to the solution/model (another one for Y(dec))


