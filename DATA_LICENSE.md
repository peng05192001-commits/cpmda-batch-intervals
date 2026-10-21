# Dataset attribution and permitted reuse

Chen, S. (2017). **Beijing Multi-Site Air Quality**. UCI Machine Learning Repository. https://doi.org/10.24432/C5RK5G

Official dataset and license: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data . UCI identifies the dataset as Creative Commons Attribution 4.0 International (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/ . The code's MIT license does not replace this data license.

The two .npz files are derived subsets for Aotizhongxin and Dongsi. They select the ten paper features and fixed calibration/query periods, exclude missing calibration responses, and retain feature NaNs. Calibration responses are included; query response labels are not. The source-member names, dates, counts and checksums are in support.json. Feature names in order: PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM. Time arrays are retained. Model fitting is not repeated by the verification script.

Air-quality observations originate from the Beijing Municipal Environmental Monitoring Center; meteorological observations are matched to the nearest China Meteorological Administration station, as described by UCI. Please retain this attribution and indicate any further changes when redistributing these subsets.
