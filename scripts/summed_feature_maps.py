from ncc import normxcorr

correlation_maps = np.zeros_like(shoeprint_feature_maps)

for i in range(shoeprint_feature_maps.shape[0]):
    correlation_maps[i] = normxcorr(shoemark_feature_maps[i], shoeprint_feature_maps[i])

