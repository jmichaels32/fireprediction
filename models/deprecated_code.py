# Deprecated code that still may be useful

'''
total_pixels = 0
total_nodata = 0
total_nofire = 0
total_fire = 0
for train in train_data:
    seg_mask_prev = train[:, -2, :, :]
    seg_mask_gold = train[:, -1, :, :]

    total_pixels += np.prod(seg_mask_prev.shape) + np.prod(seg_mask_gold.shape)
    total_nodata += np.sum(seg_mask_prev.numpy() == -1) + np.sum(seg_mask_gold.numpy() == -1)
    total_nofire += np.sum(seg_mask_prev.numpy() == 0) + np.sum(seg_mask_gold.numpy() == 0)
    total_fire += np.sum(seg_mask_prev.numpy() == 1) + np.sum(seg_mask_gold.numpy() == 1)
    
print(f"Total Pixels: {total_pixels}")
print(f"Total No Data: {total_nodata}")
print(f"Total No Data: {100 * total_nodata/total_pixels}")
print(f"Total No Fire: {total_nofire}")
print(f"Total No Fire: {100 * total_nofire/total_pixels}")
print(f"Total Fire: {total_fire}")
print(f"Total Fire: {100 * total_fire/total_pixels}")'''
    