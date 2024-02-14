import numpy as np

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    bbox = frame[ymin:ymax, xmin:xmax, :]
    bbox_colors = bbox.reshape(((ymax - ymin) * (xmax - xmin), 3))

    # hist, _ = np.histogramdd(bbox_colors, bins=hist_bin, range=[(0, 255), (0, 255), (0, 255)])
    hist, _ = np.histogramdd(bbox_colors, bins=hist_bin)

    hist = hist / np.sum(hist)

    return hist