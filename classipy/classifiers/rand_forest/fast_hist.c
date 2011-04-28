#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void fast_histogram(const int const *labels, int labels_size, int *hist) {
    int i;
    for (i = 0; i < labels_size; ++i)
	++hist[labels[i]];
}

void fast_histogram_weight(const int const *labels, const int const *weights, int labels_size, int weight_rows, int num_classes, int *hist) {
    /*
      Args:
          labels: shape(labels_size)
          weights: shape(weight_rows, labels_size)
          labels_size:
          weight_rows:
          num_classes:
          hist: shape(weight_rows, num_classes)
     */
    int i, j, k = 0;
    for (j = 0; j < weight_rows; ++j, hist += num_classes)
        for (i = 0; i < labels_size; ++i)
            hist[labels[i]] += weights[k++];
}

double fast_entropy(const double const *hist, int hist_size) {
    int i;
    double out = 0.;
    for (i = 0; i < hist_size; ++i)
	if (hist[i] > 0)
	    out -= log2(hist[i]) * hist[i];
    return out;
}
