#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void fast_histogram(const int const *labels, int labels_size, int *hist) {
    int i;
    for (i = 0; i < labels_size; ++i)
	++hist[labels[i]];
}

double fast_entropy(const double const *hist, int hist_size) {
    int i;
    double out = 0.;
    for (i = 0; i < hist_size; ++i)
	if (hist[i] > 0)
	    out -= log2(hist[i]) * hist[i];
    return out;
}
