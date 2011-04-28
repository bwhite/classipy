#ifndef FAST_HIST_H
#define FAST_HIST_H
void fast_histogram(const int const *labels, int labels_size, int  *hist);
double fast_entropy(const double const *hist, int hist_size);
void fast_histogram_weight(const int const *labels, const int const *weights, int labels_size, int weight_rows, int num_classes, int *hist);
#endif
