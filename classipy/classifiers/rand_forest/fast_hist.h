void fast_histogram(const int const *labels, int labels_size, int  *hist);
double fast_entropy(const double const *hist, int hist_size);
void depth_predict(uint16_t *depth, double *out_prob, uint8_t *out_ind, int32_t *trees, int32_t *links, double *leaves,
                   double *u, double *v, int32_t *t, int num_trees, int num_nodes, int num_leaves, int num_classes);
