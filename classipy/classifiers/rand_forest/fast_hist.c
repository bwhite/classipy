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

#define WIDTH 640
#define HEIGHT 480
#define BIGNUM 2047
#define BOUNDS(i, j) (0 <= (i) && (i) < HEIGHT && 0 <= (j) && (j) < WIDTH)
#define CLAMP(x) ((x) > BIGNUM ? (x) : BIGNUM)

inline int depth_samp(uint16_t *depth, int i, int j) {
    if (BOUNDS(i, j))
        return CLAMP(depth[WIDTH * i + j]);
    return BIGNUM;
}

inline int depth_func(uint16_t *depth, int i, int j, double uy, double ux, double vy, double vx, int32_t t) {
    double d_x_inv = 1. / depth_samp(depth, i, j);
    return (depth_samp(depth, i + uy * d_x_inv, j + ux * d_x_inv) -
            depth_samp(depth, i + vy * d_x_inv, j + vx * d_x_inv)) >= t;
}

void depth_predict(uint16_t *depth, double *out_prob, uint16_t *out_ind, int32_t *trees, int32_t *links, double *leaves,
                   double *u, double *v, int32_t *t, int num_trees, int num_nodes, int num_leaves, int num_classes) {
    int i, j, k, l, m;
    double *prob_sum = malloc(sizeof *prob_sum * num_classes);
    double max_prob;
    int max_prob_ind;
    for (i = 0; i < HEIGHT; ++i)
        for (j = 0; j < WIDTH; ++j) {
            memset(prob_sum, 0, sizeof *prob_sum * num_classes);
            for (k = 0; k < num_trees; ++k) {
                l = trees[k];
                while (l >= 0)
                    l = links[2 * l + depth_func(depth, i, j, u[2 * l], u[2 * l + 1],
                                                 v[2 * l], v[2 * l + 1], t[l])];
                l = -l + 1;
                for (m = 0; m < num_classes; ++m)
                    prob_sum[m] += leaves[num_classes * l + m];
            }
            max_prob = 0.;
            max_prob_ind = 0;
            for (m = 0; m < num_classes; ++m)
                if (max_prob < prob_sum[m]) {
                    max_prob = prob_sum[m];
                    max_prob_ind = m;
                }
            out_ind[WIDTH * i + j] = max_prob_ind;
            out_prob[WIDTH * i + j] = max_prob;
        }
    free(prob_sum);
}
