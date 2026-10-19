// Teacher-student training with every GEMM emulated bit-exactly by a study datapath.
// A fixed random 64-32-10 tanh network labels Gaussian inputs; a 64-64-10 ReLU student learns
// them with FP32 master weights and host SGD. Forward GEMMs use the recipe's forward datapath;
// dW and dX GEMMs use its backward datapath. Reports test accuracy and loss per recipe and seed.
// usage: build/train_mlp [steps] [seeds]
#include "datapaths.h"
#include <functional>
#include <iostream>
#include <random>
#include <string>

using Matrix = std::vector<double>;  // row-major
using Dot = std::function<float(const Vec &, const Vec &, double, double)>;

constexpr unsigned in_dim = 64, hidden = 64, classes = 10, batch = 64;

static double tensor_amax(const Matrix &m) {
    double x = 0;
    for (double v : m) x = std::max(x, std::fabs(v));
    return x > 0 ? x : 1;
}

// C[M x N] = A[M x K] * B[K x N]; K is zero-padded to a multiple of 32 for the block datapaths.
static Matrix gemm(const Matrix &a, const Matrix &b, unsigned m, unsigned k, unsigned n, const Dot &dot) {
    unsigned kp = (k + 31) / 32 * 32;
    double amax_a = tensor_amax(a), amax_b = tensor_amax(b);
    std::vector<Vec> rows(m, Vec(kp, 0.0)), cols(n, Vec(kp, 0.0));
    for (unsigned i = 0; i < m; ++i)
        for (unsigned x = 0; x < k; ++x) rows[i][x] = a[i * k + x];
    for (unsigned j = 0; j < n; ++j)
        for (unsigned x = 0; x < k; ++x) cols[j][x] = b[x * n + j];
    Matrix c(m * n);
    for (unsigned i = 0; i < m; ++i)
        for (unsigned j = 0; j < n; ++j) c[i * n + j] = dot(rows[i], cols[j], amax_a, amax_b);
    return c;
}

static Matrix transpose(const Matrix &a, unsigned rows, unsigned cols) {
    Matrix t(rows * cols);
    for (unsigned i = 0; i < rows; ++i)
        for (unsigned j = 0; j < cols; ++j) t[j * rows + i] = a[i * cols + j];
    return t;
}

struct Recipe { std::string name; Dot forward, backward; };

static std::vector<Recipe> recipes() {
    Dot fp32 = [](const Vec &a, const Vec &b, double, double) { return fp32_dot(a, b); };
    Dot bf16 = [](const Vec &a, const Vec &b, double, double) { return dot8_chain(a, b, dot8_bf16); };
    Dot e4m3 = [](const Vec &a, const Vec &b, double x, double y) { return dot8_chain(a, b, dot8_e4m3, x, y); };
    Dot e5m2 = [](const Vec &a, const Vec &b, double x, double y) { return dot8_chain(a, b, dot8_e5m2, x, y); };
    Dot mxfp8 = [](const Vec &a, const Vec &b, double, double) { return mx_chain(a, b, MxElement::e4m3); };
    Dot mxfp4 = [](const Vec &a, const Vec &b, double, double) { return mx_chain(a, b, MxElement::e2m1); };
    return {{"fp32", fp32, fp32},       {"bf16_dot8", bf16, bf16},     {"fp8_tensor", e4m3, e5m2},
            {"mxfp8", mxfp8, mxfp8},    {"mxfp4_fwd_mxfp8_bwd", mxfp4, mxfp8}, {"mxfp4", mxfp4, mxfp4}};
}

struct Data { Matrix x; std::vector<unsigned> y; };

static Data teacher_data(unsigned n, std::mt19937_64 &rng, const Matrix &t1, const Matrix &t2) {
    std::normal_distribution<double> normal(0, 1);
    Data d{Matrix(n * in_dim), std::vector<unsigned>(n)};
    for (double &v : d.x) v = normal(rng);
    for (unsigned s = 0; s < n; ++s) {
        double h[32], best = -1e300;
        for (unsigned j = 0; j < 32; ++j) {
            h[j] = 0;
            for (unsigned i = 0; i < in_dim; ++i) h[j] += d.x[s * in_dim + i] * t1[i * 32 + j];
            h[j] = std::tanh(h[j]);
        }
        for (unsigned c = 0; c < classes; ++c) {
            double z = 0;
            for (unsigned j = 0; j < 32; ++j) z += h[j] * t2[j * classes + c];
            if (z > best) { best = z; d.y[s] = c; }
        }
    }
    return d;
}

struct Model { Matrix w1, b1, w2, b2; };

// Forward pass through the datapath; returns logits and fills hidden pre-activations and ReLU output.
static Matrix forward(const Model &m, const Matrix &x, unsigned n, const Dot &dot, Matrix &pre, Matrix &act) {
    pre = gemm(x, m.w1, n, in_dim, hidden, dot);
    act = pre;
    for (unsigned s = 0; s < n; ++s)
        for (unsigned j = 0; j < hidden; ++j) {
            pre[s * hidden + j] += m.b1[j];
            act[s * hidden + j] = std::max(0.0, pre[s * hidden + j]);
        }
    Matrix logits = gemm(act, m.w2, n, hidden, classes, dot);
    for (unsigned s = 0; s < n; ++s)
        for (unsigned c = 0; c < classes; ++c) logits[s * classes + c] += m.b2[c];
    return logits;
}

// Softmax cross-entropy; returns mean loss and writes dlogits (already divided by n).
static double softmax_loss(const Matrix &logits, const std::vector<unsigned> &y, unsigned offset, unsigned n,
                           Matrix *grad, unsigned *correct) {
    double loss = 0;
    for (unsigned s = 0; s < n; ++s) {
        const double *z = &logits[s * classes];
        double peak = *std::max_element(z, z + classes), sum = 0;
        for (unsigned c = 0; c < classes; ++c) sum += std::exp(z[c] - peak);
        unsigned label = y[offset + s];
        loss += std::log(sum) - (z[label] - peak);
        if (correct && std::max_element(z, z + classes) - z == long(label)) ++*correct;
        if (grad)
            for (unsigned c = 0; c < classes; ++c)
                (*grad)[s * classes + c] = (std::exp(z[c] - peak) / sum - (c == label)) / n;
    }
    return loss / n;
}

int main(int argc, char **argv) {
    unsigned steps = argc > 1 ? unsigned(std::stoul(argv[1])) : 1500;
    unsigned seeds = argc > 2 ? unsigned(std::stoul(argv[2])) : 3;
    std::mt19937_64 teacher_rng(2026);
    std::normal_distribution<double> normal(0, 1);
    Matrix t1(in_dim * 32), t2(32 * classes);
    for (double &v : t1) v = normal(teacher_rng) / std::sqrt(double(in_dim)) * 2;
    for (double &v : t2) v = normal(teacher_rng) / std::sqrt(32.0);
    Data train = teacher_data(4096, teacher_rng, t1, t2), test = teacher_data(2048, teacher_rng, t1, t2);
    std::cout << "recipe seed steps train_loss test_loss test_accuracy\n";
    for (const Recipe &r : recipes()) {
        for (unsigned seed = 0; seed < seeds; ++seed) {
            std::mt19937_64 rng(100 + seed);
            Model m{Matrix(in_dim * hidden), Matrix(hidden, 0.0), Matrix(hidden * classes), Matrix(classes, 0.0)};
            for (double &v : m.w1) v = normal(rng) * std::sqrt(2.0 / in_dim);
            for (double &v : m.w2) v = normal(rng) * std::sqrt(1.0 / hidden);
            double train_loss = 0;
            for (unsigned step = 0; step < steps; ++step) {
                unsigned offset = (step * batch) % train.y.size();
                Matrix x(train.x.begin() + offset * in_dim, train.x.begin() + (offset + batch) * in_dim);
                Matrix pre, act, dlogits(batch * classes);
                Matrix logits = forward(m, x, batch, r.forward, pre, act);
                train_loss = softmax_loss(logits, train.y, offset, batch, &dlogits, nullptr);
                Matrix dw2 = gemm(transpose(act, batch, hidden), dlogits, hidden, batch, classes, r.backward);
                Matrix dact = gemm(dlogits, transpose(m.w2, hidden, classes), batch, classes, hidden, r.backward);
                for (unsigned i = 0; i < dact.size(); ++i) dact[i] = pre[i] > 0 ? dact[i] : 0;
                Matrix dw1 = gemm(transpose(x, batch, in_dim), dact, in_dim, batch, hidden, r.backward);
                constexpr double lr = 0.2;
                for (unsigned i = 0; i < m.w1.size(); ++i) m.w1[i] -= lr * dw1[i];
                for (unsigned i = 0; i < m.w2.size(); ++i) m.w2[i] -= lr * dw2[i];
                for (unsigned j = 0; j < hidden; ++j) {
                    double g = 0;
                    for (unsigned s = 0; s < batch; ++s) g += dact[s * hidden + j];
                    m.b1[j] -= lr * g;
                }
                for (unsigned c = 0; c < classes; ++c) {
                    double g = 0;
                    for (unsigned s = 0; s < batch; ++s) g += dlogits[s * classes + c];
                    m.b2[c] -= lr * g;
                }
            }
            Matrix pre, act;
            unsigned correct = 0;
            Matrix logits = forward(m, test.x, unsigned(test.y.size()), r.forward, pre, act);
            double test_loss = softmax_loss(logits, test.y, 0, unsigned(test.y.size()), nullptr, &correct);
            std::cout << r.name << ' ' << seed << ' ' << steps << ' ' << train_loss << ' ' << test_loss << ' '
                      << double(correct) / test.y.size() << std::endl;
        }
    }
    return 0;
}
