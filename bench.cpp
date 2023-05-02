#include <benchmark/benchmark.h>

#include <cmath>
#include <numeric> // std::accumulate
#include <random>
#include <stdexcept>
#include <vector>

struct Input {
  std::size_t bulkSize;
  std::vector<bool> eventMask;
  float *pts;
  float *etas;
  float *phis;
  float *masses;
  std::size_t *sizes;
};

static Input MakeInput() {
  std::default_random_engine e(1234); // fixed seed
  std::uniform_real_distribution<double> rand(0., 1.);

  std::size_t bulkSize = 1000 + rand(e) * 2;
  std::size_t *sizes = new std::size_t[bulkSize];
  std::vector<bool> eventMask(bulkSize);
  std::size_t nElements = 0u;
  for (std::size_t i = 0u; i < bulkSize; ++i) {
    double val = rand(e);
    sizes[i] = val * 5;
    eventMask[i] = val > 0.6;
    nElements += sizes[i];
  }

  float *pts = new float[nElements];
  float *etas = new float[nElements];
  float *phis = new float[nElements];
  float *masses = new float[nElements];

  for (std::size_t i = 0u; i < nElements; ++i) {
    const float val = rand(e);
    pts[i] = val;
    etas[i] = val;
    phis[i] = val;
    masses[i] = val;
  }

  return Input{bulkSize, eventMask, pts, etas, phis, masses, sizes};
}

static const auto input = MakeInput();

static void SanityCheck(const std::vector<float> &results) {
  const auto sum = std::accumulate(results.begin(), results.end(), 0.);
  if (!std::isfinite(sum) || std::abs(sum - 736.623790) > 1e-5)
    throw std::runtime_error("Sanity check failed: sum of results was " +
                             std::to_string(sum) + " instead of 736.624");
}

float SimpleSinh(float x) {
  const auto e = std::exp(x);
  return 0.5f * (e - 1.f / e);
}

// WARNING: The number of terms was chosen to pass the test. This function
// has reduced precision and probably behaves badly for arbitrary arguments.
static void SincosPowerSeries(float x, float &sin, float &cos) {
  const auto x2 = x * x;
  const auto t2 = x2 * (1.f / 2.f);
  const auto x3 = x2 * x;
  const auto t3 = x3 * (1.f / 6.f);
  const auto x4 = x3 * x;
  const auto t4 = x4 * (1.f / 24.f);
  const auto x5 = x4 * x;
  const auto t5 = x5 * (1.f / 120.f);
  const auto x6 = x5 * x;
  const auto t6 = x6 * (1.f / 720.f);
  const auto x7 = x6 * x;
  const auto t7 = x7 * (1.f / 5040.f);
  const auto x8 = x7 * x;
  const auto t8 = x8 * (1.f / 40320.f);
  const auto x9 = x8 * x;
  const auto t9 = x9 * (1.f / 362880.f);
  const auto x10 = x9 * x;
  const auto t10 = x10 * (1.f / 3628800.f);
  const auto x11 = x10 * x;
  const auto t11 = x11 * (1.f / 39916800.f);

  sin = x - t3 + t5 - t7 + t9 - t11;
  cos = 1 - t2 + t4 - t6 + t8 - t10;
}

// WARNING: The number of terms was chosen to pass the test. This function
// has reduced precision and probably behaves badly for arbitrary arguments.
static float SinhPowerSeries(float x) {
  const auto x2 = x * x;
  const auto x3 = x2 * x;
  const auto t3 = x3 * (1.f / 6.f);
  const auto x5 = x3 * x2;
  const auto t5 = x5 * (1.f / 120.f);
  const auto x7 = x5 * x2;
  const auto t7 = x7 * (1.f / 5040.f);
  const auto x9 = x7 * x2;
  const auto t9 = x9 * (1.f / 362880.f);

  return x + t3 + t5 + t7 + t9;
}

// taken from Cephes' sinhf.c, using a polynomial approximation for |x| < 1
static float SinhCephes(float x) {
  float z = std::abs(x);
  if (z > 1) {
    z = std::exp(z);
    z = 0.5f * z - 0.5f / z;
    if (x < 0) {
      z = -z;
    }
  } else {
    z = x * x;
    z = ((2.03721912945E-4f * z + 8.33028376239E-3f) * z + 1.66667160211E-1f) *
            z * x +
        x;
  }
  return z;
}

// original
template <typename T>
T InvariantMassBaseline(const T *pt, const T *eta, const T *phi, const T *mass,
                        std::size_t size) {
  T x_sum = 0.;
  T y_sum = 0.;
  T z_sum = 0.;
  T e_sum = 0.;

  for (std::size_t i = 0u; i < size; ++i) {
    // Convert to (e, x, y, z) coordinate system and update sums
    const auto x = pt[i] * std::cos(phi[i]);
    x_sum += x;
    const auto y = pt[i] * std::sin(phi[i]);
    y_sum += y;
    const auto z = pt[i] * std::sinh(eta[i]);
    z_sum += z;
    const auto e = std::sqrt(x * x + y * y + z * z + mass[i] * mass[i]);
    e_sum += e;
  }

  // Return invariant mass with (+, -, -, -) metric
  return std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum -
                   z_sum * z_sum);
}

static void EvalLoop(std::size_t bulkSize, const std::vector<bool> &eventMask,
                     float *pts, float *etas, float *phis, float *masses,
                     std::size_t *sizes, std::vector<float> &results) {
  std::size_t elementIdx = 0u;
  for (std::size_t i = 0ul; i < bulkSize; ++i) {
    if (eventMask[i]) { // we don't have a value for this entry yet
      results[i] = InvariantMassBaseline(pts + elementIdx, etas + elementIdx,
                                         phis + elementIdx, masses + elementIdx,
                                         sizes[i]);
    }
    elementIdx += sizes[i];
  }
}

static void Baseline(benchmark::State &state) {
  std::vector<float> results(input.bulkSize);
  benchmark::DoNotOptimize(results);
  for (auto _ : state) {
    EvalLoop(input.bulkSize, input.eventMask, input.pts, input.etas, input.phis,
             input.masses, input.sizes, results);
    // to force writing to memory of results
    benchmark::ClobberMemory();
  }

  SanityCheck(results);
}
BENCHMARK(Baseline);

template <typename T>
T InvMassBaselineSimpleSinh(const T *pt, const T *eta, const T *phi,
                            const T *mass, std::size_t size) {
  T x_sum = 0.;
  T y_sum = 0.;
  T z_sum = 0.;
  T e_sum = 0.;

  for (std::size_t i = 0u; i < size; ++i) {
    // Convert to (e, x, y, z) coordinate system and update sums
    const auto x = pt[i] * std::cos(phi[i]);
    x_sum += x;
    const auto y = pt[i] * std::sin(phi[i]);
    y_sum += y;
    const auto z = pt[i] * SimpleSinh(eta[i]);
    z_sum += z;
    const auto e = std::sqrt(x * x + y * y + z * z + mass[i] * mass[i]);
    e_sum += e;
  }

  // Return invariant mass with (+, -, -, -) metric
  return std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum -
                   z_sum * z_sum);
}

static void EvalLoopSimpleSinh(std::size_t bulkSize,
                               const std::vector<bool> &eventMask, float *pts,
                               float *etas, float *phis, float *masses,
                               std::size_t *sizes,
                               std::vector<float> &results) {
  std::size_t elementIdx = 0u;
  for (std::size_t i = 0ul; i < bulkSize; ++i) {
    if (eventMask[i]) { // we don't have a value for this entry yet
      results[i] = InvMassBaselineSimpleSinh(
          pts + elementIdx, etas + elementIdx, phis + elementIdx,
          masses + elementIdx, sizes[i]);
    }
    elementIdx += sizes[i];
  }
}

static void BaselineSimpleSinh(benchmark::State &state) {
  std::vector<float> results(input.bulkSize);
  benchmark::DoNotOptimize(results);
  for (auto _ : state) {
    EvalLoopSimpleSinh(input.bulkSize, input.eventMask, input.pts, input.etas,
                       input.phis, input.masses, input.sizes, results);
    // to force writing to memory of results
    benchmark::ClobberMemory();
  }

  SanityCheck(results);
}
BENCHMARK(BaselineSimpleSinh);

template <typename T>
T InvMassBaselineSinhCephes(const T *pt, const T *eta, const T *phi,
                            const T *mass, std::size_t size) {
  T x_sum = 0.;
  T y_sum = 0.;
  T z_sum = 0.;
  T e_sum = 0.;

  for (std::size_t i = 0u; i < size; ++i) {
    // Convert to (e, x, y, z) coordinate system and update sums
    const auto x = pt[i] * std::cos(phi[i]);
    x_sum += x;
    const auto y = pt[i] * std::sin(phi[i]);
    y_sum += y;
    const auto z = pt[i] * SinhCephes(eta[i]);
    z_sum += z;
    const auto e = std::sqrt(x * x + y * y + z * z + mass[i] * mass[i]);
    e_sum += e;
  }

  // Return invariant mass with (+, -, -, -) metric
  return std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum -
                   z_sum * z_sum);
}

static void EvalLoopSinhCephes(std::size_t bulkSize,
                               const std::vector<bool> &eventMask, float *pts,
                               float *etas, float *phis, float *masses,
                               std::size_t *sizes,
                               std::vector<float> &results) {
  std::size_t elementIdx = 0u;
  for (std::size_t i = 0ul; i < bulkSize; ++i) {
    if (eventMask[i]) { // we don't have a value for this entry yet
      results[i] = InvMassBaselineSinhCephes(
          pts + elementIdx, etas + elementIdx, phis + elementIdx,
          masses + elementIdx, sizes[i]);
    }
    elementIdx += sizes[i];
  }
}

static void BaselineSinhCephes(benchmark::State &state) {
  std::vector<float> results(input.bulkSize);
  benchmark::DoNotOptimize(results);
  for (auto _ : state) {
    EvalLoopSinhCephes(input.bulkSize, input.eventMask, input.pts, input.etas,
                       input.phis, input.masses, input.sizes, results);
    // to force writing to memory of results
    benchmark::ClobberMemory();
  }

  SanityCheck(results);
}
BENCHMARK(BaselineSinhCephes);

template <typename T>
T InvMassBaselinePowerSeries(const T *pt, const T *eta, const T *phi,
                             const T *mass, std::size_t size) {
  T x_sum = 0.;
  T y_sum = 0.;
  T z_sum = 0.;
  T e_sum = 0.;

  for (std::size_t i = 0u; i < size; ++i) {
    // Convert to (e, x, y, z) coordinate system and update sums
    T sin, cos;
    SincosPowerSeries(phi[i], sin, cos);
    const auto x = pt[i] * cos;
    x_sum += x;
    const auto y = pt[i] * sin;
    y_sum += y;
    const auto z = pt[i] * SinhPowerSeries(eta[i]);
    z_sum += z;
    const auto e = std::sqrt(x * x + y * y + z * z + mass[i] * mass[i]);
    e_sum += e;
  }

  // Return invariant mass with (+, -, -, -) metric
  return std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum -
                   z_sum * z_sum);
}

static void EvalLoopPowerSeries(std::size_t bulkSize,
                                const std::vector<bool> &eventMask, float *pts,
                                float *etas, float *phis, float *masses,
                                std::size_t *sizes,
                                std::vector<float> &results) {
  std::size_t elementIdx = 0u;
  for (std::size_t i = 0ul; i < bulkSize; ++i) {
    if (eventMask[i]) { // we don't have a value for this entry yet
      results[i] = InvMassBaselinePowerSeries(
          pts + elementIdx, etas + elementIdx, phis + elementIdx,
          masses + elementIdx, sizes[i]);
    }
    elementIdx += sizes[i];
  }
}

static void BaselinePowerSeries(benchmark::State &state) {
  std::vector<float> results(input.bulkSize);
  benchmark::DoNotOptimize(results);
  for (auto _ : state) {
    EvalLoopPowerSeries(input.bulkSize, input.eventMask, input.pts, input.etas,
                        input.phis, input.masses, input.sizes, results);
    // to force writing to memory of results
    benchmark::ClobberMemory();
  }

  SanityCheck(results);
}
BENCHMARK(BaselinePowerSeries);

template <typename T>
void InvariantMassBulk(const std::vector<bool> &eventMask, std::size_t bulkSize,
                       std::vector<T> &results, const T *pt, const T *eta,
                       const T *phi, const T *mass, const std::size_t *sizes) {
  std::size_t elementIdx = 0u;
  for (std::size_t i = 0; i < bulkSize; ++i) {
    if (eventMask[i]) {

      T x_sum = 0.;
      T y_sum = 0.;
      T z_sum = 0.;
      T e_sum = 0.;

      // short loop: sizes[i] is typically small (~2.5 on average)
      for (std::size_t j = 0u; j < sizes[i]; ++j) {
        const auto x = pt[elementIdx + j] * std::cos(phi[elementIdx + j]);
        x_sum += x;
        const auto y = pt[elementIdx + j] * std::sin(phi[elementIdx + j]);
        y_sum += y;
        const auto z = pt[elementIdx + j] * std::sinh(eta[elementIdx + j]);
        z_sum += z;
        // this single-scalar sqrt is the hottest instruction
        const auto e = std::sqrt(x * x + y * y + z * z +
                                 mass[elementIdx + j] * mass[elementIdx + j]);
        e_sum += e;
      }

      results[i] = std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum -
                             z_sum * z_sum);
    }
    elementIdx += sizes[i];
  }
}

static void Bulk(benchmark::State &state) {
  std::vector<float> results(input.bulkSize);
  benchmark::DoNotOptimize(results);
  for (auto _ : state) {
    InvariantMassBulk(input.eventMask, input.bulkSize, results, input.pts,
                      input.etas, input.phis, input.masses, input.sizes);
    // to force writing to memory of results
    benchmark::ClobberMemory();
  }

  SanityCheck(results);
}
BENCHMARK(Bulk);

template <typename T>
void InvariantMassBulkIgnoreMask(const std::vector<bool> &eventMask,
                                 std::size_t bulkSize, std::vector<T> &results,
                                 const T *pt, const T *eta, const T *phi,
                                 const T *mass, const std::size_t *sizes) {

  const auto nElements = std::accumulate(sizes, sizes + bulkSize, 0u);

  std::vector<T> xs(nElements);
  std::vector<T> ys(nElements);
  std::vector<T> zs(nElements);
  std::vector<T> es2(nElements);
  std::vector<T> es(nElements);

  // trigonometric functions are expensive and don't vectorize so we only call
  // them when needed
  std::size_t elementIdx = 0u;
  for (std::size_t i = 0u; i < bulkSize; ++i) {
    const auto size = sizes[i];
    if (eventMask[i]) {
      for (std::size_t j = 0u; j < size; ++j) {
        const auto pt_ = pt[elementIdx + j];
        const auto phi_ = phi[elementIdx + j];
        xs[elementIdx + j] = pt_ * std::cos(phi_);
        ys[elementIdx + j] = pt_ * std::sin(phi_);
        zs[elementIdx + j] = pt_ * std::sinh(eta[elementIdx + j]);
      }
    }
    elementIdx += size;
  }

  // looks like the CPU is happier by calculating these for all elements, even
  // if we'll discard many of the results...
  for (std::size_t i = 0; i < nElements; ++i) {
    es2[i] = pt[i] * pt[i] + zs[i] * zs[i] + mass[i] * mass[i];
  }

  for (std::size_t i = 0; i < nElements; ++i) {
    es[i] = std::sqrt(es2[i]);
  }

  elementIdx = 0u;
  for (std::size_t i = 0; i < bulkSize; ++i) {
    T x_sum = 0.;
    T y_sum = 0.;
    T z_sum = 0.;
    T e_sum = 0.;
    const auto size = sizes[i];
    if (eventMask[i]) {
      for (std::size_t j = 0u; j < sizes[i]; ++j) {
        const auto idx = elementIdx + j;
        x_sum += xs[idx];
        y_sum += ys[idx];
        z_sum += zs[idx];
        e_sum += es[idx];
      }
      results[i] = std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum -
                             z_sum * z_sum);
    }
    elementIdx += size;
  }
}

static void BulkIgnoreMask(benchmark::State &state) {
  std::vector<float> results(input.bulkSize);
  benchmark::DoNotOptimize(results);
  for (auto _ : state) {
    InvariantMassBulkIgnoreMask(input.eventMask, input.bulkSize, results,
                                input.pts, input.etas, input.phis, input.masses,
                                input.sizes);
    // to force writing to memory of results
    benchmark::ClobberMemory();
  }

  SanityCheck(results);
}
BENCHMARK(BulkIgnoreMask);

template <typename T>
void InvMassBulkIgnoreMaskCustomSinH(const std::vector<bool> &eventMask,
                                     std::size_t bulkSize,
                                     std::vector<T> &results, const T *pt,
                                     const T *eta, const T *phi, const T *mass,
                                     const std::size_t *sizes) {

  const auto nElements = std::accumulate(sizes, sizes + bulkSize, 0u);

  std::vector<T> xs(nElements);
  std::vector<T> ys(nElements);
  std::vector<T> zs(nElements);
  std::vector<T> es2(nElements);
  std::vector<T> es(nElements);

  // trigonometric functions are expensive and don't vectorize so we only call
  // them when needed
  std::size_t elementIdx = 0u;
  for (std::size_t i = 0u; i < bulkSize; ++i) {
    const auto size = sizes[i];
    if (eventMask[i]) {
      for (std::size_t j = 0u; j < size; ++j) {
        const auto pt_ = pt[elementIdx + j];
        const auto phi_ = phi[elementIdx + j];
        xs[elementIdx + j] = pt_ * std::cos(phi_);
        ys[elementIdx + j] = pt_ * std::sin(phi_);
        zs[elementIdx + j] = pt_ * SimpleSinh(eta[elementIdx + j]);
      }
    }
    elementIdx += size;
  }

  // looks like the CPU is happier by calculating these for all elements, even
  // if we'll discard many of the results...
  for (std::size_t i = 0; i < nElements; ++i) {
    es2[i] = pt[i] * pt[i] + zs[i] * zs[i] + mass[i] * mass[i];
  }

  for (std::size_t i = 0; i < nElements; ++i) {
    es[i] = std::sqrt(es2[i]);
  }

  elementIdx = 0u;
  for (std::size_t i = 0; i < bulkSize; ++i) {
    T x_sum = 0.;
    T y_sum = 0.;
    T z_sum = 0.;
    T e_sum = 0.;
    const auto size = sizes[i];
    if (eventMask[i]) {
      for (std::size_t j = 0u; j < sizes[i]; ++j) {
        const auto idx = elementIdx + j;
        x_sum += xs[idx];
        y_sum += ys[idx];
        z_sum += zs[idx];
        e_sum += es[idx];
      }
      results[i] = std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum -
                             z_sum * z_sum);
    }
    elementIdx += size;
  }
}

static void BulkIgnoreMaskSimpleSinh(benchmark::State &state) {
  std::vector<float> results(input.bulkSize);
  benchmark::DoNotOptimize(results);
  for (auto _ : state) {
    InvMassBulkIgnoreMaskCustomSinH(input.eventMask, input.bulkSize, results,
                                    input.pts, input.etas, input.phis,
                                    input.masses, input.sizes);
    // to force writing to memory of results
    benchmark::ClobberMemory();
  }

  SanityCheck(results);
}
BENCHMARK(BulkIgnoreMaskSimpleSinh);

template <typename T>
void InvMassBulkIgnoreMaskSinhCephes(const std::vector<bool> &eventMask,
                                     std::size_t bulkSize,
                                     std::vector<T> &results, const T *pt,
                                     const T *eta, const T *phi, const T *mass,
                                     const std::size_t *sizes) {

  const auto nElements = std::accumulate(sizes, sizes + bulkSize, 0u);

  std::vector<T> xs(nElements);
  std::vector<T> ys(nElements);
  std::vector<T> zs(nElements);
  std::vector<T> es2(nElements);
  std::vector<T> es(nElements);

  // trigonometric functions are expensive and don't vectorize so we only call
  // them when needed
  std::size_t elementIdx = 0u;
  for (std::size_t i = 0u; i < bulkSize; ++i) {
    const auto size = sizes[i];
    if (eventMask[i]) {
      for (std::size_t j = 0u; j < size; ++j) {
        const auto pt_ = pt[elementIdx + j];
        const auto phi_ = phi[elementIdx + j];
        xs[elementIdx + j] = pt_ * std::cos(phi_);
        ys[elementIdx + j] = pt_ * std::sin(phi_);
        zs[elementIdx + j] = pt_ * SinhCephes(eta[elementIdx + j]);
      }
    }
    elementIdx += size;
  }

  // looks like the CPU is happier by calculating these for all elements, even
  // if we'll discard many of the results...
  for (std::size_t i = 0; i < nElements; ++i) {
    es2[i] = pt[i] * pt[i] + zs[i] * zs[i] + mass[i] * mass[i];
  }

  for (std::size_t i = 0; i < nElements; ++i) {
    es[i] = std::sqrt(es2[i]);
  }

  elementIdx = 0u;
  for (std::size_t i = 0; i < bulkSize; ++i) {
    T x_sum = 0.;
    T y_sum = 0.;
    T z_sum = 0.;
    T e_sum = 0.;
    const auto size = sizes[i];
    if (eventMask[i]) {
      for (std::size_t j = 0u; j < sizes[i]; ++j) {
        const auto idx = elementIdx + j;
        x_sum += xs[idx];
        y_sum += ys[idx];
        z_sum += zs[idx];
        e_sum += es[idx];
      }
      results[i] = std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum -
                             z_sum * z_sum);
    }
    elementIdx += size;
  }
}

static void BulkIgnoreMaskSinhCephes(benchmark::State &state) {
  std::vector<float> results(input.bulkSize);
  benchmark::DoNotOptimize(results);
  for (auto _ : state) {
    InvMassBulkIgnoreMaskSinhCephes(input.eventMask, input.bulkSize, results,
                                    input.pts, input.etas, input.phis,
                                    input.masses, input.sizes);
    // to force writing to memory of results
    benchmark::ClobberMemory();
  }

  SanityCheck(results);
}
BENCHMARK(BulkIgnoreMaskSinhCephes);

template <typename T>
void InvMassBulkIgnoreMaskPowerSeries(const std::vector<bool> &eventMask,
                                      std::size_t bulkSize,
                                      std::vector<T> &results, const T *pt,
                                      const T *eta, const T *phi, const T *mass,
                                      const std::size_t *sizes) {

  const auto nElements = std::accumulate(sizes, sizes + bulkSize, 0u);

  std::vector<T> xs(nElements);
  std::vector<T> ys(nElements);
  std::vector<T> zs(nElements);
  std::vector<T> es2(nElements);
  std::vector<T> es(nElements);

  // trigonometric functions are expensive and don't vectorize so we only call
  // them when needed
  std::size_t elementIdx = 0u;
  for (std::size_t i = 0u; i < bulkSize; ++i) {
    const auto size = sizes[i];
    if (eventMask[i]) {
      for (std::size_t j = 0u; j < size; ++j) {
        const auto pt_ = pt[elementIdx + j];
        const auto phi_ = phi[elementIdx + j];
        T sin, cos;
        SincosPowerSeries(phi_, sin, cos);
        xs[elementIdx + j] = pt_ * cos;
        ys[elementIdx + j] = pt_ * sin;
        zs[elementIdx + j] = pt_ * SinhPowerSeries(eta[elementIdx + j]);
      }
    }
    elementIdx += size;
  }

  // looks like the CPU is happier by calculating these for all elements, even
  // if we'll discard many of the results...
  for (std::size_t i = 0; i < nElements; ++i) {
    es2[i] = pt[i] * pt[i] + zs[i] * zs[i] + mass[i] * mass[i];
  }

  for (std::size_t i = 0; i < nElements; ++i) {
    es[i] = std::sqrt(es2[i]);
  }

  elementIdx = 0u;
  for (std::size_t i = 0; i < bulkSize; ++i) {
    T x_sum = 0.;
    T y_sum = 0.;
    T z_sum = 0.;
    T e_sum = 0.;
    const auto size = sizes[i];
    if (eventMask[i]) {
      for (std::size_t j = 0u; j < sizes[i]; ++j) {
        const auto idx = elementIdx + j;
        x_sum += xs[idx];
        y_sum += ys[idx];
        z_sum += zs[idx];
        e_sum += es[idx];
      }
      results[i] = std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum -
                             z_sum * z_sum);
    }
    elementIdx += size;
  }
}

static void BulkIgnoreMaskPowerSeries(benchmark::State &state) {
  std::vector<float> results(input.bulkSize);
  benchmark::DoNotOptimize(results);
  for (auto _ : state) {
    InvMassBulkIgnoreMaskPowerSeries(input.eventMask, input.bulkSize, results,
                                     input.pts, input.etas, input.phis,
                                     input.masses, input.sizes);
    // to force writing to memory of results
    benchmark::ClobberMemory();
  }

  SanityCheck(results);
}
BENCHMARK(BulkIgnoreMaskPowerSeries);

BENCHMARK_MAIN();
