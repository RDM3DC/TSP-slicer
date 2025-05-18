# Refining Curvature Parameters (\lambda and \mu)

This guide outlines a structured approach for tuning the curvature control parameters used in the Curve Memory workflow. After establishing a baseline, systematically refine the values to improve surface quality, dimensional accuracy, and print efficiency.

## 1. Define Performance Metrics

Track these metrics to quantify improvements:

- **Surface finish quality** – measure roughness or record visual inspection notes.
- **Dimensional accuracy** – compare actual dimensions against CAD targets.
- **Print speed and vibration** – monitor total print time and any unwanted resonance.
- **Material usage efficiency** – note filament used per print.

## 2. Initial Test Run

Start with a baseline set of parameters. Example initial guesses:

```text
\lambda = 0.5
\mu = 0.1
```

Run a complete print with these values and record the performance metrics.

## 3. Parameter Sweep (Grid Search)

Explore a grid of \lambda and \mu combinations to find promising regions. Example table:

| \lambda | \mu  |
| ------- | ---- |
| 0.1     | 0.01 |
| 0.1     | 0.05 |
| 0.1     | 0.10 |
| 0.3     | 0.01 |
| 0.3     | 0.05 |
| 0.3     | 0.10 |
| 0.5     | 0.01 |
| **0.5** | **0.05** *(baseline)* |
| 0.5     | 0.10 |
| 0.7     | 0.01 |
| 0.7     | 0.05 |
| 0.7     | 0.10 |

## 4. Evaluation and Statistical Analysis

After each test print, calculate the mean and standard deviation of your metrics. Compare results against the baseline to identify statistically significant improvements.

## 5. Optimization Technique

Once the promising range is known, apply a more targeted search. A simple Bayesian optimisation example using `skopt`:

```python
import numpy as np
from skopt import gp_minimize

# Lower score is better
def performance_metric(params):
    lambda_curvature, mu_decay = params
    adaptive_paths = run_curve_memory_slicer(lambda_curvature, mu_decay)
    quality_score = evaluate_print_quality(adaptive_paths)
    return quality_score

search_space = [(0.1, 1.0), (0.01, 0.2)]
result = gp_minimize(performance_metric, search_space, n_calls=15, random_state=42)

optimal_lambda, optimal_mu = result.x
print(f"Optimal \u03bb: {optimal_lambda:.4f}, Optimal \u03bc: {optimal_mu:.4f}")
```

## 6. Confirmatory Testing

Print again using the optimised parameters to confirm the predicted benefits in practice.

## 7. Optional: Adaptive Online Adjustment

Advanced setups can adapt parameters on-the-fly using layer-by-layer feedback:

```python
def adaptive_update_parameters(lambda_curvature, mu_decay, feedback):
    learning_rate = 0.05
    lambda_curvature -= learning_rate * feedback.gradient_lambda
    mu_decay -= learning_rate * feedback.gradient_mu
    lambda_curvature = np.clip(lambda_curvature, 0.1, 1.0)
    mu_decay = np.clip(mu_decay, 0.01, 0.2)
    return lambda_curvature, mu_decay
```

---

By combining a structured parameter sweep with statistical analysis and optional Bayesian optimisation, you can efficiently determine the \lambda and \mu values that yield the best print quality and performance.
