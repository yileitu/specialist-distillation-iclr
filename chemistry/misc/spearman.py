import scipy.stats as stats

# Twenty-seven (Spec, Distill) accuracy pairs extracted from the table
spec_scores = [
    # Chemistry (FFT, LoRA, LST) x (ID, SID, OOD)
    59.65, 57.72, 33.60, 28.57, 60.36, 42.65, 29.61, 60.20, 44.03,
    # Physics (FFT, LoRA, LST) x (ID, SID, OOD)
    90.73, 68.43, 16.31, 92.83, 68.98, 38.03, 93.65, 70.13, 38.50,
    # Multilingualism (FFT, LoRA, LST) x (ID, SID, OOD)
    35.93, 23.86, 6.90, 20.47, 43.16, 41.41, 35.68, 43.99, 39.08
]

distill_scores = [
    # Chemistry (FFT, LoRA, LST) x (ID, SID, OOD)
    40.55, 46.14, 25.20, 29.80, 59.31, 34.89, 29.54, 60.47, 42.11,
    # Physics (FFT, LoRA, LST) x (ID, SID, OOD)
    91.85, 68.51, 14.09, 91.93, 69.87, 38.26, 92.75, 70.84, 40.80,
    # Multilingualism (FFT, LoRA, LST) x (ID, SID, OOD)
    30.77, 27.78, 15.08, 22.97, 30.87, 35.89, 34.17, 43.87, 41.23
]

rho, p_value = stats.spearmanr(spec_scores, distill_scores)
print(f"Spearman's rho: {rho:.4f}")
print(f"P-value: {p_value:.4e}")


# Output:
# Spearman's rho: 0.952
# P-value: 2.456e-14