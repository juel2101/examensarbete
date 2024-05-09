import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

dt_bank_marketing = [75729, 77513, 71405, 67759, 65643]
mean_dt_bank_marketing = np.mean(dt_bank_marketing)
std_dt_bank_marketing = np.std(dt_bank_marketing)

print(f"Mean of DT on Bank Marketing: {mean_dt_bank_marketing:.0f}")
print(f"Standard Deviation of DT on Bank Marketing: {std_dt_bank_marketing:.0f}")


svm_bank_marketing = [139531, 133019, 137571, 138334, 141213]
mean_svm_bank_marketing = np.mean(svm_bank_marketing)
std_svm_bank_marketing = np.std(svm_bank_marketing)

print(f"Mean of SVM on Bank Marketing: {mean_svm_bank_marketing:.0f}")
print(f"Standard Deviation of SVM on Bank Marketing: {std_svm_bank_marketing:.0f}")


dt_mnist = [771795, 736473, 759243, 792469, 753760]
mean_dt_mnist = np.mean(dt_mnist)
std_dt_mnist = np.std(dt_mnist)

print(f"Mean of DT on MNIST: {mean_dt_mnist:.0f}")
print(f"Standard Deviation of DT on MNIST: {std_dt_mnist:.0f}")


optimized_dt_bank_marketing = [66149, 70412, 66265, 70060, 70287]
mean_optimized_dt_bank_marketing = np.mean(optimized_dt_bank_marketing)
std_optimized_dt_bank_marketing = np.std(optimized_dt_bank_marketing)

print(f"Mean of Optimized SVM on Bank Marketing: {mean_optimized_dt_bank_marketing:.0f}")
print(f"Standard Deviation of Optimized SVM on Bank Marketing: {std_optimized_dt_bank_marketing:.0f}")


optimized_svm_bank_marketing = [164971, 174209, 168253, 168800, 176446]
mean_optimized_svm_bank_marketing = np.mean(optimized_svm_bank_marketing)
std_optimized_svm_bank_marketing = np.std(optimized_svm_bank_marketing)

print(f"Mean of Optimized SVM on Bank Marketing: {mean_optimized_svm_bank_marketing:.0f}")
print(f"Standard Deviation of Optimized SVM on Bank Marketing: {std_optimized_svm_bank_marketing:.0f}")


data = pd.DataFrame({
    'Measurement Number': range(1, len(optimized_svm_bank_marketing) + 1),
    'Energy Consumption (Joules)': optimized_svm_bank_marketing
})


# Skapa ett diagram
plt.figure(figsize=(10, 6))
plt.bar(data['Measurement Number'], data['Energy Consumption (Joules)'], color='green', alpha=0.7, label='Energi per mätning')
plt.axhline(y=mean_optimized_svm_bank_marketing, color='r', linestyle='-', label=f'Medelvärde: {mean_optimized_svm_bank_marketing:.0f} mJ')

# Lägg till titel och etiketter
plt.title('Mätning av energiförbrukning')
plt.xlabel('Mätning')
plt.ylabel('Energiförbrukning (mJ)')
plt.xticks(data['Measurement Number'])

plt.legend()
plt.show()
