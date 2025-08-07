import matplotlib.pyplot as plt
import numpy as np

latent_dim = [(192 * 288) // 2, (192 * 288) // 8, (192 * 288) // 16, (192 * 288) // 32, (192 * 288) // 64]
maes = [0.004448328632861376, 0.005859964992851019, 0.006761632859706879, 0.008195861242711544, 0.010350600816309452]

plt.figure(figsize=(6, 4))
plt.plot(latent_dim, maes, marker='o')
plt.xlabel('Latent Dim')
plt.ylabel('Mean MAE')
plt.title('Latent Dim vs Mean MAE')
plt.xticks(latent_dim, labels=latent_dim, rotation=90)
plt.grid(False)
plt.tight_layout()
plt.savefig("mae.png")

