import math
import matplotlib.pyplot as plt
import numpy as np
import json

def f(x):
    return 100 * math.sqrt(abs(1 - 0.01 * x**2) + 0.01 * abs(x + 10))

x_values = np.arange(-15, 5.01, 0.01)
y_values = [f(x) for x in x_values]

data = [{"x": x, "y": y} for x, y in zip(x_values, y_values)]
result = {"data": data}

with open('output.json', 'w') as json_file:
    json.dump(result, json_file, indent=4)

print("x \t y = f(x)")
for x, y in zip(x_values[:100], y_values[:100]):
    print(f"{x:.2f} \t {y:.5f}")  

plt.figure(figsize=(10, 5))
plt.plot(x_values, y_values, label="y = f(x)", color="green")
plt.title("График функции y = 100 * sqrt(|1 - 0.01x^2| + 0.01|x + 10|), x∈[-15;5]")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()
