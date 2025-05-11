import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn
import xml.etree.ElementTree as ET
import requests
import csv  

class RCS:
    def __init__(self, diameter, fmin, fmax):
        self.diameter = diameter
        self.fmin = fmin
        self.fmax = fmax
        self.radius = diameter / 2

    def calculate_rcs(self, frequency):
        wavelength = 3e8 / frequency
        k = 2 * np.pi / wavelength
        rcs_value = (wavelength**2 / np.pi) * np.abs(
            sum(
                (-1)**n * (n + 0.5) * (self.bn(n, k) - self.an(n, k))
                for n in range(1, 21)  # Limiting to 20 terms for practical computation
            )
        )**2
        return rcs_value

    def an(self, n, k):
        jn = spherical_jn(n, k * self.radius)
        hn = self.hn(n, k * self.radius)
        return jn / hn

    def bn(self, n, k):
        jn_prev = spherical_jn(n - 1, k * self.radius)
        jn = spherical_jn(n, k * self.radius)
        hn_prev = self.hn(n - 1, k * self.radius)
        hn = self.hn(n, k * self.radius)
        return (k * self.radius * jn_prev - n * jn) / (k * self.radius * hn_prev - n * hn)

    def hn(self, n, x):
        jn = spherical_jn(n, x)
        yn = spherical_yn(n, x)
        return jn + 1j * yn

    def plot_rcs(self):
        frequencies = np.linspace(self.fmin, self.fmax, 500)
        rcs_values = [self.calculate_rcs(f) for f in frequencies]

        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, rcs_values, label="RCS", color="blue")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("RCS (m²)")
        plt.title("Radar Cross Section vs Frequency")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Save results to CSV file
        with open("rcs_results.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(["Frequency (Hz)", "RCS (m²)"])
            # Write data rows
            for f, rcs in zip(frequencies, rcs_values):
                writer.writerow([f, rcs])

def load_variant_data(url, variant_number):
    response = requests.get(url)
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        variant = root.find(f"./variant[@number='{variant_number}']")
        diameter = float(variant.find('D').text)
        fmin = float(variant.find('fmin').text)
        fmax = float(variant.find('fmax').text)
        return diameter, fmin, fmax
    else:
        raise Exception("Failed to fetch the XML data.")

def main():
    url = "https://jenyay.net/uploads/Student/Modelling/task_rcs_02.xml"
    variant_number = 1  # Replace with your variant number
    diameter, fmin, fmax = load_variant_data(url, variant_number)

    rcs = RCS(diameter, fmin, fmax)
    rcs.plot_rcs()

if __name__ == "__main__":
    main()
