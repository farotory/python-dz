# Импорт необходимых библиотек
import numpy as np  # Для математических операций и работы с массивами
import matplotlib.pyplot as plt  # Для визуализации данных
from scipy.special import spherical_jn, spherical_yn  # Сферические функции Бесселя
import xml.etree.ElementTree as ET  # Для парсинга XML
import requests  # Для HTTP-запросов
import csv  # Для работы с CSV-файлами

class RCS:
    """Класс для расчета Эффективной Поверхности Рассеяния (ЭПР) сферы"""
    
    def __init__(self, diameter, fmin, fmax):
        """Инициализация параметров сферы и диапазона частот"""
        self.diameter = diameter  # Диаметр сферы в метрах
        self.fmin = fmin  # Минимальная частота в Гц
        self.fmax = fmax  # Максимальная частота в Гц
        self.radius = diameter / 2  # Вычисление радиуса

    def calculate_rcs(self, frequency):
        """Расчет ЭПР для конкретной частоты"""
        wavelength = 3e8 / frequency  # Длина волны (скорость света / частота)
        k = 2 * np.pi / wavelength  # Волновое число
        
        # Основная формула расчета ЭПР для сферы
        rcs_value = (wavelength**2 / np.pi) * np.abs(
            sum(
                (-1)**n * (n + 0.5) * (self.bn(n, k) - self.an(n, k))
                for n in range(1, 21)  # Суммируем 20 членов ряда
            )
        )**2
        return rcs_value  # Возвращаем значение ЭПР в м²

    def an(self, n, k):
        """Коэффициент an для сферической волны"""
        jn = spherical_jn(n, k * self.radius)  # Сферическая функция Бесселя 1-го рода
        hn = self.hn(n, k * self.radius)  # Сферическая функция Ханкеля
        return jn / hn

    def bn(self, n, k):
        """Коэффициент bn для сферической волны"""
        # Производные сферических функций
        jn_prev = spherical_jn(n - 1, k * self.radius)
        jn = spherical_jn(n, k * self.radius)
        hn_prev = self.hn(n - 1, k * self.radius)
        hn = self.hn(n, k * self.radius)
        return (k * self.radius * jn_prev - n * jn) / (k * self.radius * hn_prev - n * hn)

    def hn(self, n, x):
        """Сферическая функция Ханкеля 3-го рода"""
        jn = spherical_jn(n, x)  # Бессель
        yn = spherical_yn(n, x)  # Нейман
        return jn + 1j * yn  # Ханкель = Бессель + i*Нейман

    def plot_rcs(self):
        """Построение графика и сохранение результатов"""
        frequencies = np.linspace(self.fmin, self.fmax, 500)  # 500 точек в диапазоне
        rcs_values = [self.calculate_rcs(f) for f in frequencies]  # Расчет для каждой частоты

        # Настройка графика
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, rcs_values, label="RCS", color="blue")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("RCS (m²)")
        plt.title("Radar Cross Section vs Frequency")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Сохранение в CSV
        with open("rcs_results.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Frequency (Hz)", "RCS (m²)"])  # Заголовок
            for f, rcs in zip(frequencies, rcs_values):  # Данные
                writer.writerow([f, rcs])

def load_variant_data(url, variant_number):
    """Загрузка параметров варианта из XML"""
    response = requests.get(url)  # HTTP-запрос
    if response.status_code == 200:  # Если успешно
        root = ET.fromstring(response.content)  # Парсинг XML
        variant = root.find(f"./variant[@number='{variant_number}']")  # Поиск варианта
        diameter = float(variant.find('D').text)  # Диаметр
        fmin = float(variant.find('fmin').text)  # Минимальная частота
        fmax = float(variant.find('fmax').text)  # Максимальная частота
        return diameter, fmin, fmax
    else:
        raise Exception("Failed to fetch the XML data.")  # Ошибка загрузки

def main():
    """Основная функция"""
    url = "https://jenyay.net/uploads/Student/Modelling/task_rcs_02.xml"  # URL XML-файла
    variant_number = 3  # Номер варианта
    
    # Загрузка параметров
    diameter, fmin, fmax = load_variant_data(url, variant_number)

    # Расчет и визуализация
    rcs = RCS(diameter, fmin, fmax)
    rcs.plot_rcs()

if __name__ == "__main__":
    main()  # Точка входа