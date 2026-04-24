# Модель: Пошук IRR методами Ньютона та бісекції (5 семестр)
# Автор: Гетьманенко Людмила, група AI-231

import numpy as np
import matplotlib.pyplot as plt
import numpy_financial as npf

class IRRModel:
    def __init__(self, cashflows, tol=1e-6, maxiter=100):
        """
        Ініціалізація моделі.
        :param cashflows: Список або numpy array грошових потоків (CF0, CF1, ...)
        :param tol: Бажана точність (tolerance)
        :param maxiter: Максимальна кількість ітерацій
        """
        self.cashflows = np.array(cashflows)
        self.tol = tol
        self.maxiter = maxiter

    def npv(self, r):
        """
        Розрахунок чистої приведеної вартості (NPV) для заданої ставки r.
        NPV = CFi / (1+r)^i
        """
        if 1 + r == 0:
            return float('inf')
        
        n = np.arange(len(self.cashflows))
        return np.sum(self.cashflows / ((1 + r) ** n))

    def dnpv_dr(self, r):
        """
        Обчислення похідної функції NPV за ставкою r.
        d(NPV)/dr = sum( -i * CFi / (1+r)^(i+1) )
        """
        if 1 + r == 0:
            return float('inf')
        
        n = np.arange(len(self.cashflows))
        t = n[1:]
        deriv_parts = -t * self.cashflows[1:] / ((1 + r) ** (t + 1))
        return np.sum(deriv_parts)

    def newton_method(self, x0=0.1):
        """
        Ітераційне знаходження IRR методом Ньютона–Рафсона.
        x_new = x_old - f(x) / f'(x)
        """
        r = x0
        for i in range(self.maxiter):
            f = self.npv(r)
            f_prime = self.dnpv_dr(r)

            if abs(f_prime) < self.tol:
                return None, i + 1

            r_new = r - f / f_prime

            if abs(r_new - r) < self.tol:
                return r_new, i + 1

            r = r_new
        
        return None, self.maxiter

    def bisection_method(self, a, b):
        """
        Альтернативний метод бісекції для гарантованого пошуку кореня.
        Вимагає, щоб npv(a) та npv(b) мали різні знаки.
        """
        fa = self.npv(a)
        fb = self.npv(b)

        if fa * fb >= 0:
            print(f"Метод бісекції: Помилка - npv(a) та npv(b) не мають різних знаків (a={a}, b={b})")
            return None, 0

        for i in range(self.maxiter):
            c = (a + b) / 2
            fc = self.npv(c)

            if abs(fc) < self.tol or (b - a) / 2 < self.tol:
                return c, i + 1

            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        
        return (a + b) / 2, self.maxiter

    def plot_results(self, r_points=None, r_range=(-0.5, 0.5), title=""):
        """
        Візуалізація залежності NPV(r) і позначення знайдених IRR.
        :param r_points: Список знайдених коренів (IRR) для позначення.
        :param r_range: Кортеж (min_r, max_r) для побудови графіка.
        :param title: Заголовок для графіка.
        """
        r_values = np.linspace(r_range[0], r_range[1], 400)
        npv_values = [self.npv(r) for r in r_values]

        plt.figure(figsize=(10, 6))
        plt.plot(r_values, npv_values, label='NPV(r)', color='blue')

        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.axvline(0, color='grey', linestyle=':', linewidth=0.5)

        if r_points:
            for r in r_points:
                plt.plot(r, self.npv(r), 'ro', label=f'Знайдено IRR')

        plt.title(f'Залежність NPV від ставки дисконтування\n{title}')
        plt.xlabel('Ставка дисконтування (r)')
        plt.ylabel('Чиста приведена вартість (NPV)')
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.ylim(np.min(npv_values) - 20, np.max(npv_values) + 20)

        plt.savefig(filename)
        print(f"Графік збережено у файл '{filename}'")
        plt.show()

if __name__ == '__main__':

    print("--- Сценарій 1: Ординарний CF ---")
    cf1 = [-100, 30, 50, 70]
    model1 = IRRModel(cf1)

    irr_n, iter_n = model1.newton_method(x0=0.1)
    print(f"[Ньютон] IRR = {irr_n:.6f} (знайдено за {iter_n} ітерацій)")

    irr_b, iter_b = model1.bisection_method(a=0, b=0.5)
    print(f"[Бісекція] IRR = {irr_b:.6f} (знайдено за {iter_b} ітерацій)")

    irr_npf = npf.irr(cf1)
    print(f"[numpy_financial] IRR = {irr_npf:.6f} (еталон)")

    model1.plot_results(r_points=[irr_n], 
                        r_range=(-0.1, 0.5), 
                        title=f"CF = {cf1}",
                        filename="npv_plot_scenario1.png")

    print("\n--- Сценарій 2: Множинні IRR ---")
    cf2 = [-100, 230, -132]
    model2 = IRRModel(cf2)

    irr_n1, iter_n1 = model2.newton_method(x0=0.05)
    irr_n2, iter_n2 = model2.newton_method(x0=0.25)
    
    print(f"[Ньютон 1] IRR = {irr_n1:.6f} (знайдено за {iter_n1} ітерацій, x0=0.05)")
    print(f"[Ньютон 2] IRR = {irr_n2:.6f} (знайдено за {iter_n2} ітерацій, x0=0.25)")

    irr_npf2 = npf.irr(cf2)
    print(f"[numpy_financial] IRR = {irr_npf2:.6f} (знаходить лише один, найближчий до 0.1)")
    
    model2.plot_results(r_points=[irr_n1, irr_n2], 
                        r_range=(0, 0.3), 
                        title=f"CF = {cf2} (Множинні IRR)",
                        filename="npv_plot_scenario2.png")
