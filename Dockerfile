# Модель: Пошук IRR методами Ньютона та бісекції (5 семестр)
# Автор: Гетьманенко Людмила, група AI-231

FROM python:3.10-slim
WORKDIR /app

# Встановлення необхідних бібліотек для роботи моделі
RUN pip install --no-cache-dir numpy matplotlib numpy_financial

# Копіювання коду в контейнер
COPY irr_model.py .

# Запуск програми
CMD ["python", "irr_model.py"]
