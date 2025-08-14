class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32

    @classmethod
    def from_fahrenheit(cls, f):
        return cls((f - 32) * 5/9)

    @staticmethod
    def is_valid_temperature(temp):
        return -273.15 <= temp <= 1000
