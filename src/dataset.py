# This file is for testing the ruff extension in VS Code. It contains some simple functions for basic arithmetic operations.
def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    return a / b if b != 0 else None


a = 10
b = 5
print(add(a, b))
print(subtract(a, b))
print(multiply(a, b))
print(divide(a, b))
