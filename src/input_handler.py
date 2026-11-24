import math


def get_positive_integer(prompt):

    while True:

        try:
            value = int(input(prompt))

            if value <= 0:
                print("Please enter a positive integer.")
                continue

            return value

        except ValueError:
            print("Invalid input. Please enter an integer.")


def get_float(prompt):

    while True:

        try:
            value = float(input(prompt))

            if not math.isfinite(value):
                print("Please enter a finite number.")
                continue

            return value

        except ValueError:
            print("Invalid input. Please enter a valid number.")