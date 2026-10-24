print("=== ELECTRIC FIELD SIMULATOR ===")

import math

print("Enter the position of the electric charge:")

x_q = float(input("Enter the x-coordinate (m): "))
y_q = float(input("Enter the y-coordinate (m): "))

q = float(input("Enter the electric charge (C): "))

print("Enter the point where you want to calculate the electric field:")

x_p = float(input("Enter the x-coordinate (m): "))
y_p = float(input("Enter the y-coordinate (m): "))

K = 8.9875517923e9

dx = x_p - x_q
dy = y_p - y_q

r=math.sqrt(dx**2 + dy**2)

if r==0:
    print("Error: the electric field is undefined at the charge position.")
else:
    Ex=K*q*dx/(r**3)
    Ey=K*q*dy/(r**3)
    
    print(f"Electric field along x: {Ex:.6f} N/C")
    
    print(f"Electric field along y: {Ey:.6f} N/C")

