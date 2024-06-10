from math import sqrt

def compute_and_print_coeffecients(
        name: str,
        s1: float,
        s2: float,
        s3: float,
        s4: float,
        s5: float,
        beta: float = 1.0,
    ):

    s1 = s1 * (beta ** 1)
    s2 = s2 * (beta ** 2)
    s3 = s3 * (beta ** 3)
    s4 = s4 * (beta ** 4)
    s5 = s5 * (beta ** 5)

    a1 = s3/s1 + 3/4 * (s2/s1)**2
    a2 = 1/4 * s5/s1 + 5/8 * (s4/s1) * (s2/s1) + 5/12 * (s3/s1)**2

    b1 = s3/s1 + (s2/s1)**2
    b2 = 1/4 * (s5/s1) + (s4/s1) * (s2/s1) + 3/4 * (s3/s1)**2

    print(f"function: {name}")
    print(f"a1 = {a1}")
    print(f"a2 = {a2}")
    print(f"b1 = {b1}")
    print(f"b2 = {b2}")
    print()

    # print(f"check: {name}")
    # print(f"a1 = {a1 / (beta ** 2)}")
    # print(f"a2 = {a2 / (beta ** 4)}")
    # print(f"b1 = {b1 / (beta ** 2)}")
    # print(f"b2 = {b2 / (beta ** 4)}")
    # print()

if __name__ == "__main__":
    print()
    compute_and_print_coeffecients("tanh", s1=1, s2=0, s3=-2, s4=0, s5=16)
    compute_and_print_coeffecients("tanh, β=0.05", s1=1, s2=0, s3=-2, s4=0, s5=16, beta=0.05)
    
    compute_and_print_coeffecients("sin", s1=1, s2=0, s3=-1, s4=0, s5=1)
    compute_and_print_coeffecients("sin, β=0.05", s1=1, s2=0, s3=-1, s4=0, s5=1, beta=0.05)

    compute_and_print_coeffecients("sigmoid (shifted)", s1=1/4, s2=0, s3=-1/8, s4=0, s5=1/4)

    eps = 0.01
    s4_poly1 = -(3/8 + 8/5 * eps)
    compute_and_print_coeffecients("poly1", s1=1, s2=1, s3=-3/4, s4=s4_poly1, s5=0)

    eps_a = eps_b = 0.01
    s2_poly2 = sqrt(4*eps_a)
    s3_poly2 = -4 * eps_a
    s4_poly2 = - (eps_b + 12 * (eps_a ** 2)) / s2_poly2
    compute_and_print_coeffecients("poly2", s1=1, s2=s2_poly2, s3=s3_poly2, s4=s4_poly2, s5=0)