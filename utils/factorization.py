def factorization(x):
    d = 2
    f = []

    while d <= x:
        if x % d == 0:
            f.append(d)
            x = x / d
        else:
            d = d + 1
    
    return f