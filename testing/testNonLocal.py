def divide(x1, x2):
    try:
        z = x1/x2
        print(z)
    except:
        print('x2 should not be zero')
    finally:
        print('infinite')

divide(5, 0)
