"""
created on 2025/2/2, 18:05
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description:

"""
import multiprocessing as mp

# save all Fibonacci Sequence
fibNumbers = []


def fibonacciCalc(num):
    """ Calculate for find the Fibonacci code for the number asigned """
    if num <= 2:
        fibNumbers.append(1)
        return 1

    else:
        # calculate the fibonacci code according to a index
        fibValue = fibNumbers[num-2] + fibNumbers[num-3]
        fibNumbers.append(fibValue)
        return fibValue


def factorization(fib):
    """ Numeric factorization for a fibonacci number """
    print(f"initializing factorization for {fib}")

    divisor = 2
    values = []

    # save unfactorizable values
    if fib <= 5:
        values.append(str(fib))

    else:
        while fib != 1:

            # finding divisors
            if fib % divisor == 0:
                values.append(str(divisor))
                fib = fib/divisor

            else:
                divisor += 1
    return values


def unicode_exp(exp):
    """ Get a exponent number for html format
        https://es.stackoverflow.com/questions/316023/imprimir-exponente-en-python """

    # get different values for unitary exponenets (from 0 to 9)
    if exp == 1:
        return chr(0xB9)

    if exp == 2 or exp == 3:
        return chr(0xB0 + exp)

    else:
        return chr(0x2070 + exp)


def potenciaFormatter(values):
    """ A kind of Reduce() function is created as in Mapreduce() to obtain a
        python dictionary with key-value pairs where the key is unique and the
        value is the number of occurrences of the key in the original array """

    # if there is only 1 value
    if len(values) == 1:
        return values

    cont = 0
    current = values[0]
    dictionary = dict()

    # for all values iterate to get the number of occurrences for each key. "Reduce()"
    for val in values:
        if(val == current):
            cont += 1

        else:
            dictionary.setdefault(current, cont)
            current = val
            cont = 1
    dictionary.setdefault(current, cont)

    # the dictionary for the reduce result
    formattedValues = []
    keys = dictionary.keys()

    # save a number with exponent format only if is the exponent is greater than 1
    for key in keys:
        formattedValues.append(str(key)+unicode_exp(dictionary.get(key))) \
            if dictionary.get(key) > 1 \
            else formattedValues.append(str(key))

    return formattedValues


def app(LIMIT):
    """ Principal function to alculate fibonacci series from 1 to LIMIT """
    print('***************** Fibonacci Sequence *****************')

    # list with numbers to evalue
    valuesRange = list(range(LIMIT+1))[1:]

    # Init multiprocessing.Pool() for calculate fibonacci series
    with mp.Pool(mp.cpu_count()) as pool:
        # calculate the fibonacci for the current value i
        savesFib = pool.map(fibonacciCalc, [i for i in valuesRange])

    print("Fibonacci values has finished its calculation")

    # Init multiprocessing.Pool() for calculate factorization
    with mp.Pool(mp.cpu_count()) as pool:
        # get a array with all values to a fibonacci value
        factorValues = pool.map_async(
            factorization, [i for i in savesFib]).get()

    print("Fibonacci factorization has finished its calculation")

    # Init multiprocessing.Pool() for calculate exponents in factors
    with mp.Pool(mp.cpu_count()) as pool:
        # make a string with formated factorizacion
        formattedValues = pool.map_async(
            potenciaFormatter, [i for i in factorValues]).get()

    print("Calculate of exponents in factorsfactorization has finished")

    # print of results
    for i in valuesRange:
        currentLine = str(
            i) + ' : ' + str(savesFib[i-1]) + ' = ' + ' x '.join(formattedValues[i-1])
        print(currentLine)

    print('Finalization..')


if __name__ == "__main__":
    """ Main execution"""

    # max numer to calculate fibonacci series
    LIMIT = 40

    # Calculate fibonacci series from 1 to LIMIT
    app(LIMIT)