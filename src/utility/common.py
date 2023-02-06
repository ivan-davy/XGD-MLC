def prepareSet(test_set):
    counter = 0
    for spectrum in test_set:
        spectrum.rebin().calcCountRate()
        counter += 1
        print('\r', counter, '/', len(test_set), spectrum.path, end='')
    print('\n')
    return test_set


def flatten(lst):
    return [x for xs in lst for x in xs]


def linear(x, a, b):
    return a * x + b
