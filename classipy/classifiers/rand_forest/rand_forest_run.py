def data_generator(num_points):
    # Here we make a few fake classes and see if the classifier can get it
    cgens = [[(.2, .4), (0, 1)], [(.3, .6), (0, 1)], [(0., 1.), (.4, .5)]]
    print(cgens)
    out = []
    for x in range(num_points):
        label = random.randint(0, len(cgens) - 1)
        value = [np.random.uniform(x, y) for x, y in cgens[label]]
        out.append((label, value))
    return out


def main():
    label_values = data_generator(1000)
    dims = [(0., 1.), (0., 1.)]
    train(label_values, dims)
    #val = make_feature([(0, 1), (3, 5)])
    #print(val([.5, 4.]))

if __name__ == '__main__':
    main()
