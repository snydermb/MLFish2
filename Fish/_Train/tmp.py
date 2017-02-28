import mnist_loader
import Network

def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    rates = [0.5, 1.0, 3.0, 5.0, 10.0, 30.0]
    batch = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    for b in batch:
        for r in rates:
            net = Network.Network([784, 25, 8])
            avg = net.SGD(training_data, 50, b, r, test_data=test_data)
            print "Mini Batch Size ", b, "  Learning Rate  =  ", r, " -- AVERAGE CORRECT", avg
        print "\n"
if __name__ == "__main__":
    main()
