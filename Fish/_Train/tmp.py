import mnist_loader
import Network

def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network.Network([784, 20, 8])
    net.SGD(training_data, 50, 10, 3.0, test_data=test_data)

if __name__ == "__main__":
    main()
