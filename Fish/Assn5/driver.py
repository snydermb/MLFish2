from Network import Network

def main():
    ## First tmp is initialized to a Network representation of the XOR problem.
    tmp = Network([[0,0,1], [0,1,1], [1,0,1], [1,1,1]], [[0], [1], [1], [0]])
    ## The gradient descent is performed with 1 hidden layer (so 3 total layers)
    ## and an alpha value
    tmp.SGD(3, 10, 50000)

if __name__ == "__main__":
    main()
