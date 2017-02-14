import DecisionTree

def main():
    file = open('Fish_trainset_1.csv')
    target = "Fish Species"
    data = [[]]
    for line in file:
        line = line.strip("\n")
        data.append(line.split(','))
    data.remove([])
    attributes = data[0]
    data.remove(attributes)
    print DecisionTree.makeTreeID3(data, attributes, target)

if __name__ == "__main__":
    main()
