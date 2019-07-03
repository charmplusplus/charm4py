from charm4py import charm, Chare, Group, Reducer

class MyChare(Chare):

    def work(self, data):
        self.reduce(self.thisProxy[0].collectResult, data, Reducer.sum)

    def collectResult(self, result):
        print("Result is", result)
        exit()

def main(args):
    my_group = Group(MyChare)
    my_group.work(3)

if __name__ == '__main__':
    charm.start(main)
