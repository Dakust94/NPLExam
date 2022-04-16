from functions.hello import hello
from functions.novel import novel
#from functions.tensor import tensor

hello()

def main():
    try:
        novel()
    except NameError:
        print('Pas d éxécution des nouvelles')

main