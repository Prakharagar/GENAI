#from __future__ import annotations
from abc import ABC,abstractmethod

class QUERY(ABC):
    @abstractmethod
    def query(self):
        print("abstract class")

class q1(QUERY):
    def query(self):
        print("child 1")

# class q2(QUERY):
#     def query(self):
#         print("child 2")

print(q1().query())

#print(q2().query())