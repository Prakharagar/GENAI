class TEST:
    def __init__(self,a,b):
        self.a=a
        self.b=b
    '''
    Property Turn a Method into an Attribute
    '''
    @property
    def multiply(self):
        return self.a * self.b
    
    @classmethod
    def sumnum(cls,c,d):
        return cls(c,d)

    @staticmethod
    def is_valid(num:int)->bool:
        if num < 0:
            return False
        else:
            return True
    

#print(TEST(3,5).multiply)

#print(TEST(2,3).is_valid(-1))

t=TEST.sumnum(3,4)
print(t)