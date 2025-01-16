# PythonBackground.py
# Jacob Child

# Intro to object oriented programming, looking at [this](https://www.youtube.com/watch?v=q2SGW2VgwAM)
# A class is a blueprint that defines what attributes and methods (actions) an object can have

# If this file was a module, we could import it into another file and use the class
class Car: # so to import into other file we would say `from PythonBackground import Car`
    # This is a constructor, it is called when we create a new object
    def __init__(self, makef, modelf, yearf): # self is a reference to the object itself
        # attributes, what the object is or has
        self.make = makef
        self.model = modelf
        self.year = yearf
        self.odometer_reading = 0
        
    # Methods (actions) are functions that belong to a class
    def start(self): 
        print("this "+self.model+" is started")
        
    def stop(self):
        print("this car is stopped")
        
        
# Main code below 
# This is how we create an object from a class
my_car = Car("Toyota", "Corolla", 2020)
print(my_car.make)
my_car.start()
    