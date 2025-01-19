class Restaurant:
    def __init__(self, restaurant_name, cuisine_type):
        self.restaurant_name = restaurant_name
        self.cuisine_type = cuisine_type
        self.number_served = 0

    def describe_restaurant(self):
        print("Restaurant Name:", self.restaurant_name)
        print("Cuisine Type:", self.cuisine_type)

    def open_restaurant(self):
        print("The restaurant", self.restaurant_name, "is now open.")

    def set_number_served(self, number):
        self.number_served = number

    def increment_number_served(self, increment):
        self.number_served += increment


class IceCreamStand(Restaurant):
    def __init__(self, restaurant_name, cuisine_type, flavors):
        super().__init__(restaurant_name, cuisine_type)
        self.flavors = flavors

    def display_flavors(self):
        print("Available Ice Cream Flavors:")
        for flavor in self.flavors:
            print("-", flavor)


# 创建Restaurant实例
restaurant = Restaurant("Good Eats", "Chinese")
print("Number of people served:", restaurant.number_served)
restaurant.number_served = 50
print("Number of people served:", restaurant.number_served)
restaurant.set_number_served(100)
print("Number of people served:", restaurant.number_served)
restaurant.increment_number_served(20)
print("Number of people served:", restaurant.number_served)

# 创建IceCreamStand实例
ice_cream_stand = IceCreamStand("Sweet Treats", "Dessert", ["Vanilla", "Chocolate", "Strawberry"])
ice_cream_stand.describe_restaurant()
ice_cream_stand.display_flavors()
