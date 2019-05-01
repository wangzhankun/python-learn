class Product():
    def __init__(self,id,name,price,yieldly):
        self.id = id
        self.name = name
        self.price = price
        self.yieldly = yieldly


class Cart():
    def __init__(self):
        self.product = {}
        self.products = []
        self.total_price = 0

    def buy_product(self,product,quantity):
        self.product['name'] = product.name
        self.product['quantity'] = quantity
        self.product['price'] = product.price
        self.products.append(self.product)
    
    def delete_product(self,product,quantity):
        if product in self.products[][name]:
            if quantity >= self.products[product][quantity]:
                self.products.remove[product]
            else:
                self.products[product][quantity] -= quantity
        else:
            pass

    def cal_total_price(self):
        for i in self.products:
            self.total_price += self.products[product][quantity] * self.products[product][price]
        return self.total_price

class Account():
    def __init__(self):
        self.username = ''
        self.passwd = ''
        self.cart = ''
    
    def create_account(self,username,passwd):
        self.username = username
        self.passwd = passwd
        self.cart = Cart()
    
    def login(self,username,passwd):
        if username != self.username:
            print("Username Error!")
        else:
            if passwd != self.passwd:
                print("Passwd Error!")
            else:
                print("Log In!")


