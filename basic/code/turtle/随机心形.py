import turtle as t
import random as r
def pink():
    color = (1, r.random(), 1)
    return color

def randomrange(min, max):
    return min + (max- min)*r.random()

def moveto(x, y):
    t.penup()
    t.goto(x, y)
    t.pendown()

def heart(r, a):
    factor = 180
    t.seth(a)
    t.circle(-r, factor)
    t.fd(2 * r)
    t.right(90)
    t.fd(2 * r)
    t.circle(-r, factor)

t.setup(800, 800, 200, 200)
t.speed(9)
t.pensize(1)
t.penup()

for i in range(20):
    t.goto(randomrange(-300, 300), randomrange(-300, 300))
    t.begin_fill()
    t.fillcolor(pink())
    heart(randomrange(10, 50), randomrange(0, 90))
    t.end_fill()

moveto(400, -400)
t.done()