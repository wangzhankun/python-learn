#彩色螺旋线
import turtle
turtle.speed(1)
turtle.pensize(2)
turtle.bgcolor("black")
colors = ["red","yellow","purple","blue"]
turtle.tracer(False)
for x in range(800):
    turtle.forward(2*x)
    turtle.color(colors[x % 4])
    turtle.left(89)
turtle.tracer(True)
turtle.done()