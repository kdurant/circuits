# 解线性方程组
```py
x, y, t = symbols("x y t")
print(solve([x + y - 2, x - y], [x, y]))
```

# 解符号方程组
```py
x, y, a, b = symbols("x y a b")
print(solve([x + y - a, x - y + b], [x, y]))
```
# 一重定积分
```py
integrate(x**2 + exp(x) + 1, (x, 0, 1))
integrate(x, (x, 0, 1))
```

# 一重不定积分
```py
print(integrate(x**2 + exp(x) + 1, x))
print(integrate(x, (x)))
pprint(Integral(x, t))
```

# 微分
```py
print(diff(x**2, (x)))
pprint(Derivative(cos(x), x))
```

# 求解微分方程
```py
t = symbols("t")
v = Function('v')

# 根据KVL,列写开关打开后的电路微分方程
# eq = 20 - 2* diff(v(t), t) * 4 - v(t) 
eq = 20 - 2* v(t).diff(t) * 4 - v(t) 


# 换路时刻前，电容的电压作为微分方程的初始值
ics = {v(0) : 4}
pprint(dsolve(eq, ics = ics))
```

# 傅立叶级数
```py
s = fourier_series(x, (x, -pi, pi))
pprint(s)
```

# 拉普拉斯变换
```py
t, s, a = symbols("t s a")
laplace_transform(t**4, t, s)
```

# 数列求和
```py
x, a = symbols("x a")
Sum(2 ** x, (x, 1, a)).doit()
```

# 复数相关
## 返回复数实数部分
```py
re(2*I + 17)
```

## 返回复数的虚数部分
```py
im(2*I + 17)
```

## 复数的模
```py
abs(I+1)
abs(sqrt(2)*I+sqrt(2))
```

## 共轭复数
```py
conjugate(1 - I)
```

## 计算角度值
```py
import math
cos(math.radians(50))
```

## 弧度转换为角度
```py
math.degrees(math.pi/2)  
```

## 角度转换为弧度
```py
math.radians(180)
```

# 特殊函数图像
## 表示分段函数
```py
p, v, i, t, C = symbols("p v i t C")
v = Piecewise((5*t, And(t > 0, t <= 2)), (-5*(t-4), And(t > 2, t <= 6)), (5*(t-8), And(t > 6, t < 8)) )
plot(v, diff(v, t))
```

## 阶跃函数
```py
x = symbols("x")
plot(Heaviside(x), Heaviside(x)*x, (x,0, 5))
plot(Heaviside(x-1) - Heaviside(x-5))
```

##  狄拉克里函数
```py
plot(DiracDelta(x))
```

## 绘制转移函数的bode图
```py
from sympy.physics.control.lti import TransferFunction
from sympy.physics.control.control_plots import bode_plot
s = symbols("s")

tf1 = TransferFunction(5, s**2 + 8*s + 5, s)
# tf1 = TransferFunction(1*s**2 + 0.1*s + 7.5, 1*s**4 + 0.12*s**3 + 9*s**2, s)
bode_plot(tf1, phase_unit="deg")  
```

# 矩阵相关
```py
from sympy import Matrix

A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])
C = A + B
```

# 向量相关
## 创建向量
```py
from sympy.vector import CoordSys3D
N = CoordSys3D('N')   # 定义一个三维坐标系
v = 1*N.i + 2*N.j + 3*N.k  # 定义一个三维向量
w = 4*N.i + 5*N.j + 6*N.k  # 定义另一个三维向量
```

## 向量加减法
```py
u = v + w  # 计算向量和
u = v - w  # 计算向量差 
```
## 向量点积
```py
u.dot(v)
```

## 向量叉积
```py
u.cross(v)
```

## 向量模长
```py
u.magnitude()
```

## 向量夹角
```py
cos_angle = v.dot(w) / (v.magnitude() * w.magnitude())  # 计算夹角的余弦值
angle = sympy.acos(cos_angle)  # 计算夹角
angle_deg = sympy.deg(angle)  # 将弧度制夹角转换为度数制
```

# 梯度，散度，旋度

## 梯度
```py
from sympy import symbols, diff

x, y, z = symbols('x y z')
f = x**2*y + y*z**3

grad_f = [diff(f, var) for var in [x, y, z]]
grad_f  # Output the gradient as a list
```

## 散度
```py
from sympy import symbols, Matrix, diff

x, y, z = symbols('x y z')
F = Matrix([2*x, 3*y**2, z**3])

div_F = diff(F[0], x) + diff(F[1], y) + diff(F[2], z)
div_F.simplify()  # Simplify the result
```

## 旋度
```py
from sympy import symbols, Matrix, diff

x, y, z = symbols('x y z')
F = Matrix([x**2, y**2, z**2])

curl_F = Matrix([diff(F[2], y) - diff(F[1], z),
                 diff(F[0], z) - diff(F[2], x),
                 diff(F[1], x) - diff(F[0], y)])
curl_F.simplify()  # Simplify the result
```
