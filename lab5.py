import random
from scipy.stats import f, t
from prettytable import PrettyTable
import numpy as np
from datetime import datetime

x1_min = -6
x1_max = 1
x2_min = -4
x2_max = 4
x3_min = -2
x3_max = 7

average_x_max = (x1_max + x2_max + x3_max) / 3
average_x_min = (x1_min + x2_min + x3_min) / 3
y_max = int(200 + average_x_max)
y_min = int(200 + average_x_min)

x01 = (x1_max+x1_min)/2
x02 = (x2_max+x2_min)/2
x03 = (x3_max+x3_min)/2
delta_x1 = x1_max-x01
delta_x2 = x2_max-x02
delta_x3 = x3_max-x03

m = 3

X11 = [-1, -1, -1, -1, 1, 1, 1, 1, -1.215, 1.215, 0, 0, 0, 0, 0]
X22 = [-1, -1, 1, 1, -1, -1, 1, 1, 0, 0, -1.215, 1.215, 0, 0, 0]
X33 = [-1, 1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0, -1.215, 1.215, 0]

def sum_k_f2(x1, x2):
    xn = []
    for i in range(len(x1)):
        xn.append(round(x1[i] * x2[i],3))
    return xn


def sum_k_f3(x1, x2, x3):
    xn = []
    for i in range(len(x1)):
        xn.append(round(x1[i] * x2[i] * x3[i],3))
    return xn


def kv(x):
    xn = []
    for i in range(len(x)):
        xn.append(round(x[i] * x[i],3))
    return xn


X12 = sum_k_f2(X11, X22)
X13 = sum_k_f2(X11, X33)
X23 = sum_k_f2(X22, X33)
X123 = sum_k_f3(X11, X22, X33)
X1kv = kv(X11)
X2kv = kv(X22)
X3kv = kv(X33)

for i in range(1, m + 1):
    globals()['Y%s' % i] = [random.randrange(y_min, y_max, 1) for k in range(15)]


y1_av1, y2_av2, y3_av3, y4_av4, y5_av5, y6_av6, y7_av7, y8_av8, y9_av9, y10_av10, y11_av11, y12_av12, y13_av13, y14_av14, y15_av15 = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
for i in range(1, m + 1):
    for k in range(15):
        globals()['y%s_av%s' % (k + 1, k + 1)] += globals()['Y%s' % i][k]/m

y_av = []
for i in range(15):
    y_av.append(round(globals()['y%s_av%s' % (i+1, i+1)] ,3 ))

print("y=b0+b1*x1+b2*x2+b3*x3+b12*x1*x2+b13*x1*x3+b23*x2*x3+b123*x1*x2*x3+b11*x1^2+b22*x2^2+b33*x3^2")
table1 = PrettyTable()
table1.add_column("№", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
table1.add_column("X1", X11)
table1.add_column("X2", X22)
table1.add_column("X3", X33)
table1.add_column("X12", X12)
table1.add_column("X13", X13)
table1.add_column("X23", X23)
table1.add_column("X123", X123)
table1.add_column("X1^2", X1kv)
table1.add_column("X2^2", X2kv)
table1.add_column("X3^2", X3kv)
for i in range(1, m + 1):
    table1.add_column("Y" + str(i), globals()['Y%s' % i])
table1.add_column("Y", y_av)
print("Матриця планування експерименту для ОЦКП при k=3 із нормованими значеннями факторів наведена нижче")
print(table1)

X1 = [x1_min, x1_min, x1_min, x1_min, x1_max, x1_max, x1_max, x1_max, round(-1.215*delta_x1+x01,3), round(1.215*delta_x1+x01,3), x01, x01 ,x01 , x01, x01]
X2 = [x2_min, x2_min, x2_max, x2_max, x2_min, x2_min, x2_max, x2_max,  x02, x02, round(-1.215*delta_x2+x02,3), round(1.215*delta_x2+x02,3), x02, x02, x02]
X3 = [x3_min, x3_max, x3_min, x3_max, x3_min, x3_max, x3_min, x3_max, x03, x03, x03, x03, round(-1.215*delta_x3+x03,3), round(1.215*delta_x3+x03,3), x03]
X12 = sum_k_f2(X1, X2)
X13 = sum_k_f2(X1, X3)
X23 = sum_k_f2(X2, X3)
X123 = sum_k_f3(X1, X2, X3)
X1kv = kv(X1)
X2kv = kv(X2)
X3kv = kv(X3)

table2 = PrettyTable()
table2.add_column("№", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
table2.add_column("X1", X1)
table2.add_column("X2", X2)
table2.add_column("X3", X3)
table2.add_column("X12", X12)
table2.add_column("X13", X13)
table2.add_column("X23", X23)
table2.add_column("X123", X123)
table2.add_column("X1^2", X1kv)
table2.add_column("X2^2", X2kv)
table2.add_column("X3^2", X3kv)
for i in range(1, m + 1):
    table2.add_column("Y" + str(i), globals()['Y%s' % i])
table2.add_column("Y", y_av)
print("Матриця планування експерименту для ОЦКП при k=3 із натуралізованими значеннями факторів має вигляд:")
print(table2)


for i in range(15):
    globals()['d%s' % (i + 1)] = 0
for k in range(1, m + 1):
    for i in range(15):
        globals()['d%s' % (i + 1)] += ((globals()['Y%s' % (k)][i]) -  globals()['y%s_av%s' % (i + 1, i + 1)] ) ** 2/m

X0 =[1]*15

b = np.linalg.lstsq(list(zip(X0 , X1, X2, X3, X12, X13, X23, X123, X1kv, X2kv, X3kv)), y_av, rcond=None)[0]
b = [round(i , 3) for i in b]
print("\nКоефіцієти b:" ,b)
print("Перевірка:")
for i in range(15):
        print("y"+str(i+1)+"_av"+str(i+1)+" = "+str(round(b[0] + b[1]*X1[i]+b[2]*X2[i]+b[3]*X3[i]+b[4]*X1[i]*X2[i]+b[5]*X1[i]*X3[i]+b[6]*X2[i]*X3[i]+b[7]*X1[i]*X2[i]*X3[i]+b[8]*X1kv[i]+b[9]*X2kv[i]+b[10]*X3kv[i],3))+" = "+ str(round( globals()['y%s_av%s' % (i + 1, i + 1)],3)))
print()

dcouple = []
for i in range(15):
    dcouple.append(round(globals()['d%s' % (i+1)] ,3 ))


Gp = max(dcouple) / sum(dcouple)
q = 0.05
start_time = datetime.now()
f1 = m - 1
f2 = N = 15
fisher = f.isf(*[q / f2, f1, (f2 - 1) * f1])
Gt = round(fisher / (fisher + (f2 - 1)), 4)
print("Gp ="+str(Gp)+", Gt ="+str(Gt))
if Gp < Gt:
    print(f"\nТест Кохрена продовжувався: {(datetime.now() - start_time).total_seconds()} секунд\n")
    print("Дисперсія однорідна")
    print("Критерій Стьюдента")
    start_time = datetime.now()
    sb = sum(dcouple) / N
    ssbs = sb / N * m
    sbs = ssbs ** 0.5

    b_0 = (y1_av1*1+y2_av2*1+y3_av3*1+y4_av4*1+y5_av5*1+y6_av6*1+y7_av7*1+y8_av8*1+y9_av9*(-1.215)+y10_av10*1.215+y11_av11*0+y12_av12*0+y13_av13*0+y14_av14*0+y15_av15*0)/15
    b_1 = (y1_av1*(-1)+y2_av2*(-1)+y3_av3*(-1)+y4_av4*(-1)+y5_av5*1+y6_av6*1+y7_av7*1+y8_av8*1+y9_av9*0+y10_av10*0+y11_av11*(-1.215)+y12_av12*1.215+y13_av13*0+y14_av14*0+y15_av15*0)/15
    b_2 = (y1_av1*(-1)+y2_av2*(-1)+y3_av3*1+y4_av4*1+y5_av5*(-1)+y6_av6*(-1)+y7_av7*1+y8_av8*1+y9_av9*0+y10_av10*0+y11_av11*0+y12_av12*0+y13_av13*(-1.215)+y14_av14*1.215+y15_av15*0)/15
    b_3 = (y1_av1*(-1)+y2_av2*1+y3_av3*(-1)+y4_av4*1+y5_av5*(-1)+y6_av6*1+y7_av7*(-1)+y8_av8*1)/15
    b_4 = (y1_av1*1+y2_av2*1+y3_av3*(-1)+y4_av4*(-1)+y5_av5*(-1)+y6_av6*(-1)+y7_av7*1+y8_av8*1)/15
    b_5 = (y1_av1*1+y2_av2*(-1)+y3_av3*1+y4_av4*(-1)+y5_av5*(-1)+y6_av6*1+y7_av7*(-1)+y8_av8*1)/15
    b_6 = (y1_av1*1+y2_av2*(-1)+y3_av3*(-1)+y4_av4*1+y5_av5*1+y6_av6*(-1)+y7_av7*(-1)+y8_av8*1)/15
    b_7 = (y1_av1*(-1)+y2_av2*1+y3_av3*1+y4_av4*(-1)+y5_av5*1+y6_av6*(-1)+y7_av7*(-1)+y8_av8*1)/15
    b_8 = (y1_av1*1+y2_av2*1+y3_av3*1+y4_av4*1+y5_av5*1+y6_av6*1+y7_av7*1+y8_av8*1+y9_av9*1.46723+y10_av10*1.46723)/15
    b_9 = (y1_av1*1+y2_av2*1+y3_av3*1+y4_av4*1+y5_av5*1+y6_av6*1+y7_av7*1+y8_av8*1+y11_av11*1.46723+y12_av12*1.46723)/15
    b_10 = (y1_av1*1+y2_av2*1+y3_av3*1+y4_av4*1+y5_av5*1+y6_av6*1+y7_av7*1+y8_av8*1+y13_av13*1.46723+y14_av14*1.46723)/15

    f3 = f1 * f2
    ttabl = round(abs(t.ppf(q / 2, f3)), 4)

    d = 11
    for i in range(11):
        if ((abs(globals()['b_%s' % (i)]) / sbs) < ttabl):
            print("t%s < ttabl, b%s не значимий" % (i,i))
            globals()['b%s' % i ] = 0
            d = d - 1
    print(f"\nТест Стьюдента продовжувався: {(datetime.now() - start_time).total_seconds()} секунд\n")
    print("\nПеревірка в спрощене рівняння регресії:")
    for i in range(15):
        print("y"+str(i+1)+"_av"+str(i+1)+" = "+str(round(b[0] + b[1]*X1[i]+b[2]*X2[i]+b[3]*X3[i]+b[4]*X1[i]*X2[i]+b[5]*X1[i]*X3[i]+b[6]*X2[i]*X3[i]+b[7]*X1[i]*X2[i]*X3[i]+b[8]*X1kv[i]+b[9]*X2kv[i]+b[10]*X3kv[i],3))+" = "+ str(round( globals()['y%s_av%s' % (i + 1, i + 1)],3)))

    y_y1 = b[0]+b[1]*x1_min+b[2]*x2_min+b[3]*x3_min+b[4]*x1_min*x2_min+b[5]*x1_min*x3_min+b[6]*x2_min*x3_min+b[7]*x1_min*x2_min*x3_min+b[8]*x1_min*x1_min+b[9]*x2_min*x2_min+b[10]*x3_min*x3_min
    y_y2 = b[0]+b[1]*x1_min+b[2]*x2_min+b[3]*x3_max+b[4]*x1_min*x2_min+b[5]*x1_min*x3_max+b[6]*x2_min*x3_max+b[7]*x1_min*x2_min*x3_max+b[8]*x1_min*x1_min+b[9]*x2_min*x2_min+b[10]*x3_max*x3_max
    y_y3 = b[0]+b[1]*x1_min+b[2]*x2_max+b[3]*x3_min+b[4]*x1_min*x2_max+b[5]*x1_min*x3_min+b[6]*x2_max*x3_min+b[7]*x1_min*x2_max*x3_min+b[8]*x1_min*x1_min+b[9]*x2_max*x2_max+b[10]*x3_min*x3_min
    y_y4 = b[0]+b[1]*x1_min+b[2]*x2_max+b[3]*x3_max+b[4]*x1_min*x2_max+b[5]*x1_min*x3_max+b[6]*x2_max*x3_max+b[7]*x1_min*x2_max*x3_max+b[8]*x1_min*x1_min+b[9]*x2_max*x2_max+b[10]*x3_max*x3_max
    y_y5 = b[0]+b[1]*x1_max+b[2]*x2_min+b[3]*x3_min+b[4]*x1_max*x2_min+b[5]*x1_max*x3_min+b[6]*x2_min*x3_min+b[7]*x1_max*x2_min*x3_min+b[8]*x1_max*x1_max+b[9]*x2_min*x2_min+b[10]*x3_min*x3_min
    y_y6 = b[0]+b[1]*x1_max+b[2]*x2_min+b[3]*x3_max+b[4]*x1_max*x2_min+b[5]*x1_max*x3_max+b[6]*x2_min*x3_max+b[7]*x1_max*x2_min*x3_max+b[8]*x1_max*x1_max+b[9]*x2_min*x2_min+b[10]*x3_min*x3_max
    y_y7 = b[0]+b[1]*x1_max+b[2]*x2_max+b[3]*x3_min+b[4]*x1_max*x2_max+b[5]*x1_max*x3_min+b[6]*x2_max*x3_min+b[7]*x1_max*x2_min*x3_max+b[8]*x1_max*x1_max+b[9]*x2_max*x2_max+b[10]*x3_min*x3_min
    y_y8 = b[0]+b[1]*x1_max+b[2]*x2_max+b[3]*x3_max+b[4]*x1_max*x2_max+b[5]*x1_max*x3_max+b[6]*x2_max*x3_max+b[7]*x1_max*x2_max*x3_max+b[8]*x1_max*x1_max+b[9]*x2_max*x2_max+b[10]*x3_min*x3_max

    y_y9 = b[0]+b[1]*X1[8]+b[2]*X2[8]+b[3]*X3[8]+b[4]*X12[8]+b[5]*X13[8]+b[6]*X23[8]+b[7]*X123[8]+b[8]*X1kv[8]+b[9]*X2kv[8]+b[10]*X3kv[8]
    y_y10 = b[0]+b[1]*X1[9]+b[2]*X2[9]+b[3]*X3[9]+b[4]*X12[9]+b[5]*X13[9]+b[6]*X23[9]+b[7]*X123[9]+b[8]*X1kv[9]+b[9]*X2kv[9]+b[10]*X3kv[9]
    y_y11 = b[0]+b[1]*X1[10]+b[2]*X2[10]+b[3]*X3[10]+b[4]*X12[10]+b[5]*X13[10]+b[6]*X23[10]+b[7]*X123[10]+b[8]*X1kv[10]+b[9]*X2kv[10]+b[10]*X3kv[10]
    y_y12 = b[0]+b[1]*X1[11]+b[2]*X2[11]+b[3]*X3[11]+b[4]*X12[11]+b[5]*X13[11]+b[6]*X23[11]+b[7]*X123[11]+b[8]*X1kv[11]+b[9]*X2kv[11]+b[10]*X3kv[11]
    y_y13 = b[0]+b[1]*X1[12]+b[2]*X2[12]+b[3]*X3[12]+b[4]*X12[12]+b[5]*X13[12]+b[6]*X23[12]+b[7]*X123[12]+b[8]*X1kv[12]+b[9]*X2kv[12]+b[10]*X3kv[12]
    y_y14 = b[0]+b[1]*X1[13]+b[2]*X2[13]+b[3]*X3[13]+b[4]*X12[13]+b[5]*X13[13]+b[6]*X23[13]+b[7]*X123[13]+b[8]*X1kv[13]+b[9]*X2kv[13]+b[10]*X3kv[13]
    y_y15 = b[0]+b[1]*X1[14]+b[2]*X2[14]+b[3]*X3[14]+b[4]*X12[14]+b[5]*X13[14]+b[6]*X23[14]+b[7]*X123[14]+b[8]*X1kv[14]+b[9]*X2kv[14]+b[10]*X3kv[14]
    print("\nКритерій Фішера")
    start_time = datetime.now()
    print(d, " значимих коефіцієнтів")
    f4 = N - d
    sad = ((y_y1-y1_av1)**2+(y_y2-y2_av2)**2+(y_y3-y3_av3)**2+(y_y4-y4_av4)**2+(y_y5-y5_av5)**2+(y_y6-y6_av6)**2+(y_y7-y7_av7)**2+(y_y8-y8_av8)**2+ (y_y9-y9_av9)**2+(y_y10-y10_av10)**2+(y_y11-y11_av11)**2+(y_y12-y12_av12)**2+(y_y13-y13_av13)**2+(y_y14-y14_av14)**2+(y_y15-y15_av15)**2)*(m/(N-d))

    Fp = sad / sb
    print("Fp = ", round(Fp, 2))

    Ft = round(abs(f.isf(q, f4, f3)), 4)

    cont = 0
    if Fp > Ft:
        print(f"\nТест Фішера продовжувався: {(datetime.now() - start_time).total_seconds()} секунд\n")
        print("Fp =", round(Fp, 2), " > Ft", Ft, "\nРівняння неадекватно оригіналу")
        cont = 1
    else:
        print(f"\nТест Фішера продовжувався: {(datetime.now() - start_time).total_seconds()} секунд\n")
        print("Fp =", round(Fp, 2), " < Ft", Ft, "\nРівняння адекватно оригіналу")

else:
    print(f"\nТест Кохрена продовжувався: {(datetime.now() - start_time).total_seconds()} секунд\n")
    print("Дисперсія  неоднорідна")