# 0x02. Calculus
![](https://github.com/paurbano/holbertonschool-machine_learning/blob/master/images/calculus.jpg)

## General
* **Summation and Product notation**

  Summation = \sum
  Product = 

* **What is a series?**
  A series is the sum of all terms in a sequence e.i : 5+10+15+20

* **Common series**

  Finite & Infinite

* **What is a derivative?**

  the derivative is a way to show rate of change: that is, the amount by which a function is changing at one given point. it is the slope of the tangent line at a point on a graph. Is the "moment-by-moment" behavior of the function

* **What is the product rule?**

  the product rule is a formula used to find the derivatives of products of two or more functions

  $$(f*g)' = *f'*g + f*g'*$$

* **What is the chain rule?**

  The chain rule lets us "zoom into" a function and see how an initial change (x) can effect the final result down the line (g).

* **Common derivative rules**
    * Chain Rule
    * Product Rule
    * Power Rule

* **What is a partial derivative?**

  the partial derivative takes the derivative of certain indicated variables of a function and doesn't differentiate the other variable(s).

* **What is an indefinite integral?**
* **What is a definite integral?**
* **What is a double integral?**

## 0. Sigma is for Sum
<img src="https://latex.codecogs.com/gif.latex?\sum_{i=2}^{5}&amp;space;i">

1. 3 + 4 + 5
2. 3 + 4
3. 2 + 3 + 4 + 5
4. 2 + 3 + 4

## 1. The Greeks pronounce it sEEgma
<img src="https://latex.codecogs.com/gif.latex?\sum_{k=1}^{4}&amp;space;9i&amp;space;-&amp;space;2k">

1. 90 - 20
2. 36i - 20
3. 90 - 8k
4. 36i - 8k

## 2. Pi is for Product
<img src="https://latex.codecogs.com/gif.latex?\prod_{i&amp;space;=&amp;space;1}^{m}&amp;space;i">

1. (m - 1)!
2. 0
3. (m + 1)!
4. m!

## 3. The Greeks pronounce it pEE
<img src="https://latex.codecogs.com/gif.latex?\prod_{i&amp;space;=&amp;space;0}^{10}&amp;space;i">

1. 10!
2. 9!
3. 100
4. 0

## 4. Hello, derivatives!
<img src="https://latex.codecogs.com/gif.latex?\frac{dy}{dx}"> where <img src="https://latex.codecogs.com/gif.latex?y&amp;space;=&amp;space;x^4&amp;space;+&amp;space;3x^3&amp;space;-&amp;space;5x&amp;space;+&amp;space;1">
 
 1. <img src="https://latex.codecogs.com/gif.latex?3x^3&amp;space;+&amp;space;6x^2&amp;space;-4">
 2. <img src="https://latex.codecogs.com/gif.latex?4x^3&amp;space;+&amp;space;6x^2&amp;space;-&amp;space;5">
 3. <img src="https://latex.codecogs.com/gif.latex?4x^3&amp;space;+&amp;space;9x^2&amp;space;-&amp;space;5">
 4. <img src="https://latex.codecogs.com/gif.latex?4x^3&amp;space;+&amp;space;9x^2&amp;space;-&amp;space;4">

## 5. A log on the fire
<img src="https://latex.codecogs.com/gif.latex?\frac{d&amp;space;(xln(x))}{dx}">

1. <img src="https://latex.codecogs.com/gif.latex?ln(x)">
2. <img src="https://latex.codecogs.com/gif.latex?\frac{1}{x} + 1">
3. <img src="https://latex.codecogs.com/gif.latex?ln(x) + 1">
4. <img src="https://latex.codecogs.com/gif.latex?\frac{1}{x}">

## 6. It is difficult to free fools from the chains they revere
<img src="https://latex.codecogs.com/gif.latex?\frac{d&amp;space;(ln(x^2))}{dx}">

1. <img src="https://latex.codecogs.com/gif.latex?\frac{2}{x}">
2. <img src="https://latex.codecogs.com/gif.latex?\frac{1}{x^2}">
3. <img src="https://latex.codecogs.com/gif.latex?\frac{2}{x^2}">
4. <img src="https://latex.codecogs.com/gif.latex?4x^3&amp;space;+&amp;space;9x^2&amp;space;-&amp;space;4">

## 7. Partial truths are often more insidious than total falsehoods
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial&amp;space;y}&amp;space;f(x,&amp;space;y)"> where <img src="https://latex.codecogs.com/gif.latex?f(x,&amp;space;y)&amp;space;=&amp;space;e^{xy}">

1. <img src="https://latex.codecogs.com/gif.latex?e^{xy}">
2. <img src="https://latex.codecogs.com/gif.latex?ye^{xy}">
3. <img src="https://latex.codecogs.com/gif.latex?xe^{xy}">
4. <img src="https://latex.codecogs.com/gif.latex?e^{x}">

## 8. Put it all together and what do you get?
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial^2}{\partial&amp;space;y\partial&amp;space;x}(e^{x^2y})">where <img src="https://latex.codecogs.com/gif.latex?\frac{\partial&amp;space;x}{\partial&amp;space;y}=\frac{\partial&amp;space;y}{\partial&amp;space;x}=0">

1. <img src="https://latex.codecogs.com/gif.latex?2x(1+y)e^{x^2y}">
2. <img src="https://latex.codecogs.com/gif.latex?xe^{xy}">
3. <img src="https://latex.codecogs.com/gif.latex?2x(1+x^2y)e^{x^2y}">
4. <img src="https://latex.codecogs.com/gif.latex?e^{2x}">

## 9. Our life is the sum total of all the decisions we make every day, and those decisions are determined by our priorities
Write a function `def summation_i_squared(n):` that calculates <img src="https://latex.codecogs.com/gif.latex?\sum_{i=1}^{n}&amp;space;i^2">:

* `n` is the stopping condition
* Return the integer value of the sum
* If `n` is not a valid number, return `None`
* You are not allowed to use any loops

```
alexa@ubuntu:0x02-calculus$ cat 9-main.py 
#!/usr/bin/env python3

summation_i_squared = __import__('9-sum_total').summation_i_squared

n = 5
print(summation_i_squared(n))
alexa@ubuntu:0x02-calculus$ ./9-main.py 
55
alexa@ubuntu:0x02-calculus$
```

## 10. Derive happiness in oneself from a good day's work 
Write a function `def poly_derivative(poly):` that calculates the derivative of a polynomial:

* `poly` is a list of coefficients representing a polynomial
    * the index of the list represents the power of `x` that the coefficient belongs to
    * Example: if f(x) = x^3 + 3x +5, `poly` is equal to `[5, 3, 0, 1]`
* If `poly` is not valid, return `None`
* If the derivative is `0`, return `[0]`
* Return a new list of coefficients representing the derivative of the polynomial

```
alexa@ubuntu:0x02-calculus$ cat 10-main.py 
#!/usr/bin/env python3

poly_derivative = __import__('10-matisse').poly_derivative

poly = [5, 3, 0, 1]
print(poly_derivative(poly))
alexa@ubuntu:0x02-calculus$ ./10-main.py 
[3, 0, 3]
alexa@ubuntu:0x02-calculus$
```

## 11. Good grooming is integral and impeccable style is a must
<img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/6/ada047ad4cbee23dfed8.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200726%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20200726T040210Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=f912a8096803080a3643fd19ce17d095b1e9dee610c98035f883ce2e2a1d6761" alt="" style="">

1. 3x^2 + C
2. x^4/4 + C
3. x^4 + C
4. x^4/3 + C

## 12. We are all an integral part of the web of life
<img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/6/9ed107b0dcdde8dd49ac.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200726%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20200726T040210Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=72bd243ee601167d9b3f44552297febcadf0aadb99ee88c451dbbd040104373b" alt="" style="">

1. e^2y + C
2. e^y + C
3. e^2y/2 + C
4. e^y/2 + C

## 13. Create a definite plan for carrying out your desire and begin at once
<img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/6/b94ec3cf3ae61acd0275.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&amp;X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200726%2Fus-east-1%2Fs3%2Faws4_request&amp;X-Amz-Date=20200726T040210Z&amp;X-Amz-Expires=86400&amp;X-Amz-SignedHeaders=host&amp;X-Amz-Signature=fd13e5adb912ebc6314f0c14198b416768b1094600585dd2df18fa57326dc44d" alt="" style="">

1. 3
2. 6
3. 9
4. 27

## 14. My talents fall within definite limitations

1. -1
2. 0
3. 1
4. undefined

## 15. Winners are people with definite purpose in life

1. 5
2. 5x
3. 25
4. 25x

## 16. Double whammy

1. 9ln(2)
2. 9
3. 27ln(2)
4. 27

## 17. Integrate 
Write a function `def poly_integral(poly, C=0):` that calculates the integral of a polynomial:

* `poly` is a list of coefficients representing a polynomial
    * the index of the list represents the power of `x` that the coefficient belongs to
    * Example: if f(x) = x^3 + 3x +5, `poly` is equal to `[5, 3, 0, 1]`
* `C` is an integer representing the integration constant
* If a coefficient is a whole number, it should be represented as an integer
* If `poly` or `C` are not valid, return `None`
* Return a new list of coefficients representing the integral of the polynomial
* The returned list should be as small as possible

```
alexa@ubuntu:0x02-calculus$ cat 17-main.py 
#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral

poly = [5, 3, 0, 1]
print(poly_integral(poly))
alexa@ubuntu:0x02-calculus$ ./17-main.py 
[0, 5, 1.5, 0, 0.25]
alexa@ubuntu:0x02-calculus$
```
