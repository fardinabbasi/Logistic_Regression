# Logistic Regression
Implementing logistic regression with **L2Regularization** **from scratch** to classify two **circular datasets**.

Since circular datasets are not **linearly separable**, it is necessary to map the **feature space** into higher dimensions. For instance, here is a **feature mapping** from 2 dimensions to 32 dimensions.

$X = [x_1,x_2]^T$

$f(X) = [x_1,x_2,x_1^2,x_1x_2,x_2^2,x_1^3,x_1^2x_2,x_1x_2^2,x_2^3,...,x_1x_2^6,x_2^7]^T; f: R^2 \rightarrow R^{35}$

## First Case
| Degree 1 | Degree 2 | Degree 3 |
| --- | --- | --- |
| <img src="/readme_images/a1.png"> | <img src="/readme_images/a2.png"> | <img src="/readme_images/a3.png"> |
| **Degree 4** | **Degree 5** | **Degree 6** |
| <img src="/readme_images/a4.png"> | <img src="/readme_images/a5.png"> | <img src="/readme_images/a6.png"> |
| **Degree 7** | **Degree 8** | **Degree 9** |
| <img src="/readme_images/a7.png"> | <img src="/readme_images/a8.png"> | <img src="/readme_images/a9.png"> |

| Degree 1 | Degree 2 | Degree 3 | Degree 4 | Degree 5 | Degree 6 | Degree 7 | Degree 8 | Degree 9 | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 63.3% | 77.5% | 76.6% | 76.6% | 76.6% | 76.6% | 76.6% | 77.5% | 77.5% |


## Second Case
| Degree 1 | Degree 2 | Degree 3 |
| --- | --- | --- |
| <img src="/readme_images/b1.png"> | <img src="/readme_images/b2.png"> | <img src="/readme_images/b3.png"> |
| **Degree 4** | **Degree 5** | **Degree 6** |
| <img src="/readme_images/b4.png"> | <img src="/readme_images/b5.png"> | <img src="/readme_images/b6.png"> |
| **Degree 7** | **Degree 8** | **Degree 9** |
| <img src="/readme_images/b7.png"> | <img src="/readme_images/b8.png"> | <img src="/readme_images/b9.png"> |

| Degree 1 | Degree 2 | Degree 3 | Degree 4 | Degree 5 | Degree 6 | Degree 7 | Degree 8 | Degree 9 | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 63.3% | 77.5% | 76.6% | 76.6% | 76.6% | 76.6% | 76.6% | 77.5% | 77.5% |
