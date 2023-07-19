# Logistic Regression
Implementing logistic regression with **L2Regularization** **from scratch** to classify two **circular datasets**.

Since circular datasets are not **linearly separable**, it is necessary to map the **feature space** into higher dimensions. For instance, here is a **feature mapping** from 2 dimensions to 32 dimensions.

$X = [x_1,x_2]^T$

$f(X) = [x_1,x_2,x_1^2,x_1x_2,x_2^2,x_1^3,x_1^2x_2,x_1x_2^2,x_2^3,...,x_1x_2^6,x_2^7]^T; f: R^2 \rightarrow R^{35}$
