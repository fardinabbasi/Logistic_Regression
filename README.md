# Logistic Regression
Implementing logistic regression with **L2Regularization** **from scratch** to classify two **circular datasets**.

Since circular datasets are not **linearly separable**, it is necessary to map the **feature space** into higher dimensions. For instance, here is a **feature mapping** from 2 dimensions to 32 dimensions.

$X = [x_1,x_2]^T$

$f(X) = [x_1,x_2,x_1^2,x_1x_2,x_2^2,x_1^3,x_1^2x_2,x_1x_2^2,x_2^3,...,x_1x_2^6,x_2^7]^T; f: R^2 \rightarrow R^{35}$
Here is the implementation of logistic regression with **L2 regularization** built **from scratch**.
```ruby
class LogisticRegression():
    def __init__(self, degree, learning_rate, iterations, Lambda):
        self.degree = degree
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.Lambda = Lambda
        
    def transform(self, X):
        X_transformed = []
        x1 = X[:, 0].reshape(X.shape[0], 1)
        x2 = X[:, 1].reshape(X.shape[0], 1)
        for i in range(1, self.degree + 1):
            for j in range(0, i + 1):
                power_x1 = i - j
                power_x2 = j
                X_transformed.append((x1 ** power_x1) * (x2 ** power_x2))  
        return np.squeeze(np.array(X_transformed)).T
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def h_theta(self, X, theta):
        z = X.dot(theta)
        return self.sigmoid(z)

    def scale_features(self, X, mode='train'):
        if mode == 'train':
            self.mean = np.mean(X, axis = 0) 
            self.sd = np.std(X, axis = 0) 
        X_scaled = (X-self.mean)/self.sd
        return X_scaled
    
    def batch_gradient_descent(self):
        m = len(self.X_train)
        theta = np.zeros((self.X_train.shape[1], 1))
        for iteration in range(self.iterations):
            gradients = 1 / m * (self.X_train.T.dot(self.h_theta(self.X_train, theta) - self.y_train) + self.Lambda * theta)
            theta -= self.learning_rate * gradients
        return theta
    
    def fit(self, X_train, y_train):
        X_transformed = self.transform(X_train)
        X_scaled = self.scale_features(X_transformed)
        self.X_train = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))
        self.y_train = y_train
        self.theta = self.batch_gradient_descent()
        
    def predict(self, X_test):
        X_transformed = self.transform(X_test)
        X_scaled = self.scale_features(X_transformed, mode='test')
        X_test = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))
        return np.where(self.h_theta(X_test, self.theta) > 0.5, 1.0, 0.0)
```
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
