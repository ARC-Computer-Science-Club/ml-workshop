h = expit(np.matmul(features, thetas_h.transpose()))

y_hats = expit(np.matmul(h, thetas_y_hat))  # TODO reshape
y_hats = y_hats.reshape(y_hats.shape[0], 1)  # TODO remove

# Backpropagation
delta = labels - y_hats
gradient_y_hat = np.sum(delta * h, axis=0, keepdims=True)

t = h * (1 - h) * thetas_y_hat.transpose()
s = features * delta
for i in range(n):
    gradient_h += np.matmul(t[i][np.newaxis, :].transpose(), s[i][np.newaxis, :])





# Matrix version
# TODO fix this code and place it instead of the loops code

# Forward Pass
h = expit(np.matmul(features, thetas_h.transpose()))
y_hats = expit(np.matmul(h, thetas_y_hat))  # TODO reshape
y_hats = y_hats.reshape(y_hats.shape[0], 1)  # TODO remove

# Backpropagation
delta = labels - y_hats
gradient_y_hat = np.sum(delta * h, axis=0, keepdims=True)

t = h * (1 - h) * thetas_y_hat.transpose()
s = features * delta
for i in range(n):
    gradient_h += np.matmul(t[i][np.newaxis, :].transpose(), s[i][np.newaxis, :])

thetas_y_hat += rate * gradient_y_hat.transpose()
thetas_h += rate * gradient_h

LL = logLikelihood(labels, y_hats)
replot(fig, ax, line, nTimes, xdata, ydata, k, LL)