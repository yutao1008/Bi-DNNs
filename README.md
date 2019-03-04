# Bi-DNNs
Various DNN layers implemented by bilinear projections
The codes are based on Tensorflow ver 1.12.

These layers can be used in a similar way with the layers defined in `tensorflow.python.keras.layers`.
  
Note that the bilinear projection is implemented in a `slow` way, because the flatten and unflatten in each 
layer are essentially unnecessary. However, the Tensorflow library does not provide the basic operators on
tensors with 2D structures, e.g., batch-normalization and max-pooling. 

