# TF-net confidence domain


## Generating training data
```python3 pvalue.py training 0 | grep LOG```
## First round of training
```python3 run_model.py```
## Generating clipping data
```python3 clipping.py 0 | grep LOG```
## Second round of training
```python3 run_model.py clipping```

## Manim visualization: 
``` manimgl visualize.py VisualizeSigma -w```