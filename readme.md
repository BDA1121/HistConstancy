# Illuminant Spectral Direction Histogram Projection
Masters Project Under Prof Dr. Bruce A. Maxwell

Please also feel free to reach out with suggestions or feedback!

# Train/Test

for training 
```
python train.py --optimizer adamw --transform log --net color --batch_size 32 --epochs 100 --save_interval 10 --save_dir '' --resume ''
```
for testing
```
python testmodel.py --transform log --checkpoint ''
```