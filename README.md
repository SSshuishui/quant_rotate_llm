# Rotation of LLMs


### A4W4KV16 RTN
```
python main.py --model (path)  --w_rtn --a_bits 4 --v_bits 16 --k_bits 16 --w_bits 4 --w_clip
```


### A4W4KV16 GPTQ
```
python main.py --model (path)  --a_bits 4 --v_bits 16 --k_bits 16 --w_bits 4 --w_clip
```

### A4W4KV16 RS
```
python main.py --model (path)  --a_bits 4 --v_bits 16 --k_bits 16 --w_bits 4 --w_clip --a_runtime_smooth
```

### A4W4KV16 QuaRot
```
python main.py --model (path)  --rotate --a_bits 4 --v_bits 16 --k_bits 16 --w_bits 4 --w_clip
```

### A4W4KV16 RRS
```
python main.py --model (path)  --rotate --a_bits 4 --v_bits 16 --k_bits 16 --w_bits 4 --w_clip --a_runtime_smooth --act_scale_g128
```