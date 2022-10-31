cd src
python test.py --num_iter 10 --num_warmup 1 --device cuda --precision float16 --jit --nv_fuser --profile
