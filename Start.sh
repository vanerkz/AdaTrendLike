export CUDA_VISIBLE_DEVICES=0
#python main.py --num_epochs 100  --batch_size  64 --mode train --dataset MSL  --data_path dataset/MSL 

#python main.py --num_epochs 100  --batch_size 64 --mode train --dataset SMAP  --data_path dataset/SMAP

python main.py --num_epochs 100    --batch_size 64 --mode train --dataset PSM  --data_path dataset/PSM

#python main.py --num_epochs 100 --batch_size  64 --mode train --dataset SWaT  --data_path dataset/SWAT

#python main.py --num_epochs 100   --batch_size 64  --mode train --dataset SMD  --data_path dataset/SMD 


#python main.py --num_epochs 100  --batch_size  64 --mode train --dataset UCR  --data_path dataset/UCR

#python main.py --num_epochs 100  --batch_size  64 --mode train --dataset WADI  --data_path dataset/WADI