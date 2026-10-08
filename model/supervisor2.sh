#!/bin/bash
cd /app/conversations/6a1549534ed513da2a9fbd1b/cifar
while [ ! -f DATA_DONE ]; do
  python3 hf_dl.py >> hf_log.txt 2>&1
  sleep 2
done
echo "DATA READY — starting training" >> cifar_train_log.txt
while [ ! -f TRAIN_DONE ]; do
  python3 cifar_ddpm.py 55 >> cifar_train_log.txt 2>&1
  sleep 2
done
echo "ALL TRAINING COMPLETE" >> cifar_train_log.txt
python3 cifar_ddpm.py sample >> cifar_train_log.txt 2>&1
echo "SAMPLES GENERATED" >> cifar_train_log.txt
