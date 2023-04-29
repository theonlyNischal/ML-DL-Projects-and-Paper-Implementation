#!/bin/bash
echo "Random Forest:"
python src/train.py --fold 0 --model random_forest
python src/train.py --fold 1 --model random_forest
python src/train.py --fold 2 --model random_forest
python src/train.py --fold 3 --model random_forest
python src/train.py --fold 4 --model random_forest
echo "Logistic Regression:"
python src/train.py --fold 0 --model logistic_regression
python src/train.py --fold 1 --model logistic_regression
python src/train.py --fold 2 --model logistic_regression
python src/train.py --fold 3 --model logistic_regression
python src/train.py --fold 4 --model logistic_regression
echo "Linear SVC:"
python src/train.py --fold 0 --model linear_svc
python src/train.py --fold 1 --model linear_svc
python src/train.py --fold 2 --model linear_svc
python src/train.py --fold 3 --model linear_svc
python src/train.py --fold 4 --model linear_svc
sleep 10