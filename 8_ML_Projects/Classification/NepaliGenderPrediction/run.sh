#!/bin/bash
echo "Random Forest:"
echo "Random Forest, Fold 0"
python src/train.py --fold 0 --model random_forest
echo "Random Forest, Fold 1"
python src/train.py --fold 1 --model random_forest
echo "Random Forest, Fold 2"
python src/train.py --fold 2 --model random_forest
echo "Random Forest, Fold 3"
python src/train.py --fold 3 --model random_forest
echo "Random Forest, Fold 4"
python src/train.py --fold 4 --model random_forest
echo "Logistic Regression:"
echo "Logistic Regression, Fold 0"
python src/train.py --fold 0 --model logistic_regression
echo "Logistic Regression, Fold 1"
python src/train.py --fold 1 --model logistic_regression
echo "Logistic Regression, Fold 2"
python src/train.py --fold 2 --model logistic_regression
echo "Logistic Regression, Fold 3"
python src/train.py --fold 3 --model logistic_regression
echo "Logistic Regression, Fold 4"
python src/train.py --fold 4 --model logistic_regression
echo "Linear SVC:"
echo "Linear SVC, Fold 0"
python src/train.py --fold 0 --model linear_svc
echo "Linear SVC, Fold 1"
python src/train.py --fold 1 --model linear_svc
echo "Linear SVC, Fold 2"
python src/train.py --fold 2 --model linear_svc
echo "Linear SVC, Fold 3"
python src/train.py --fold 3 --model linear_svc
echo "Linear SVC, Fold 4"
python src/train.py --fold 4 --model linear_svc

echo "Multinomial NB:"
echo "Multinomial NB, Fold 0"
python src/train.py --fold 0 --model multinomial_nb
echo "Multinomial NB, Fold 1"
python src/train.py --fold 1 --model multinomial_nb
echo "Multinomial NB, Fold 2"
python src/train.py --fold 2 --model multinomial_nb
echo "Multinomial NB, Fold 3"
python src/train.py --fold 3 --model multinomial_nb
echo "Multinomial NB, Fold 4"
python src/train.py --fold 4 --model multinomial_nb
sleep 10